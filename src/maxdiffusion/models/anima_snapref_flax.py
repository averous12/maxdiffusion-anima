"""SnapRef reference adapter (format `anima_ref_kv/1`) for the Flax Anima transformer.

Semantics are those of the reference ComfyUI node (0ppxnhximxr/snapref-v1a,
comfyui/snapref/__init__.py), re-expressed for this port:

* the reference image is VAE-encoded, put through the transformer *once* at sigma 0 with its
  own LoRA stream ("ref"), at RoPE position `ref_t` and centred on the target's token grid;
  every block's self-attention (k, v) is cached;
* each target forward prepends those cached keys/values in self-attention with a per-key
  additive log bias (strength + log weight_i), and runs with the "tgt" LoRA stream;
* `mode` (cs / char) is a learned offset added to the thin timestep embedding of the
  reference pass only (the node adds it after t_embedding_norm).

The weights are LoRA deltas applied functionally at inference (`_proj` in
anima_cosmos_flax.py): merging them into the 4 GB param tree per stream would need a second
and third copy of every weight, and this port must not do that.

Nothing in here runs unless a request sets `snapref`: every entry point is opt-in.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

SNAPREF_FORMAT = "anima_ref_kv/1"

# adapter module name -> path inside one block's LoRA tree.
# The tree mirrors the model: {"attn1": {dense name: (A, B)}, "attn2": {...},
#                               "ff_in": {"self": (A, B)}, "ff_out": {...}}
_ATTN_TARGETS = {
    "self_attn.q_proj": ("attn1", "to_q"),
    "self_attn.k_proj": ("attn1", "to_k"),
    "self_attn.v_proj": ("attn1", "to_v"),
    "self_attn.output_proj": ("attn1", "to_out"),
    "cross_attn.q_proj": ("attn2", "to_q"),
    "cross_attn.k_proj": ("attn2", "to_k"),
    "cross_attn.v_proj": ("attn2", "to_v"),
    "cross_attn.output_proj": ("attn2", "to_out"),
}
_MLP_TARGETS = {"mlp.layer1": "ff_in", "mlp.layer2": "ff_out"}

# Anima-2.9B inserts these blocks into the 28-block base, shifting its original layers.
_INSERTED_29B = frozenset((2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36))

# in-features of each target, for a cheap sanity check against the loaded checkpoint.
_EXPECTED_IN = {"to_q": 2048, "to_k": 2048, "to_v": 2048, "to_out": 2048, "ff_in": 2048, "ff_out": 8192}


class SnapRefError(ValueError):
  """Raised when an adapter file or setting is not usable."""


def adapter_block_map(adapter_layers, model_layers):
  """Which model block each adapter block belongs to (28 -> 40 for Anima-2.9B), as in the node."""
  indices = set(int(i) for i in adapter_layers)
  if not indices:
    raise SnapRefError("adapter has no diffusion block weights")
  if max(indices) + 1 == model_layers:
    return tuple(range(model_layers))
  if model_layers == 40 and indices == set(range(28)):
    return tuple(i for i in range(40) if i not in _INSERTED_29B)
  raise SnapRefError(f"adapter covers {max(indices) + 1} blocks but the model has {model_layers}; "
                     "only the native layout and the 28-to-40 Anima-2.9B expansion are supported")


@dataclass
class SnapRefAdapter:
  """One loaded `anima_ref_kv/1` file, in the shapes the Flax model wants."""

  path: str
  ref_t: int = 10
  modes: tuple = ("cs",)
  mode_embed: Optional[jax.Array] = None
  layers: int = 28
  lora: dict = field(default_factory=dict)   # stream -> {path tuple: (A, B)} stacked per layer
  meta: dict = field(default_factory=dict)

  def stream_names(self):
    return tuple(sorted(self.lora))

  def lora_tree(self, stream):
    """The per-layer LoRA argument for one stream, or None if the stream is absent.

    Stacked arrays with a leading axis of `layers` — the shape nn.scan consumes directly.
    """
    return self.lora.get(stream) or None

  def mode_index(self, mode):
    try:
      return self.modes.index(mode)
    except ValueError:
      return None

  def mode_vector(self, mode):
    """mode_embed row for `mode`, or None (unknown mode = the trained default behaviour)."""
    idx = self.mode_index(mode)
    if idx is None or self.mode_embed is None:
      return None
    return self.mode_embed[idx]

  def describe(self, stream=None):
    out = [f"format={self.meta.get('format')} version={self.meta.get('version')} "
           f"step={self.meta.get('step')} ref_t={self.ref_t} modes={list(self.modes)}"]
    for name in ([stream] if stream else self.stream_names()):
      tree = self.lora.get(name) or {}
      leaves = jax.tree_util.tree_leaves(tree)
      shapes = [tuple(x.shape) for x in leaves[:2]]
      out.append(f"  stream {name}: {len(leaves) // 2} matrices, stacked leaf shapes {shapes}")
    return "\n".join(out)


def _dequant(tensors, key):
  """bf16 matrix; FP8 files carry a per-row float32 <key>_scale."""
  w = tensors.get_tensor(key).astype(np.float32)
  if key + "_scale" in tensors.keys():
    w = w * tensors.get_tensor(key + "_scale").astype(np.float32)
  return w


def load_snapref(path, num_layers=28, dtype=jnp.float32, expect_layers=None):
  """Load an `anima_ref_kv/1` adapter into (A, B) LoRA pairs + the mode embedding."""
  with safe_open(path, framework="np") as handle:
    meta = handle.metadata() or {}
    raw = json.loads(meta.get("snapref") or meta.get("anima_ref_kv") or "{}")
    if raw.get("format") != SNAPREF_FORMAT:
      raise SnapRefError(f"{path} is not an {SNAPREF_FORMAT} adapter (metadata: {raw.get('format')!r})")
    keys = set(handle.keys())
    mode_embed = None
    if "anima_ref_kv.mode_embed" in keys:
      mode_embed = jnp.asarray(handle.get_tensor("anima_ref_kv.mode_embed"), dtype=jnp.float32)
    # stream -> path -> {layer: (A, B)}
    acc = {}
    for key in keys:
      if not key.endswith(".lora_down"):
        continue
      root = key[: -len("lora_down")]          # "anima_ref_kv.<stream>.blocks.<i>.<module>."
      parts = root[:-1].split(".", 2)          # ["anima_ref_kv", stream, "blocks.<i>.<module>"]
      if len(parts) != 3 or not parts[2].startswith("blocks."):
        continue
      stream = parts[1]
      _, index, module = parts[2].split(".", 2)
      down = _dequant(handle, key)             # torch (rank, in)
      up = _dequant(handle, key[: -len("lora_down")] + "lora_up")   # torch (out, rank)
      alpha = float(handle.get_tensor(root + "alpha")[()]) if root + "alpha" in keys else down.shape[0]
      scale = alpha / down.shape[0]
      # torch Linear convention -> Flax: A (in, rank) = down.T * scale, B (rank, out) = up.T
      A = (down.T.astype(np.float32) * scale)
      B = up.T.astype(np.float32)
      acc.setdefault(stream, {}).setdefault(_flax_path(module), {})[int(index)] = (A, B)

  adapter_layers = set()
  for tree in acc.values():
    for per_layer in tree.values():
      adapter_layers |= set(per_layer)
  if expect_layers is not None and adapter_layers and len(adapter_layers) != expect_layers:
    raise SnapRefError(f"adapter has {len(adapter_layers)} blocks, expected {expect_layers}")

  lora = {}
  for stream, tree in acc.items():
    out = {}
    for path_key, per_layer in tree.items():
      missing = sorted(set(range(num_layers)) - set(per_layer))
      if missing:
        raise SnapRefError(f"stream {stream} {path_key}: missing layers {missing[:5]}")
      A = jnp.asarray(np.stack([per_layer[i][0] for i in range(num_layers)]), dtype=dtype)
      B = jnp.asarray(np.stack([per_layer[i][1] for i in range(num_layers)]), dtype=dtype)
      _check_shapes(path_key, A, B)
      if len(path_key) == 1:                       # ff_in / ff_out
        out[path_key[0]] = {"self": (A, B)}
      else:
        out.setdefault(path_key[0], {})[path_key[1]] = (A, B)
    lora[stream] = out

  return SnapRefAdapter(path=path, ref_t=int(raw.get("ref_t", 10)),
                        modes=tuple(raw.get("modes", ["cs"])), mode_embed=mode_embed,
                        layers=num_layers, lora=lora, meta=raw)


def _flax_path(module):
  if module in _ATTN_TARGETS:
    return _ATTN_TARGETS[module]
  if module in _MLP_TARGETS:
    return (_MLP_TARGETS[module],)
  raise SnapRefError(f"adapter targets an unknown module: {module}")


def _check_shapes(path_key, A, B):
  name = path_key[-1]
  want = _EXPECTED_IN.get(name)
  if want is not None and A.shape[-2] != want:
    raise SnapRefError(f"adapter target {'.'.join(path_key)} expects in_features {want}, got {A.shape[-2]}")


def reference_biases(strength, weights):
  """Per-reference log bias on the reference attention keys (node: bias + log(w))."""
  return [float(strength) + math.log(float(w)) for w in weights]


def key_biases(sizes, biases, target_len, extra=0.0):
  """The additive bias for every key of the extended sequence: refs first, then target keys."""
  out = []
  for size, bias in zip(sizes, biases):
    out.append(np.full(int(size), float(bias) + float(extra), dtype=np.float32))
  out.append(np.zeros(int(target_len), dtype=np.float32))
  return np.concatenate(out)


def ref_token_axes(target_tokens_hw, ref_tokens_hw, ref_t):
  """RoPE coordinates of the reference tokens: temporal `ref_t`, centred on the target grid."""
  th, tw = int(target_tokens_hw[0]), int(target_tokens_hw[1])
  rh, rw = int(ref_tokens_hw[0]), int(ref_tokens_hw[1])
  h0 = int(round(th / 2 - rh / 2))
  w0 = int(round(tw / 2 - rw / 2))
  h = np.repeat(np.arange(h0, h0 + rh), rw)
  w = np.tile(np.arange(w0, w0 + rw), rh)
  t = np.full(rh * rw, int(ref_t), dtype=np.int32)
  return t, h, w


def reference_rope(target_tokens_hw, ref_tokens_hw, ref_t, head_dim=128, rope_scale=(1.0, 4.0, 4.0)):
  """cos/sin for the reference stream (see cosmos_rope_from_positions)."""
  from maxdiffusion.models.anima_cosmos_flax import cosmos_rope_from_positions

  t, h, w = ref_token_axes(target_tokens_hw, ref_tokens_hw, ref_t)
  return cosmos_rope_from_positions(jnp.asarray(t), jnp.asarray(h), jnp.asarray(w),
                                    head_dim, rope_scale)


def reference_latent_size(width, height, megapixels):
  """Reference pixel size: shrunk (never enlarged) to `megapixels`, sides multiples of 16."""
  scale = min(1.0, math.sqrt((float(megapixels) * 1024.0 * 1024.0) / float(width * height)))
  return max(16, int(width * scale) // 16 * 16), max(16, int(height * scale) // 16 * 16)


def uncond_bias_for_fidelity(fidelity, ref_in_uncond=False, ref_guidance_on=False):
  """log(1 - f) for the negative branch, or None when the setting does not apply (node semantics)."""
  f = float(fidelity)
  if f >= 1.0 or ref_in_uncond or ref_guidance_on:
    return None
  return math.log(max(1.0 - f, 1e-6))


def guidance_active(sigma, ref_end):
  """ref_guidance runs while sigma >= percent_to_sigma(ref_end); ComfyUI's flow mapping is 1 - p."""
  if ref_end >= 1.0:
    return True
  return float(sigma) >= (1.0 - float(ref_end)) - 1e-5


def ref_guidance_combine(cond_ref, uncond_ref, uncond_base, guidance, ref_guidance):
  """guided = base + rg * (uncond_ref - base) + cfg * (cond_ref - uncond_ref).

  Algebraically what the node's post-CFG hook produces; base is the negative branch without
  the reference (the one extra model call per step until ref_end).
  """
  return uncond_base + float(ref_guidance) * (uncond_ref - uncond_base) \
      + float(guidance) * (cond_ref - uncond_ref)


def snapref_correct(guided, pos, correction="apg", rescale_phi=0.5):
  """APG / CFG-rescale on the denoised (x0) predictions — port of the node's _correct()."""
  dims = tuple(range(1, guided.ndim))
  if correction == "apg":
    pos64 = pos.astype(jnp.float64)
    update = (guided.astype(jnp.float64) - pos64)
    flat = pos64.reshape(pos64.shape[0], -1)
    norm = jnp.maximum(jnp.linalg.norm(flat, axis=1), 1e-12)
    unit = (flat / norm[:, None]).reshape(pos64.shape)
    proj = jnp.sum(update * unit, axis=dims, keepdims=True)
    return (pos64 + (update - proj * unit)).astype(guided.dtype)
  if correction == "rescale":
    ratio = pos.std(axis=dims, keepdims=True) / guided.std(axis=dims, keepdims=True)
    return rescale_phi * guided * ratio + (1.0 - rescale_phi) * guided
  return guided


def concat_reference_kv(records):
  """[(k, v) per layer] for several references -> one cached set, concatenated on the token axis."""
  if len(records) == 1:
    return records[0]
  return [(jnp.concatenate([r[i][0] for r in records], axis=1),
           jnp.concatenate([r[i][1] for r in records], axis=1))
          for i in range(len(records[0]))]


def reference_sizes(records):
  """Token count of each reference, in the order they were encoded."""
  return [int(r[0][0].shape[1]) for r in records]


def encode_reference_latent(vae, image_chw, latents_mean, latents_std):
  """VAE-encode one reference image into the sampler's latent space.

  image_chw: (1, 3, 1, H, W) float32 in [-1, 1] (the same convention decode_latents inverts).
  Follows diffusers' Anima encoder: latents = (mode(encode(x)) - mean) / std, with the
  distribution's *mode* so the same picture always gives the same latent.
  """
  from flax import nnx

  from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWanCache

  graphdef, state, rest = nnx.split(vae, nnx.Param, ...)
  merged = nnx.merge(graphdef, state, rest)
  posterior = merged.encode(jnp.asarray(image_chw), AutoencoderKLWanCache(merged),
                            return_dict=False)[0]
  z = posterior.mode()
  # The Anima/Wan VAE encodes to channels-LAST (B, T, H, W, C) — measured: a (1,3,1,64,64)
  # image comes back as (1,1,8,8,16). The sampler, the transformer's patchify and the
  # per-layer K/V cache all work in (B, C, T, H, W), so transpose here, once. Without this
  # the reference pass reads C=1,T=H,H=W,W=C and patchifies to the wrong token count.
  n_ch = jnp.asarray(latents_mean).shape[0]
  if z.ndim == 5 and z.shape[1] != n_ch and z.shape[-1] == n_ch:
    z = jnp.transpose(z, (0, 4, 1, 2, 3))
  mean = jnp.asarray(latents_mean, dtype=z.dtype).reshape(1, -1, 1, 1, 1)
  std = jnp.asarray(latents_std, dtype=z.dtype).reshape(1, -1, 1, 1, 1)
  return (z - mean) / std, merged
