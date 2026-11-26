"""Anima ControlNet-LLLite (3D pose / outline / regional) for the Flax Anima transformer.

Faithful port of ComfyUI's reference — `comfy/ldm/anima/lllite.py` plus the hook site in
`comfy/ldm/cosmos/predict2.py`. Every contract below was read out of those files or the
checkpoint metadata, not guessed:

  * **Where it hooks.** `Attention.compute_qkv` (predict2.py:152-202) runs the `attn1_patch`
    on `q_input`/`k_input`/`v_input` *before* `q_proj`/`k_proj`/`v_proj` are applied
    (lines 160-179). So LLLite edits the **hidden states pre-projection**, not the projected
    tensors. That is what the checkpoint's `lllite.target_atomics =
    self_attn_q_pre,self_attn_kv_pre` means, and it is why tensors can still be named
    `..._self_attn_q_proj` — the `_proj` suffix names the *projection being patched*, not the
    tensor being edited. The module's `in_dim` equals the model width, which is only
    consistent pre-projection.
  * **`self_attn_kv_pre`** is satisfied by one `apply()` per target: in self-attention the
    context is `x`, so the k input and the v input are each edited separately.
  * **`mlp_layer1`** hooks the block's MLP input the same way (predict2.py:589-591).
  * **Conditioning** runs once per forward: bicubic-resize the control image to latent x 8,
    clamp to [0,1], map to [-1,1]; a 4-channel (inpaint) checkpoint then appends the mask as
    `mask*2-1` and may zero the masked pixels first. The 3D-pose checkpoints are 3-channel.

Implemented functionally over the torch-named tensors (kernels transposed once at load) rather
than as Flax modules: the naming and layout mapping stay explicit, and the parity test against
PyTorch is then exact. Nothing is merged into the base parameters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

LLLITE_FORMAT = "anima-lllite/2"
MODULE_PATTERN = re.compile(
    r"lllite_dit_blocks_(\d+)_(self_attn_[qkv]_proj|cross_attn_q_proj|mlp_layer1)$")

_SUFFIX_TO_TARGET = {
    "self_attn_q_proj": "self_attn_q",
    "self_attn_k_proj": "self_attn_k",
    "self_attn_v_proj": "self_attn_v",
    "cross_attn_q_proj": "cross_attn_q",
    "mlp_layer1": "mlp_layer1",
}


class LLLiteError(ValueError):
  """Raised when an LLLite checkpoint or setting cannot be used."""


def _md(md, key, default=None):
  if not md:
    return default
  v = md.get(key)
  return default if v is None else v


def silu(x):
  return x * jax.nn.sigmoid(x)


# ---------------------------------------------------------------------------
# config + loading
# ---------------------------------------------------------------------------

@dataclass
class LLLiteConfig:
  cond_in_channels: int
  cond_dim: int
  cond_emb_dim: int
  cond_resblocks: int
  aspp_dilations: Tuple[int, ...]
  mlp_dim: int
  model_dim: int
  inpainting: bool
  blocks: Dict[int, Tuple[str, ...]] = field(default_factory=dict)
  metadata: Dict[str, str] = field(default_factory=dict)

  @property
  def block_count(self) -> int:
    return (max(self.blocks) + 1) if self.blocks else 0


@dataclass
class LLLite:
  """Loaded LLLite.

  `cond` holds the conditioning-net tensors and `blocks[(block, target)]` the per-module
  tensors, both in compute-ready layout: conv kernels as HWIO (torch stores OIHW) and dense
  kernels as (in, out) (torch stores (out, in)). Norms hold scale/bias.
  """
  config: LLLiteConfig
  cond: Dict[str, jnp.ndarray]
  blocks: Dict[Tuple[int, str], Dict[str, jnp.ndarray]]

  def has(self, block_index: int, target: str) -> bool:
    return (block_index, target) in self.blocks


def _to_hwio(w: np.ndarray) -> jnp.ndarray:
  """torch Conv2d (out, in, kh, kw) -> JAX (kh, kw, in, out)."""
  if w.ndim != 4:
    raise LLLiteError(f"expected a 4-D conv weight, got shape {w.shape}")
  return jnp.asarray(np.transpose(w, (2, 3, 1, 0)))


def _to_io(w: np.ndarray) -> jnp.ndarray:
  """torch Linear (out, in) -> JAX (in, out)."""
  if w.ndim != 2:
    raise LLLiteError(f"expected a 2-D linear weight, got shape {w.shape}")
  return jnp.asarray(w.T)


def config_from_handle(handle) -> LLLiteConfig:
  """Derive the contract (shapes, version, block map) without loading the weights."""
  keys = list(handle.keys())
  md = dict(handle.metadata() or {})
  version = _md(md, "lllite.version", "2")
  if version != "2":
    raise LLLiteError(f"Unsupported Anima LLLite version {version!r}; only v2 named-key "
                      f"checkpoints are supported")
  module_names = sorted({k.split(".", 1)[0] for k in keys if k.startswith("lllite_dit_blocks_")})
  if not module_names:
    raise LLLiteError("Anima LLLite checkpoint has no lllite_dit_blocks_* modules")
  for required in ("lllite_conditioning1.conv1.weight", "lllite_conditioning1.conv3.weight",
                   "lllite_conditioning1.proj.weight"):
    if required not in keys:
      raise LLLiteError(f"checkpoint is missing {required}")

  cond_in_channels = int(handle.get_slice("lllite_conditioning1.conv1.weight").get_shape()[1])
  cond_dim = int(handle.get_slice("lllite_conditioning1.conv3.weight").get_shape()[0])
  cond_emb_dim = int(handle.get_slice("lllite_conditioning1.proj.weight").get_shape()[0])
  rb = {int(k.split(".")[2]) for k in keys if k.startswith("lllite_conditioning1.resblocks.")}
  cond_resblocks = max(rb) + 1 if rb else 0
  use_aspp = any(k.startswith("lllite_conditioning1.aspp.") for k in keys)
  aspp: Tuple[int, ...] = ()
  if use_aspp:
    raw = _md(md, "lllite.aspp_dilations", "1,2,4,8")
    aspp = tuple(int(v) for v in raw.split(",") if v.strip())

  md_emb = _md(md, "lllite.cond_emb_dim")
  if md_emb is not None and int(md_emb) != cond_emb_dim:
    raise LLLiteError(f"metadata cond_emb_dim {md_emb} != proj out channels {cond_emb_dim}")

  model_dim = mlp_dim = None
  blocks: Dict[int, list] = {}
  for name in module_names:
    m = MODULE_PATTERN.fullmatch(name)
    if m is None:
      raise LLLiteError(f"Unsupported Anima LLLite module name: {name}")
    idx, suffix = int(m.group(1)), m.group(2)
    out_ch, in_ch = handle.get_slice(f"{name}.down.weight").get_shape()
    if model_dim is None:
      model_dim, mlp_dim = int(in_ch), int(out_ch)
    elif model_dim != int(in_ch):
      raise LLLiteError(f"Anima LLLite model dim mismatch in {name}: {in_ch} != {model_dim}")
    film_in = handle.get_slice(f"{name}.cond_to_film.weight").get_shape()[1]
    if int(film_in) != cond_emb_dim:
      raise LLLiteError(f"Anima LLLite cond dim mismatch in {name}: {film_in} != {cond_emb_dim}")
    mid_out, mid_in = handle.get_slice(f"{name}.mid.weight").get_shape()
    if int(mid_in) != int(mlp_dim) + cond_emb_dim:
      raise LLLiteError(f"Anima LLLite mid input {mid_in} != mlp_dim+cond_emb_dim "
                        f"({mlp_dim}+{cond_emb_dim})")
    up_out, up_in = handle.get_slice(f"{name}.up.weight").get_shape()
    # torch stores Linear as (out_features, in_features): up maps mlp_dim -> model_dim
    if int(up_in) != mlp_dim or int(up_out) != model_dim:
      raise LLLiteError(f"Anima LLLite up-projection shape mismatch in {name}: "
                        f"got ({up_out}, {up_in}), expected ({model_dim}, {mlp_dim})")
    blocks.setdefault(idx, []).append(_SUFFIX_TO_TARGET[suffix])

  return LLLiteConfig(
      cond_in_channels=cond_in_channels, cond_dim=cond_dim, cond_emb_dim=cond_emb_dim,
      cond_resblocks=cond_resblocks, aspp_dilations=aspp,
      mlp_dim=int(mlp_dim), model_dim=int(model_dim),
      inpainting=_md(md, "lllite.inpaint_masked_input", "false").lower() == "true",
      blocks={k: tuple(sorted(v)) for k, v in sorted(blocks.items())},
      metadata=md,
  )


def _read(handle, key, keyset) -> Optional[np.ndarray]:
  return handle.get_tensor(key) if key in keyset else None


def _norm(handle, prefix, ks):
  w = _read(handle, f"{prefix}.weight", ks)
  b = _read(handle, f"{prefix}.bias", ks)
  if w is None:
    raise LLLiteError(f"missing {prefix}.weight")
  return {"scale": jnp.asarray(w),
          "bias": jnp.asarray(b if b is not None else np.zeros(w.shape[0], np.float32))}


def _conv_pair(handle, prefix, ks, out_ch):
  w = _read(handle, f"{prefix}.weight", ks)
  if w is None:
    raise LLLiteError(f"missing {prefix}.weight")
  b = _read(handle, f"{prefix}.bias", ks)
  return {"kernel": _to_hwio(w),
          "bias": jnp.asarray(b if b is not None else np.zeros(out_ch, np.float32))}


def load_lllite(path: str, dtype=jnp.float32) -> LLLite:
  """Load an Anima LLLite safetensors file into compute-ready arrays."""
  from safetensors import safe_open
  with safe_open(path, "np") as handle:
    cfg = config_from_handle(handle)
    ks = set(handle.keys())
    half = cfg.cond_dim // 2
    cond: Dict[str, jnp.ndarray] = {}

    for name, out_ch in (("conv1", half), ("conv2", half), ("conv3", cfg.cond_dim)):
      cond.update({f"{name}.{k}": v for k, v in
                   _conv_pair(handle, f"lllite_conditioning1.{name}", ks, out_ch).items()})
    for name in ("norm1", "norm2", "norm3"):
      cond.update({f"{name}.{k}": v for k, v in
                   _norm(handle, f"lllite_conditioning1.{name}", ks).items()})
    cond.update({f"proj.{k}": v for k, v in
                 _conv_pair(handle, "lllite_conditioning1.proj", ks, cfg.cond_emb_dim).items()})
    cond.update({f"out_norm.{k}": v for k, v in
                 _norm(handle, "lllite_conditioning1.out_norm", ks).items()})

    for i in range(cfg.cond_resblocks):
      p = f"lllite_conditioning1.resblocks.{i}"
      for name in ("norm1", "norm2"):
        cond.update({f"rb{i}.{name}.{k}": v for k, v in _norm(handle, f"{p}.{name}", ks).items()})
      for name in ("conv1", "conv2"):
        cond.update({f"rb{i}.{name}.{k}": v for k, v in
                     _conv_pair(handle, f"{p}.{name}", ks, cfg.cond_dim).items()})

    if cfg.aspp_dilations:
      p = "lllite_conditioning1.aspp"
      for i in range(len(cfg.aspp_dilations)):
        cond.update({f"aspp.b{i}.{k}": v for k, v in
                     _conv_pair(handle, f"{p}.branches.{i}.0", ks, cfg.cond_dim).items()})
        cond.update({f"aspp.b{i}.gn.{k}": v for k, v in
                     _norm(handle, f"{p}.branches.{i}.1", ks).items()})
      cond.update({f"aspp.g.{k}": v for k, v in
                   _conv_pair(handle, f"{p}.global_conv.0", ks, cfg.cond_dim).items()})
      cond.update({f"aspp.g.gn.{k}": v for k, v in
                   _norm(handle, f"{p}.global_conv.1", ks).items()})
      cond.update({f"aspp.proj.{k}": v for k, v in
                   _conv_pair(handle, f"{p}.proj.0", ks, cfg.cond_dim).items()})
      cond.update({f"aspp.proj.gn.{k}": v for k, v in
                   _norm(handle, f"{p}.proj.1", ks).items()})

    blocks: Dict[Tuple[int, str], Dict[str, jnp.ndarray]] = {}
    for idx, targets in cfg.blocks.items():
      for target in targets:
        suffix = {v: k for k, v in _SUFFIX_TO_TARGET.items()}[target]
        name = f"lllite_dit_blocks_{idx}_{suffix}"
        mod: Dict[str, jnp.ndarray] = {}
        for layer in ("down", "mid", "cond_to_film", "up"):
          w = _read(handle, f"{name}.{layer}.weight", ks)
          b = _read(handle, f"{name}.{layer}.bias", ks)
          if w is None:
            raise LLLiteError(f"missing {name}.{layer}.weight")
          mod[f"{layer}.kernel"] = _to_io(w)
          mod[f"{layer}.bias"] = jnp.asarray(
              b if b is not None else np.zeros(w.shape[0], np.float32))
        de = _read(handle, f"{name}.depth_embed", ks)
        if de is None:
          raise LLLiteError(f"missing {name}.depth_embed")
        mod["depth_embed"] = jnp.asarray(de)
        blocks[(idx, target)] = mod

  cast = lambda a: jnp.asarray(a, dtype)
  return LLLite(config=cfg,
                cond=jax.tree_util.tree_map(cast, cond),
                blocks={k: jax.tree_util.tree_map(cast, v) for k, v in blocks.items()})


# ---------------------------------------------------------------------------
# conditioning network (control image -> per-token embedding)
# ---------------------------------------------------------------------------

def _groups_for(channels: int) -> int:
  g = 8
  while g > 1 and channels % g != 0:
    g //= 2
  return g


def _group_norm(x, scale, bias, groups: int, eps: float = 1e-5):
  """GroupNorm over NHWC, equivalent to torch GroupNorm over NCHW."""
  b, h, w, c = x.shape
  if c % groups != 0:
    raise LLLiteError(f"GroupNorm: {c} channels not divisible by {groups} groups")
  xr = x.reshape(b, h, w, groups, c // groups)
  mean = jnp.mean(xr, axis=(1, 2, 4), keepdims=True)
  var = jnp.var(xr, axis=(1, 2, 4), keepdims=True)
  xr = ((xr - mean) / jnp.sqrt(var + eps)).reshape(b, h, w, c)
  return xr * scale.reshape(1, 1, 1, c) + bias.reshape(1, 1, 1, c)


def _conv(x, kernel, bias, stride: int = 1, dilation: int = 1, same: bool = False):
  """2-D convolution over NHWC; `same=True` pads to keep the spatial size."""
  kh, kw, cin, cout = kernel.shape
  if x.shape[-1] != cin:
    raise LLLiteError(f"conv channel mismatch: input {x.shape[-1]} vs kernel {cin}")
  if same:
    ph = dilation * (kh - 1) // 2
    pw = dilation * (kw - 1) // 2
    x = jnp.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
  y = jax.lax.conv_general_dilated(
      x, kernel, window_strides=(stride, stride), padding="VALID",
      rhs_dilation=(dilation, dilation), dimension_numbers=("NHWC", "HWIO", "NHWC"))
  return y + bias.reshape(1, 1, 1, cout)


def _resblock(x, p, prefix: str, channels: int):
  g = _groups_for(channels)
  h = silu(_group_norm(x, p[f"{prefix}.norm1.scale"], p[f"{prefix}.norm1.bias"], g))
  h = _conv(h, p[f"{prefix}.conv1.kernel"], p[f"{prefix}.conv1.bias"], same=True)
  h = silu(_group_norm(h, p[f"{prefix}.norm2.scale"], p[f"{prefix}.norm2.bias"], g))
  h = _conv(h, p[f"{prefix}.conv2.kernel"], p[f"{prefix}.conv2.bias"], same=True)
  return x + h


def _aspp(x, p, cfg: LLLiteConfig):
  h, w = x.shape[1], x.shape[2]
  g = _groups_for(cfg.cond_dim)
  outs = []
  for i, dilation in enumerate(cfg.aspp_dilations):
    if dilation == 1:
      b = _conv(x, p[f"aspp.b{i}.kernel"], p[f"aspp.b{i}.bias"])
    else:
      b = _conv(x, p[f"aspp.b{i}.kernel"], p[f"aspp.b{i}.bias"], dilation=dilation, same=True)
    outs.append(silu(_group_norm(b, p[f"aspp.b{i}.gn.scale"], p[f"aspp.b{i}.gn.bias"], g)))
  gp = _conv(jnp.mean(x, axis=(1, 2), keepdims=True), p["aspp.g.kernel"], p["aspp.g.bias"])
  gp = silu(_group_norm(gp, p["aspp.g.gn.scale"], p["aspp.g.gn.bias"], g))
  outs.append(jax.image.resize(gp, (gp.shape[0], h, w, gp.shape[3]), method="bilinear"))
  out = _conv(jnp.concatenate(outs, axis=-1), p["aspp.proj.kernel"], p["aspp.proj.bias"])
  return silu(_group_norm(out, p["aspp.proj.gn.scale"], p["aspp.proj.gn.bias"], g))


def encode_conditioning(lllite: LLLite, image_nhwc, eps: float = 1e-5):
  """Control image (B,H,W,C) in [-1,1] -> embedding (B, (H/16)*(W/16), cond_emb_dim)."""
  cfg, p = lllite.config, lllite.cond
  if image_nhwc.shape[-1] != cfg.cond_in_channels:
    raise LLLiteError(f"control image has {image_nhwc.shape[-1]} channels but the checkpoint "
                      f"expects {cfg.cond_in_channels}")
  half = cfg.cond_dim // 2
  x = _conv(image_nhwc, p["conv1.kernel"], p["conv1.bias"], stride=4)
  x = silu(_group_norm(x, p["norm1.scale"], p["norm1.bias"], _groups_for(half), eps))
  x = _conv(x, p["conv2.kernel"], p["conv2.bias"], same=True)
  x = silu(_group_norm(x, p["norm2.scale"], p["norm2.bias"], _groups_for(half), eps))
  x = _conv(x, p["conv3.kernel"], p["conv3.bias"], stride=4)
  x = silu(_group_norm(x, p["norm3.scale"], p["norm3.bias"], _groups_for(cfg.cond_dim), eps))
  for i in range(cfg.cond_resblocks):
    x = _resblock(x, p, f"rb{i}", cfg.cond_dim)
  if cfg.aspp_dilations:
    x = _aspp(x, p, cfg)
  x = _conv(x, p["proj.kernel"], p["proj.bias"])
  b, h, w, c = x.shape
  x = x.reshape(b, h * w, c)
  mean = jnp.mean(x, axis=-1, keepdims=True)
  var = jnp.var(x, axis=-1, keepdims=True)
  x = (x - mean) / jnp.sqrt(var + eps)
  return x * p["out_norm.scale"].reshape(1, 1, -1) + p["out_norm.bias"].reshape(1, 1, -1)


# ---------------------------------------------------------------------------
# per-block application
# ---------------------------------------------------------------------------

def apply_module_params(p, x, cond_emb, strength):
  """LLLite edit with the module weights passed in directly.

  Split out from `apply_module` so the same math can run from a stacked per-layer scan
  input, where the layer's weights arrive as a slice rather than being looked up by index.
  """
  if p is None:
    return x
  b, s, _d = x.shape
  cb, cs, _cd = cond_emb.shape
  if cb != b:
    if b % cb != 0:
      raise LLLiteError(f"Anima LLLite batch mismatch: model input batch {b}, "
                        f"control batch {cb}")
    cond_emb = jnp.repeat(cond_emb, b // cb, axis=0)
  if cs != s:
    raise LLLiteError(f"Anima LLLite sequence mismatch: model input has {s} tokens, "
                      f"control has {cs}")

  cond_local = cond_emb + p["depth_embed"]
  hidden = silu(x @ p["down.kernel"] + p["down.bias"])
  film = cond_local @ p["cond_to_film.kernel"] + p["cond_to_film.bias"]
  gamma, beta = jnp.split(film, 2, axis=-1)
  hidden = jnp.concatenate([cond_local, hidden], axis=-1)
  # `mid` is linear here — the reference applies its single activation AFTER the FiLM
  # modulation, so adding one after mid as well double-activates and drifts the output.
  hidden = hidden @ p["mid.kernel"] + p["mid.bias"]
  hidden = silu(hidden * (1 + gamma) + beta)
  return x + (hidden @ p["up.kernel"] + p["up.bias"]) * strength


def apply_module(lllite: LLLite, x, cond_emb, block_index: int, target: str, strength: float):
  """One module's edit, addressed by (block, target) — the unstacked/CPU path."""
  return apply_module_params(lllite.blocks.get((block_index, target)), x, cond_emb, strength)


def stack_lllite_weights(lllite: LLLite, num_layers: Optional[int] = None):
  """Fold the per-block modules into per-layer stacked leaves, the layout nn.scan wants.

  Same idiom as the SnapRef LoRA tree and `stack_block_params`: every leaf gains a leading
  axis of length num_layers, so the weights ride through the scanned body as a scan input
  and only one copy exists. Returns {target: {param_name: (num_layers, ...)}}.
  """
  n = int(num_layers or lllite.config.block_count)
  targets = sorted({t for (_b, t) in lllite.blocks})
  out = {}
  for target in targets:
    rows = [lllite.blocks.get((i, target)) for i in range(n)]
    present = [r for r in rows if r is not None]
    if not present:
      continue
    if len(present) != n:
      raise LLLiteError(f"Anima LLLite target {target!r} is missing for "
                        f"{n - len(present)} of {n} blocks; cannot stack it for the scan")
    keys = set(present[0])
    for r in present[1:]:
      if set(r) != keys:
        raise LLLiteError(f"Anima LLLite target {target!r} has inconsistent parameter keys")
    out[target] = {k: jnp.stack([r[k] for r in present], axis=0) for k in sorted(keys)}
  return out


def attention_inputs_w(lllite_w, x, cond_emb, strength, is_self: bool, context=None):
  """`attention_inputs` against a per-layer weight dict (the nn.scan path).

  Same hook points, but the layer's module weights arrive as a slice of the stacked tree
  rather than being looked up by block index — which is what lets LLLite ride the scan.
  """
  ctx = x if context is None else context
  w = lllite_w or {}

  def edit(target):
    return apply_module_params(w.get(target), x, cond_emb, strength)

  if not is_self:
    return edit("cross_attn_q"), ctx, ctx
  return edit("self_attn_q"), edit("self_attn_k"), edit("self_attn_v")


def attention_inputs(lllite: LLLite, x, cond_emb, block_index: int, strength: float,
                     is_self: bool, context=None):
  """The (q_in, k_in, v_in) the projections should consume.

  The reference patches these before `q_proj`/`k_proj`/`v_proj` (predict2.py:160-179). In
  self-attention all three start as `x`; in cross-attention only q comes from `x`.
  """
  ctx = x if context is None else context
  if not is_self:
    return apply_module(lllite, x, cond_emb, block_index, "cross_attn_q", strength), ctx, ctx
  q_in = apply_module(lllite, x, cond_emb, block_index, "self_attn_q", strength)
  k_in = apply_module(lllite, x, cond_emb, block_index, "self_attn_k", strength)
  v_in = apply_module(lllite, x, cond_emb, block_index, "self_attn_v", strength)
  return q_in, k_in, v_in


def time_snr_shift(alpha: float, t: float) -> float:
  """ComfyUI's flow-matching sigma shift, used by percent_to_sigma."""
  return alpha * t / (1.0 + (alpha - 1.0) * t)


def percent_to_sigma(percent: float, shift: float = 1.0) -> float:
  """Map a 0..1 sampling position to sigma, matching ModelSamplingDiscreteFlow.

  The workflow exposes the LLLite window as `start_percent`/`end_percent` and the node
  converts them with `sampling.percent_to_sigma(...)` before building the patch. Anima is a
  flow-matching model, so percent 0 is sigma 1.0 (fully noised) and percent 1 is sigma 0.0.
  With the default shift of 1.0 this reduces to `1 - percent`.
  """
  if percent <= 0.0:
    return 1.0
  if percent >= 1.0:
    return 0.0
  return time_snr_shift(shift, 1.0 - percent)


def sigma_gate(sigma, sigma_start: float, sigma_end: float, strength: float) -> bool:
  """AnimaLLLitePatch gating: skip when strength is 0 or sigma leaves [end, start]."""
  if strength == 0.0:
    return False
  return float(sigma_end) <= float(sigma) <= float(sigma_start)


def active_at(sigma, strength: float, start_percent: float, end_percent: float,
              shift: float = 1.0) -> bool:
  """True when LLLite should apply at this sigma, using the workflow's percent inputs.

  The default workflow ships `strength 1, start 0, end 1` — i.e. active across the whole
  schedule. Its note also suggests ending nearer 0.6-0.8 when the pose comes out too rigid.
  """
  return sigma_gate(sigma, percent_to_sigma(start_percent, shift),
                    percent_to_sigma(end_percent, shift), strength)


# ---------------------------------------------------------------------------
# the chain the reference workflow actually runs
# ---------------------------------------------------------------------------

@dataclass
class WorkflowChain:
  """Defaults taken from `anima_charref_pose_simple.json` in the CharRef-LLLite repo.

  The pack order matters and is reproduced here so the UI and the sampler agree:

    UNETLoader -> Power Lora Loader -> LLLite (patch) -> Ref Adapter controls loader
      -> ModelComputeDtype(bf16) -> ModelAttentionBackend -> SafePAG -> AnimaRefGuider
      -> SamplerCustomAdvanced -> VAEDecode

  So LLLite is installed on the model *before* the reference adapter, it hooks attn1
  (q, k, v) and attn2 (cross q), and its conditioning is encoded once per forward through
  the model's post-input patch. Turbo sampling is 8 steps, euler, simple, CFG 1.
  """
  # LLLite node widgets: strength 1.0, start_percent 0.0, end_percent 1.0
  lllite_strength: float = 1.0
  lllite_start_percent: float = 0.0
  lllite_end_percent: float = 1.0
  # sampler
  steps: int = 8
  sampler: str = "euler"
  scheduler: str = "simple"
  cfg: float = 1.0
  # second pass safety
  ref_guidance: float = 1.0
  safe_pag: float = 1.3
  compute_dtype: str = "bf16"
  # output sizing: the pose image sets it, scaled to ~1 MP with lanczos
  megapixels: float = 1.0
  latent_width: int = 896
  latent_height: int = 1152

  def note(self) -> str:
    return ("LLLite is trained on Anima Base; with Turbo it is experimental. "
            "Lower lllite_end_percent to 0.6-0.8 if the pose reads too rigid.")


def preprocess_control(image, target_hw: Tuple[int, int], mask=None, inpaint: bool = False,
                       zero_masked: bool = False):
  """Control image -> (1,H,W,C) float32 in [-1,1].

  Matches AnimaLLLitePatch.__call__: bicubic resize to latent x 8, clamp [0,1], map to [-1,1].
  """
  from PIL import Image
  h, w = int(target_hw[0]), int(target_hw[1])
  arr = np.asarray(image, dtype=np.float32)
  if arr.ndim == 2:
    arr = arr[..., None]
  if arr.max() > 1.5:
    arr = arr / 255.0
  if arr.shape[-1] == 1:
    arr = np.repeat(arr, 3, axis=-1)
  arr = np.stack([np.asarray(Image.fromarray(arr[..., c]).resize((w, h), Image.BICUBIC),
                            np.float32) for c in range(arr.shape[-1])], axis=-1)
  arr = np.clip(arr, 0.0, 1.0) * 2.0 - 1.0
  if inpaint:
    if mask is None:
      raise LLLiteError("inpainting LLLite checkpoint requires a mask")
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
      m = m[..., None]
    if m.max() > 1.5:
      m = m / 255.0
    m = np.asarray(Image.fromarray((m[..., 0] * 255).astype(np.uint8)).resize(
        (w, h), Image.NEAREST), np.float32)[..., None] / 255.0
    m = (m >= 0.5).astype(np.float32)
    if zero_masked:
      arr = arr * (m < 0.5).astype(np.float32)
    arr = np.concatenate([arr, m * 2.0 - 1.0], axis=-1)
  return arr[None]
