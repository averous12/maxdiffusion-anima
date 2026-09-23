import math
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.axes_scan import broadcast as _scan_broadcast
from flax.traverse_util import flatten_dict, unflatten_dict


def cosmos_patchify(x, patch_size=(1, 2, 2)):
  b, c, t, h, w = x.shape
  pt, ph, pw = patch_size
  if t % pt or h % ph or w % pw:
    raise ValueError("Input dimensions must be divisible by patch_size")
  y = x.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
  y = jnp.transpose(y, (0, 2, 4, 6, 1, 3, 5, 7))
  return y.reshape(b, (t // pt) * (h // ph) * (w // pw), c * pt * ph * pw)


def cosmos_unpatchify(tokens, output_channels, spatial_shape, patch_size=(1, 2, 2)):
  t, h, w = spatial_shape
  pt, ph, pw = patch_size
  b, n, d = tokens.shape
  expected = (t // pt) * (h // ph) * (w // pw)
  if n != expected or d != output_channels * pt * ph * pw:
    raise ValueError("Invalid token shape for unpatchify")
  # Diffusers splits the output channel axis as (ph, pw, pt, cout) — ph major, cout minor —
  # then permutes (0, cout, nf, pt, nh, ph, nw, pw) back to (B, C, T, H, W).
  y = tokens.reshape(b, t // pt, h // ph, w // pw, ph, pw, pt, output_channels)
  y = jnp.transpose(y, (0, 7, 1, 6, 2, 4, 3, 5))
  return y.reshape(b, output_channels, t, h, w)


def _rotate_half(x):
  half = x.shape[-1] // 2
  return jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def _rotate_pairs(x):
  # Cosmos/diffusers style: x reshaped (..., D//2, 2); (even, odd) pairs rotate together.
  # rotated = cat([-x_imag, x_real], -1) where x_real=x[...,0,:], x_imag=x[...,1,:]
  d = x.shape[-1]
  pairs = x.reshape(x.shape[:-1] + (2, d // 2))
  x_real = pairs[..., 0, :]
  x_imag = pairs[..., 1, :]
  return jnp.concatenate([-x_imag, x_real], axis=-1)


def _rms(x, weight, eps=1e-6):
  xf = x.astype(jnp.float32)
  return (xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps) * weight.astype(jnp.float32)).astype(jnp.float32)


class AnimaDtypePolicy:
  """One-knob bf16 ladder: flip fields individually, never rewrite the model.

  Baseline (all fp32): proven Fern run. Stage 1 flips only MATMUL_PRECISION.
  Later stages flip ACT/RESIDUAL/etc. one at a time.
  """
  ACT_DTYPE = jnp.float32        # latent/ctx/pad in/out of transformer
  RESIDUAL_DTYPE = jnp.float32   # block residual adds
  TEMB_DTYPE = jnp.float32       # timestep embedding streams
  ROPE_DTYPE = jnp.float32       # RoPE cos/sin
  ATTN_SCORE_DTYPE = jnp.float32 # QK scores + softmax
  NORM_DTYPE = jnp.float32       # RMS/LayerNorm + norm_out
  PARAM_DTYPE = jnp.bfloat16     # stored weights (memory win, no arithmetic change)
  MATMUL_PRECISION = "BF16_BF16_F32"  # bf16 operands, fp32 accumulation


def cosmos_rope(seq_len, head_dim, dtype, rope_scale=(1.0, 4.0, 4.0), grid=None):
  if grid is None:
    pos = jnp.arange(seq_len, dtype=jnp.float32)[:, None]
    inv = 1.0 / (10000.0 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    f = pos * inv[None, :]
  else:
    t, h, w = grid
    dh = head_dim // 6 * 2
    dw = dh
    dt = head_dim - dh - dw
    def axis(n, d, scale):
      # Diffusers NTK-style theta scaling: theta = 10000 * scale ** (d / (d - 2))
      theta = 10000.0 * scale ** (d / (d - 2))
      inv = 1.0 / (theta ** (jnp.arange(0, d, 2, dtype=jnp.float32) / d))
      return jnp.arange(n, dtype=jnp.float32)[:, None] * inv[None, :]
    ft, fh, fw = axis(t, dt, rope_scale[0]), axis(h, dh, rope_scale[1]), axis(w, dw, rope_scale[2])
    f = jnp.concatenate([
        jnp.tile(ft[:, None, None, :], (1, h, w, 1)),
        jnp.tile(fh[None, :, None, :], (t, 1, w, 1)),
        jnp.tile(fw[None, None, :, :], (t, h, 1, 1)),
    ], axis=-1).reshape(t * h * w, -1)
  e = jnp.concatenate([f, f], axis=-1)
  return jnp.cos(e).astype(dtype), jnp.sin(e).astype(dtype)


class _AdaLN(nn.Module):
  hidden: int
  adaln_dim: int
  @nn.compact
  def __call__(self, x, embedded_timestep, temb):
    et = embedded_timestep.astype(jnp.float32)
    tb = temb.astype(jnp.float32)
    h = nn.silu(et)
    h = nn.Dense(self.adaln_dim, use_bias=False, dtype=jnp.float32, param_dtype=jnp.float32, name="linear_1")(h)
    h = nn.Dense(3 * self.hidden, use_bias=False, dtype=jnp.float32, param_dtype=jnp.float32, name="linear_2")(h)
    h = h + tb
    shift, scale, gate = jnp.split(h, 3, axis=-1)
    y = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x.astype(jnp.float32))
    return (y * (1 + scale[:, None, :]) + shift[:, None, :]).astype(x.dtype), gate[:, None, :].astype(x.dtype)


class _Attention(nn.Module):
  hidden: int
  heads: int
  context_dim: Optional[int] = None
  cross: bool = False
  @nn.compact
  def __call__(self, x, context=None, cos=None, sin=None, mask=None):
    ctx = x if context is None else context
    d = self.hidden // self.heads
    qd = self.hidden // self.heads
    q = nn.Dense(self.hidden, use_bias=False, name="to_q")(x).reshape(x.shape[0], x.shape[1], self.heads, qd)
    k = nn.Dense(self.hidden, use_bias=False, name="to_k")(ctx).reshape(ctx.shape[0], ctx.shape[1], self.heads, qd)
    v = nn.Dense(self.hidden, use_bias=False, name="to_v")(ctx).reshape(ctx.shape[0], ctx.shape[1], self.heads, qd)
    q = _rms(q, self.param("norm_q", nn.initializers.ones, (qd,)))
    k = _rms(k, self.param("norm_k", nn.initializers.ones, (qd,)))
    if not self.cross and cos is not None:
      cos = cos.astype(jnp.float32); sin = sin.astype(jnp.float32)
      q = q.astype(jnp.float32); k = k.astype(jnp.float32)
      q = q * cos[None, :, None, :] + _rotate_pairs(q) * sin[None, :, None, :]
      k = k * cos[None, :, None, :] + _rotate_pairs(k) * sin[None, :, None, :]
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) / math.sqrt(d)
    if mask is not None:
      scores = jnp.where(mask[:, None, None, :].astype(bool), scores, -1e4)
    y = jnp.einsum("bhqk,bkhd->bqhd", nn.softmax(scores, axis=-1).astype(jnp.float32), v.astype(jnp.float32))
    b, q_len, h, d = y.shape
    y = y.reshape(b, q_len, h * d)
    return nn.Dense(self.hidden, use_bias=False, name="to_out")(y)


class _Block(nn.Module):
  hidden: int
  heads: int
  context_dim: int
  adaln_dim: int
  @nn.compact
  def __call__(self, x, embedded_timestep, temb, context, cos, sin, mask=None):
    h, gate1 = _AdaLN(self.hidden, self.adaln_dim, name="norm1")(x.astype(jnp.float32), embedded_timestep, temb)
    # fp32 residual accumulation: bf16 activations in, fp32 adds out.
    x = x.astype(jnp.float32) + gate1 * _Attention(self.hidden, self.heads, name="attn1")(h, cos=cos, sin=sin).astype(jnp.float32)
    h, gate2 = _AdaLN(self.hidden, self.adaln_dim, name="norm2")(x, embedded_timestep, temb)
    x = x + gate2 * _Attention(self.hidden, self.heads, self.context_dim, cross=True, name="attn2")(h, context=context, mask=mask).astype(jnp.float32)
    h, gate3 = _AdaLN(self.hidden, self.adaln_dim, name="norm3")(x, embedded_timestep, temb)
    h = nn.Dense(self.hidden * 4, use_bias=False, name="ff_in")(h)
    h = nn.gelu(h, approximate=False)
    x = x.astype(jnp.float32) + gate3.astype(jnp.float32) * nn.Dense(self.hidden, use_bias=False, name="ff_out")(h).astype(jnp.float32)
    return x


class _ScannedBlock(nn.Module):
  """One transformer block, in the shape nn.scan needs.

  Returns (carry, output) — identical here — so the residual stream threads
  through the scan. Params get a leading axis of length `layers`.
  """
  hidden: int
  heads: int
  context_dim: int
  adaln_dim: int
  @nn.compact
  def __call__(self, x, embedded_timestep, temb, context, cos, sin, mask):
    out = _Block(self.hidden, self.heads, self.context_dim, self.adaln_dim, name="b")(
        x, embedded_timestep, temb, context, cos, sin, mask)
    return out, out


class FlaxAnimaCosmosTransformer(nn.Module):
  in_channels: int = 16
  out_channels: int = 16
  heads: int = 16
  head_dim: int = 128
  layers: int = 28
  context_dim: int = 1024
  adaln_dim: int = 256
  patch_size: Tuple[int, int, int] = (1, 2, 2)
  rope_scale: Tuple[float, float, float] = (1.0, 4.0, 4.0)
  active_layers: Optional[int] = None
  scan_blocks: bool = True  # lax.scan over blocks: one compiled body, not 28
  diag_sync: bool = False  # diagnostic only: block_until_ready per stage (destroys perf)
  @nn.compact
  def __call__(self, hidden_states, timestep, encoder_hidden_states, attention_mask=None, padding_mask=None):
    b, c, t, h, w = hidden_states.shape
    if padding_mask is None:
      padding_mask = jnp.zeros((b, 1, h, w), dtype=hidden_states.dtype)
    if padding_mask.ndim == 3:
      padding_mask = padding_mask[:, None, :, :]
    # Reference takes the image-resolution mask and nearest-resizes it to the
    # latent grid before concatenating (transformer_cosmos.py). Accept either.
    if padding_mask.shape[-2:] != (h, w):
      ph, pw = padding_mask.shape[-2:]
      assert ph % h == 0 and pw % w == 0, (padding_mask.shape, (h, w))
      sh, sw = ph // h, pw // w
      padding_mask = padding_mask[:, :, : h * sh : sh, : w * sw : sw]
    padding_channel = jnp.repeat(padding_mask[:, :, None, :, :], t, axis=2)
    x = jnp.concatenate([hidden_states, padding_channel], axis=1)
    tokens = cosmos_patchify(x, self.patch_size)
    hidden = self.heads * self.head_dim
    x = nn.Dense(hidden, use_bias=False, name="patch_embed", dtype=jnp.float32, param_dtype=jnp.bfloat16)(tokens)
    x = x.astype(jnp.float32)
    half = hidden // 2
    # time_proj: flip_sin_to_cos=True, downscale_freq_shift=0.0 -> exponent / half, concat [cos, sin]
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    tfeat = timestep.astype(jnp.float32)[:, None] * freqs[None, :]
    tproj = jnp.concatenate([jnp.cos(tfeat), jnp.sin(tfeat)], axis=-1).astype(jnp.float32)
    # Stream 1: t_embedder(tproj) = linear_1 -> silu -> linear_2(3*hidden); added in every AdaLN
    temb = nn.Dense(hidden, use_bias=False, name="time_embed_linear_1", dtype=jnp.float32, param_dtype=jnp.float32)(tproj)
    temb = nn.silu(temb)
    temb = nn.Dense(3 * hidden, use_bias=False, name="time_embed_linear_2", dtype=jnp.float32, param_dtype=jnp.float32)(temb)
    # Stream 2: embedded_timestep = RMS(tproj); flows through each AdaLN's own MLP
    embedded_timestep = _rms(tproj, self.param("time_embed_norm", nn.initializers.ones, (hidden,)))
    grid = (t // self.patch_size[0], h // self.patch_size[1], w // self.patch_size[2])
    cos, sin = cosmos_rope(x.shape[1], self.head_dim, jnp.float32, self.rope_scale, grid)
    if self.scan_blocks:
      # One block body compiled once and looped, instead of 28 unrolled
      # subgraphs. This is what keeps host compile memory bounded.
      n = self.layers if self.active_layers is None else self.active_layers
      ScanBlock = nn.scan(
          _ScannedBlock,
          variable_axes={"params": 0},
          split_rngs={"params": False},
          length=n,
          in_axes=(_scan_broadcast,) * 6,
      )
      x, _ = ScanBlock(hidden, self.heads, self.context_dim, self.adaln_dim, name="scan")(
          x, embedded_timestep, temb, encoder_hidden_states, cos, sin, attention_mask)
      x = x.astype(jnp.float32)
    else:
      for i in range(self.layers):
        if self.active_layers is not None and i >= self.active_layers:
          break
        x = _Block(hidden, self.heads, self.context_dim, self.adaln_dim, name=f"transformer_blocks_{i}")(x, embedded_timestep, temb, encoder_hidden_states, cos, sin, attention_mask)
        x = x.astype(jnp.float32)
        if self.diag_sync:
          try:
            x.block_until_ready()
          except Exception:
            pass
    # norm_out (CosmosAdaLayerNorm): silu -> lin1 -> lin2(2*hidden), + temb[..., :2h], chunk2
    y = nn.silu(embedded_timestep)
    y = nn.Dense(self.adaln_dim, use_bias=False, name="norm_out_linear_1", dtype=jnp.float32, param_dtype=jnp.float32)(y)
    y = nn.Dense(2 * hidden, use_bias=False, name="norm_out_linear_2", dtype=jnp.float32, param_dtype=jnp.float32)(y)
    y = y + temb[:, : 2 * hidden]
    shift, scale = jnp.split(y, 2, axis=-1)
    y = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x.astype(jnp.float32))
    y = y * (1 + scale[:, None, :]) + shift[:, None, :]
    y = nn.Dense(self.out_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2], use_bias=False, name="proj_out", dtype=jnp.float32, param_dtype=jnp.float32)(y)
    return cosmos_unpatchify(y, self.out_channels, (t, h, w), self.patch_size)


def _transpose_weight(value):
  return value.T if value.ndim == 2 else value


def stack_block_params(converted, num_layers):
  """Fold per-block unrolled params into the nn.scan layout.

  ('transformer_blocks_i', *rest) for i in 0..N-1  ->  ('scan', 'b', *rest)
  with a leading axis of length N. Leaves non-block entries untouched.
  """
  out = {}
  block_keys = None
  for i in range(num_layers):
    prefix = f"transformer_blocks_{i}"
    for k, v in converted.items():
      if k[0] != prefix:
        continue
      rest = k[1:]
      if block_keys is None:
        block_keys = set()
      block_keys.add(rest)
      out.setdefault(("scan", "b") + rest, []).append((i, v))
  if block_keys is None:
    raise ValueError("stack_block_params: no per-block params found")
  stacked = {}
  for k, vals in out.items():
    vals.sort(key=lambda t: t[0])
    if len(vals) != num_layers:
      raise ValueError(f"{k}: expected {num_layers} layers, got {len(vals)}")
    stacked[k] = jnp.stack([v for _, v in vals], axis=0)
  for k, v in converted.items():
    if k[0].startswith("transformer_blocks_"):
      continue
    stacked[k] = v
  return stacked


def _aesthetic_get(available, name):
  key = f"model.diffusion_model.{name}"
  if key not in available:
    raise KeyError(f"Missing Anima aesthetic key: {key}")
  return key


def convert_anima_aesthetic_weights(safetensors_path, flax_params, dtype=jnp.bfloat16, num_layers=28, stacked=True):
  """Convert official single-file Anima-Aesthetic transformer weights.

  When `stacked` is True the per-block weights are folded into the nn.scan
  layout ('scan','b',*rest) with a leading axis of length num_layers.
  """
  from safetensors import safe_open
  flat = flatten_dict(flax_params)
  converted = {}
  block = {}  # rest -> {layer_index: value}
  def block_put(rest, i, value):
    if tuple(value.shape) != tuple(flat[("scan", "b") + rest].shape[1:]):
      raise ValueError(f"Shape mismatch block {i} {rest}: {value.shape} vs {flat[('scan','b')+rest].shape[1:]}")
    block.setdefault(rest, {})[i] = value
  with safe_open(safetensors_path, framework="pt", device="cpu") as tensors:
    available = set(tensors.keys())
    consumed = set()
    def put(dst, src, tr=True):
      src = _aesthetic_get(available, src)
      value = tensors.get_tensor(src).float().numpy()
      value = value.T if tr and value.ndim == 2 else value
      value = jnp.asarray(value, dtype=dtype)
      if tuple(value.shape) != tuple(flat[dst].shape):
        raise ValueError(f"Shape mismatch {src}: {value.shape} != {dst}: {flat[dst].shape}")
      converted[dst] = value; consumed.add(src)
    print("[aesthetic] core embeddings", flush=True)
    put(("patch_embed", "kernel"), "x_embedder.proj.1.weight")
    put(("time_embed_linear_1", "kernel"), "t_embedder.1.linear_1.weight")
    put(("time_embed_linear_2", "kernel"), "t_embedder.1.linear_2.weight")
    put(("time_embed_norm",), "t_embedding_norm.weight", False)
    put(("norm_out_linear_1", "kernel"), "final_layer.adaln_modulation.1.weight")
    put(("norm_out_linear_2", "kernel"), "final_layer.adaln_modulation.2.weight")
    put(("proj_out", "kernel"), "final_layer.linear.weight")
    for i in range(num_layers):
      print(f"[aesthetic] transformer block {i+1}/{num_layers}", flush=True)
      s=f"blocks.{i}"
      for norm, source in (("norm1","self_attn"),("norm2","cross_attn"),("norm3","mlp")):
        for lin in ("linear_1", "linear_2"):
          src = _aesthetic_get(available, f"{s}.adaln_modulation_{source}.{lin[-1]}.weight")
          v = jnp.asarray(tensors.get_tensor(src).float().numpy().T, dtype=dtype)
          block_put((norm, lin, "kernel"), i, v); consumed.add(src)
      for attn, source in (("attn1","self_attn"),("attn2","cross_attn")):
        for proj, dst_proj in (("q_proj","to_q"),("k_proj","to_k"),("v_proj","to_v")):
          src = _aesthetic_get(available, f"{s}.{source}.{proj}.weight")
          v = jnp.asarray(tensors.get_tensor(src).float().numpy().T, dtype=dtype)
          block_put((attn, dst_proj, "kernel"), i, v); consumed.add(src)
        src = _aesthetic_get(available, f"{s}.{source}.output_proj.weight")
        v = jnp.asarray(tensors.get_tensor(src).float().numpy().T, dtype=dtype)
        block_put((attn, "to_out", "kernel"), i, v); consumed.add(src)
        for nrm in ("q_norm", "k_norm"):
          src = _aesthetic_get(available, f"{s}.{source}.{nrm}.weight")
          v = jnp.asarray(tensors.get_tensor(src).float().numpy(), dtype=dtype)
          block_put((attn, "norm_q" if nrm == "q_norm" else "norm_k"), i, v); consumed.add(src)
      for dst_name, src_name in (("ff_in", "layer1"), ("ff_out", "layer2")):
        src = _aesthetic_get(available, f"{s}.mlp.{src_name}.weight")
        v = jnp.asarray(tensors.get_tensor(src).float().numpy().T, dtype=dtype)
        block_put((dst_name, "kernel"), i, v); consumed.add(src)
    for rest, per_layer in block.items():
      missing = [i for i in range(num_layers) if i not in per_layer]
      if missing:
        raise ValueError(f"block param {rest} missing layers {missing[:5]}")
      st = jnp.stack([per_layer[i] for i in range(num_layers)], axis=0)
      dst = ("scan", "b") + rest
      if tuple(st.shape) != tuple(flat[dst].shape):
        raise ValueError(f"Stacked shape mismatch {rest}: {st.shape} != {flat[dst].shape}")
      converted[dst] = st
    extras = set(tensors.keys()) - consumed
    if extras:
      # The LLM adapter is validated by its separate converter.
      all_extras = {x for x in extras if x != "__metadata__" and not x.startswith("model.diffusion_model.llm_adapter.")}
      if all_extras:
        raise ValueError(f"Unconsumed aesthetic transformer keys: {sorted(all_extras)[:10]}")
  return unflatten_dict(converted)

def convert_anima_cosmos_weights(safetensors_path, flax_params, dtype=jnp.bfloat16, num_layers=28, strict=True, stacked=True, low_memory=False):
  """Strictly map Diffusers Cosmos/Anima names to this Flax module.

  When `stacked` is True the per-block weights are folded into the nn.scan
  layout ('scan','b',*rest) with a leading axis of length num_layers.

  `low_memory` fills preallocated numpy buffers in place and converts to jnp only
  once at the end, instead of holding a per-layer dict and the stacked result at
  the same time. Measured peak for a 28-layer bf16 model: ~12GB -> ~8GB, which is
  the difference between fitting and being OOM-killed on a 13GB CPU host. The
  mapping and results are identical; the default path is unchanged.
  """
  import numpy as np
  from safetensors import safe_open
  flat = flatten_dict(flax_params)
  converted = {}
  block = {}  # rest -> {layer_index: value}   (default path only)
  if low_memory:
    scan_rests = [k[2:] for k in flat if k[0] == "scan"]
    np_dtype = jnp.dtype(dtype)
    bufs = {r: np.zeros((num_layers,) + tuple(flat[("scan", "b") + r].shape[1:]),
                        dtype=np_dtype) for r in scan_rests}
    filled = {r: 0 for r in scan_rests}
  def block_put(rest, i, value):
    if tuple(value.shape) != tuple(flat[("scan", "b") + rest].shape[1:]):
      raise ValueError(f"Shape mismatch block {i} {rest}: {value.shape} vs {flat[('scan','b')+rest].shape[1:]}")
    if low_memory:
      bufs[rest][i] = value
      filled[rest] += 1
    else:
      block.setdefault(rest, {})[i] = value
  with safe_open(safetensors_path, framework="pt", device="cpu") as tensors:
    available = set(tensors.keys())
    consumed = set()
    def put(dst, src):
      if src not in available:
        raise KeyError(f"Missing transformer weight: {src}")
      value = jnp.asarray(_transpose_weight(tensors.get_tensor(src).float().numpy()), dtype=dtype)
      if tuple(value.shape) != tuple(flat[dst].shape):
        raise ValueError(f"Shape mismatch {src}: {value.shape} != {dst}: {flat[dst].shape}")
      converted[dst] = value
      consumed.add(src)
    def bput(rest, src):
      if src not in available:
        raise KeyError(f"Missing transformer weight: {src}")
      raw = _transpose_weight(tensors.get_tensor(src).float().numpy())
      if low_memory:
        block_put(rest, i, np.asarray(raw, dtype=np_dtype))
      else:
        block_put(rest, i, jnp.asarray(raw, dtype=dtype))
      consumed.add(src)
    put(("patch_embed", "kernel"), "patch_embed.proj.weight")
    put(("time_embed_linear_1", "kernel"), "time_embed.t_embedder.linear_1.weight")
    put(("time_embed_linear_2", "kernel"), "time_embed.t_embedder.linear_2.weight")
    put(("time_embed_norm",), "time_embed.norm.weight")
    put(("norm_out_linear_1", "kernel"), "norm_out.linear_1.weight")
    put(("norm_out_linear_2", "kernel"), "norm_out.linear_2.weight")
    put(("proj_out", "kernel"), "proj_out.weight")
    for i in range(num_layers):
      s = f"transformer_blocks.{i}"
      for norm in ("norm1", "norm2", "norm3"):
        bput((norm, "linear_1", "kernel"), f"{s}.{norm}.linear_1.weight")
        bput((norm, "linear_2", "kernel"), f"{s}.{norm}.linear_2.weight")
      for attn in ("attn1", "attn2"):
        for proj in ("to_q", "to_k", "to_v"):
          bput((attn, proj, "kernel"), f"{s}.{attn}.{proj}.weight")
        bput((attn, "to_out", "kernel"), f"{s}.{attn}.to_out.0.weight")
        bput((attn, "norm_q"), f"{s}.{attn}.norm_q.weight")
        bput((attn, "norm_k"), f"{s}.{attn}.norm_k.weight")
      bput(("ff_in", "kernel"), f"{s}.ff.net.0.proj.weight")
      bput(("ff_out", "kernel"), f"{s}.ff.net.2.weight")
    if low_memory:
      # convert one stacked buffer at a time and drop it immediately, so the
      # numpy and jnp copies never all coexist
      for r in scan_rests:
        if filled[r] != num_layers:
          raise ValueError(f"block param {r} filled {filled[r]}/{num_layers} layers")
        dst = ("scan", "b") + r
        st = jnp.asarray(bufs.pop(r))
        if tuple(st.shape) != tuple(flat[dst].shape):
          raise ValueError(f"Stacked shape mismatch {r}: {st.shape} != {flat[dst].shape}")
        converted[dst] = st
    else:
      for rest, per_layer in block.items():
        missing = [i for i in range(num_layers) if i not in per_layer]
        if missing:
          raise ValueError(f"block param {rest} missing layers {missing[:5]}")
        st = jnp.stack([per_layer[i] for i in range(num_layers)], axis=0)
        dst = ("scan", "b") + rest
        if tuple(st.shape) != tuple(flat[dst].shape):
          raise ValueError(f"Stacked shape mismatch {rest}: {st.shape} != {flat[dst].shape}")
        converted[dst] = st
    extras = available - consumed
    if strict and extras:
      raise ValueError(f"Unconsumed official transformer keys: {sorted(extras)[:10]}")
  return unflatten_dict(converted)
