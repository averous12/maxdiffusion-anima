import math
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
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
  return (xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps) * weight).astype(x.dtype)


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
    h = nn.silu(embedded_timestep)
    h = nn.Dense(self.adaln_dim, use_bias=False, name="linear_1")(h)
    h = nn.Dense(3 * self.hidden, use_bias=False, name="linear_2")(h)
    h = h + temb
    shift, scale, gate = jnp.split(h, 3, axis=-1)
    y = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x)
    return y * (1 + scale[:, None, :]) + shift[:, None, :], gate[:, None, :]


class _Attention(nn.Module):
  hidden: int
  heads: int
  context_dim: Optional[int] = None
  cross: bool = False
  @nn.compact
  def __call__(self, x, context=None, cos=None, sin=None, mask=None):
    ctx = x if context is None else context
    d = self.hidden // self.heads
    q = nn.Dense(self.hidden, use_bias=False, name="to_q")(x).reshape(x.shape[0], x.shape[1], self.heads, d)
    k = nn.Dense(self.hidden, use_bias=False, name="to_k")(ctx).reshape(ctx.shape[0], ctx.shape[1], self.heads, d)
    v = nn.Dense(self.hidden, use_bias=False, name="to_v")(ctx).reshape(ctx.shape[0], ctx.shape[1], self.heads, d)
    q = _rms(q, self.param("norm_q", nn.initializers.ones, (d,)))
    k = _rms(k, self.param("norm_k", nn.initializers.ones, (d,)))
    if not self.cross and cos is not None:
      q = q * cos[None, :, None, :] + _rotate_pairs(q) * sin[None, :, None, :]
      k = k * cos[None, :, None, :] + _rotate_pairs(k) * sin[None, :, None, :]
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) / math.sqrt(d)
    if mask is not None:
      scores = jnp.where(mask[:, None, None, :].astype(bool), scores, -1e4)
    y = jnp.einsum("bhqk,bkhd->bqhd", nn.softmax(scores, axis=-1).astype(x.dtype), v)
    return nn.Dense(self.hidden, use_bias=False, name="to_out")(y.reshape(x.shape[0], x.shape[1], self.hidden))


class _Block(nn.Module):
  hidden: int
  heads: int
  context_dim: int
  adaln_dim: int
  @nn.compact
  def __call__(self, x, embedded_timestep, temb, context, cos, sin, mask=None):
    h, gate1 = _AdaLN(self.hidden, self.adaln_dim, name="norm1")(x, embedded_timestep, temb)
    x = x + gate1 * _Attention(self.hidden, self.heads, name="attn1")(h, cos=cos, sin=sin)
    h, gate2 = _AdaLN(self.hidden, self.adaln_dim, name="norm2")(x, embedded_timestep, temb)
    x = x + gate2 * _Attention(self.hidden, self.heads, self.context_dim, cross=True, name="attn2")(h, context=context, mask=mask)
    h, gate3 = _AdaLN(self.hidden, self.adaln_dim, name="norm3")(x, embedded_timestep, temb)
    h = nn.Dense(self.hidden * 4, use_bias=False, name="ff_in")(h)
    h = nn.gelu(h, approximate=False)
    return x + gate3 * nn.Dense(self.hidden, use_bias=False, name="ff_out")(h)


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
    x = nn.Dense(hidden, use_bias=False, name="patch_embed")(tokens)
    half = hidden // 2
    # time_proj: flip_sin_to_cos=True, downscale_freq_shift=0.0 -> exponent / half, concat [cos, sin]
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    tfeat = timestep.astype(jnp.float32)[:, None] * freqs[None, :]
    tproj = jnp.concatenate([jnp.cos(tfeat), jnp.sin(tfeat)], axis=-1).astype(x.dtype)
    # Stream 1: t_embedder(tproj) = linear_1 -> silu -> linear_2(3*hidden); added in every AdaLN
    temb = nn.Dense(hidden, use_bias=False, name="time_embed_linear_1")(tproj)
    temb = nn.silu(temb)
    temb = nn.Dense(3 * hidden, use_bias=False, name="time_embed_linear_2")(temb)
    # Stream 2: embedded_timestep = RMS(tproj); flows through each AdaLN's own MLP
    embedded_timestep = _rms(tproj, self.param("time_embed_norm", nn.initializers.ones, (hidden,)))
    grid = (t // self.patch_size[0], h // self.patch_size[1], w // self.patch_size[2])
    cos, sin = cosmos_rope(x.shape[1], self.head_dim, x.dtype, self.rope_scale, grid)
    for i in range(self.layers):
      if self.active_layers is not None and i >= self.active_layers:
        break
      x = _Block(hidden, self.heads, self.context_dim, self.adaln_dim, name=f"transformer_blocks_{i}")(x, embedded_timestep, temb, encoder_hidden_states, cos, sin, attention_mask)
    # norm_out (CosmosAdaLayerNorm): silu -> lin1 -> lin2(2*hidden), + temb[..., :2h], chunk2
    y = nn.silu(embedded_timestep)
    y = nn.Dense(self.adaln_dim, use_bias=False, name="norm_out_linear_1")(y)
    y = nn.Dense(2 * hidden, use_bias=False, name="norm_out_linear_2")(y)
    y = y + temb[:, : 2 * hidden]
    shift, scale = jnp.split(y, 2, axis=-1)
    y = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x)
    y = y * (1 + scale[:, None, :]) + shift[:, None, :]
    y = nn.Dense(self.out_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2], use_bias=False, name="proj_out")(y)
    return cosmos_unpatchify(y, self.out_channels, (t, h, w), self.patch_size)


def _transpose_weight(value):
  return value.T if value.ndim == 2 else value


def convert_anima_cosmos_weights(safetensors_path, flax_params, dtype=jnp.bfloat16, num_layers=28, strict=True):
  """Strictly map Diffusers Cosmos/Anima names to this Flax module."""
  from safetensors import safe_open
  flat = flatten_dict(flax_params)
  converted = {}
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
    put(("patch_embed", "kernel"), "patch_embed.proj.weight")
    put(("time_embed_linear_1", "kernel"), "time_embed.t_embedder.linear_1.weight")
    put(("time_embed_linear_2", "kernel"), "time_embed.t_embedder.linear_2.weight")
    put(("time_embed_norm",), "time_embed.norm.weight")
    put(("norm_out_linear_1", "kernel"), "norm_out.linear_1.weight")
    put(("norm_out_linear_2", "kernel"), "norm_out.linear_2.weight")
    put(("proj_out", "kernel"), "proj_out.weight")
    for i in range(num_layers):
      s = f"transformer_blocks.{i}"
      t = f"transformer_blocks_{i}"
      for norm in ("norm1", "norm2", "norm3"):
        put((t, norm, "linear_1", "kernel"), f"{s}.{norm}.linear_1.weight")
        put((t, norm, "linear_2", "kernel"), f"{s}.{norm}.linear_2.weight")
      for attn in ("attn1", "attn2"):
        for proj in ("to_q", "to_k", "to_v"):
          put((t, attn, proj, "kernel"), f"{s}.{attn}.{proj}.weight")
        put((t, attn, "to_out", "kernel"), f"{s}.{attn}.to_out.0.weight")
        put((t, attn, "norm_q"), f"{s}.{attn}.norm_q.weight")
        put((t, attn, "norm_k"), f"{s}.{attn}.norm_k.weight")
      put((t, "ff_in", "kernel"), f"{s}.ff.net.0.proj.weight")
      put((t, "ff_out", "kernel"), f"{s}.ff.net.2.weight")
    extras = available - consumed
    if strict and extras:
      raise ValueError(f"Unconsumed official transformer keys: {sorted(extras)[:10]}")
  return unflatten_dict(converted)
