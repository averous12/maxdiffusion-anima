"""Anima pixel head: decode the DiT's last-layer features straight to RGB.

A ConvNeXt readout over the input of the DiT's `final_layer.linear` (the adaLN-modulated
LayerNorm output, 2048 per token) plus the 2x2 latent patch (64 values) that token covers.
It writes a 16x16 RGB patch per token in [-1, 1] -- an alternative to the VAE decode that
skips the multi-GB decoder activations entirely.

Ported from `library/anima/pixel_head.py` in sorryhyun/anima_lora (vendored into the
ComfyUI-Anima-PixelDecoder node pack). The weight layout, the layer order and the
pixel-shuffle convention all follow that reference; the tests compare this port against it
numerically.
"""

import json
import math
import struct

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

PATCH = 16          # image pixels per DiT token side (VAE /8 x DiT patch 2)
LAT_PATCH_DIM = 64  # 16 latent channels x 2 x 2
DEFAULT_SIGMA_RANGES = ((0.0577, 0.1364), (0.5, 0.5))
SIGMA_TOL = 2e-3


class SnapRefError(RuntimeError):
  """Kept local so this module does not depend on the SnapRef adapter code."""


class PixelHeadError(RuntimeError):
  pass


class ConvNeXtBlock(nn.Module):
  dim: int

  @nn.compact
  def __call__(self, x):  # (B, H, W, C)
    h = nn.Conv(self.dim, (3, 3), padding="SAME", feature_group_count=self.dim,
                use_bias=True, name="dw")(x)
    h = nn.LayerNorm(epsilon=1e-6, name="norm")(h)
    h = nn.Dense(4 * self.dim, name="fc1")(h)
    # Flax's linen gelu defaults to approximate=True (tanh). The reference uses torch's
    # exact erf gelu; at |x|~145 that difference is 4.7e-4 in fp64 and shows up in the
    # decoded pixels, so pin it explicitly.
    h = nn.gelu(h, approximate=False)
    h = nn.Dense(self.dim, name="fc2")(h)
    gamma = self.param("gamma", nn.initializers.zeros, (self.dim,))
    return x + h * gamma


def pixel_shuffle2(x):
  """NHWC pixel shuffle by 2, matching torch's channel order (c*r^2 + i*r + j)."""
  b, h, w, c4 = x.shape
  c = c4 // 4
  y = x.reshape(b, h, w, c, 2, 2)
  y = jnp.transpose(y, (0, 1, 4, 2, 5, 3))
  return y.reshape(b, 2 * h, 2 * w, c)


def lat_patches(lat):
  """(C, h, w) latent -> (h/2, w/2, 4C) token patches, as the reference does."""
  if lat.ndim == 4:
    lat = lat[0]
  c, h, w = lat.shape
  if h % 2 or w % 2:
    raise PixelHeadError(f"latent {h}x{w} is not divisible by the 2x2 DiT patch")
  y = lat.reshape(c, h // 2, 2, w // 2, 2)
  y = jnp.transpose(y, (1, 3, 0, 2, 4))
  return y.reshape(h // 2, w // 2, 4 * c)


class PixelHead(nn.Module):
  """ConvNeXt readout: features + x_t patches -> RGB in [-1, 1]."""
  d_feat: int = 2048
  d_lat: int = LAT_PATCH_DIM
  width: int = 1024
  depth: int = 3
  up: int = 1
  channels: int = 3

  @nn.compact
  def __call__(self, f, lat):
    """f (gh, gw, d_feat), lat (gh, gw, d_lat) -> (channels, gh*patch, gw*patch)."""
    if f.shape[:2] != lat.shape[:2]:
      raise PixelHeadError(f"feature grid {f.shape[:2]} != latent grid {lat.shape[:2]}")
    h = nn.Dense(self.width, name="proj")(f) + nn.Dense(self.width, name="lat_embed")(lat)
    h = h[None]  # (1, gh, gw, width)
    for i in range(self.depth - self.up):
      h = ConvNeXtBlock(self.width, name=f"blocks_{i}")(h)
    w_out = self.width
    if self.up:
      w_out = self.width // 2
      u = nn.LayerNorm(epsilon=1e-6, name="up_norm")(h)
      u = nn.Dense(4 * w_out, name="up_proj")(u)
      h = pixel_shuffle2(u)
      for i in range(self.up):
        h = ConvNeXtBlock(w_out, name=f"up_blocks_{i}")(h)
    h = nn.LayerNorm(epsilon=1e-6, name="norm_out")(h)
    patch = PATCH // 2 if self.up else PATCH
    tok = nn.Dense(patch * patch * self.channels, name="out")(h)  # (1, H, W, p*p*C)
    n, m = h.shape[1], h.shape[2]
    c = self.channels
    img = tok.reshape(n, m, patch, patch, c)
    img = jnp.transpose(img, (4, 0, 2, 1, 3))
    return img.reshape(c, n * patch, m * patch)


# --- weight loading ----------------------------------------------------------

def _read_safetensors(path):
  """Minimal safetensors reader: header + raw buffers -> {name: np.ndarray}."""
  with open(path, "rb") as fh:
    blob = fh.read()
  if len(blob) < 8:
    raise PixelHeadError(f"{path}: too short to be a safetensors file")
  n = struct.unpack("<Q", blob[:8])[0]
  if n <= 0 or 8 + n > len(blob):
    raise PixelHeadError(f"{path}: bad safetensors header length {n}")
  header = json.loads(blob[8:8 + n].decode("utf-8"))
  meta = header.pop("__metadata__", {}) or {}
  body = memoryview(blob)[8 + n:]
  dtypes = {"F32": np.float32, "F16": np.float16, "BF16": None,
            "I64": np.int64, "I32": np.int32, "U8": np.uint8}
  tensors = {}
  for name, info in header.items():
    start, end = info["data_offsets"]
    dt = dtypes.get(info["dtype"])
    if dt is None:
      raise PixelHeadError(f"{path}: {name} has unsupported dtype {info['dtype']}")
    arr = np.frombuffer(body[start:end], dtype=dt)
    tensors[name] = arr.reshape(info["shape"])
  return tensors, meta


def _dense_kernel(w):
  """torch Linear weight (out, in) -> Flax Dense kernel (in, out)."""
  return jnp.asarray(np.ascontiguousarray(w.T))


def _depthwise_kernel(w):
  """torch Conv2d depthwise weight (C, 1, kh, kw) -> Flax (kh, kw, 1, C)."""
  return jnp.asarray(np.ascontiguousarray(np.transpose(w, (2, 3, 1, 0))))


def _conv_block_params(t, prefix, dim):
  if f"{prefix}.fc1.weight" not in t:
    raise PixelHeadError(f"missing {prefix}.fc1.weight")
  return {
      "dw": {"kernel": _depthwise_kernel(t[f"{prefix}.dw.weight"]),
             "bias": jnp.asarray(t[f"{prefix}.dw.bias"])},
      "norm": {"scale": jnp.asarray(t[f"{prefix}.norm.weight"]),
               "bias": jnp.asarray(t[f"{prefix}.norm.bias"])},
      "fc1": {"kernel": _dense_kernel(t[f"{prefix}.fc1.weight"]),
              "bias": jnp.asarray(t[f"{prefix}.fc1.bias"])},
      "fc2": {"kernel": _dense_kernel(t[f"{prefix}.fc2.weight"]),
              "bias": jnp.asarray(t[f"{prefix}.fc2.bias"])},
      "gamma": jnp.asarray(t[f"{prefix}.gamma"]),
  }


def head_shape(path):
  """(d_feat, width, depth, up, channels, meta) from the file itself."""
  tensors, meta = _read_safetensors(path)
  proj = tensors.get("proj.weight")
  if proj is None:
    raise PixelHeadError(f"{path}: no proj.weight (not a pixel head)")
  d_feat, width = int(proj.shape[1]), int(proj.shape[0])
  n_blocks = len({k.split(".")[1] for k in tensors if k.startswith("blocks.")})
  n_up = len({k.split(".")[1] for k in tensors if k.startswith("up_blocks.")})
  return d_feat, width, n_blocks + n_up, n_up, int(meta.get("channels", "3")), meta, tensors


def parse_sigma_ranges(stamp):
  """'0.0577-0.1364,0.5' -> ((0.0577, 0.1364), (0.5, 0.5)); None -> the released default."""
  if not stamp:
    return DEFAULT_SIGMA_RANGES
  out = []
  for part in str(stamp).split(","):
    part = part.strip()
    if not part:
      continue
    if "-" in part[1:]:
      lo, hi = part.split("-", 1)
      out.append((float(lo), float(hi)))
    else:
      out.append((float(part), float(part)))
  return tuple(out) or DEFAULT_SIGMA_RANGES


def out_of_range(sigma, ranges=DEFAULT_SIGMA_RANGES):
  return not any(lo - SIGMA_TOL <= sigma <= hi + SIGMA_TOL for lo, hi in ranges)


def load_pixel_head(path, dtype=jnp.float32):
  """Load a released pixel head -> (module, params, info dict).

  `dtype` applies to the weights only. The reference runs its output layer in fp32, and
  `decode` always promotes the final linear to fp32 regardless of this setting.
  """
  d_feat, width, depth, up, channels, meta, t = head_shape(path)
  if "x2_depth" in meta:
    raise PixelHeadError(f"{path}: this is an x2 head; only x1 files are supported here")
  if channels != 3:
    raise PixelHeadError(f"{path}: {channels}-channel (RGBA) head is not supported here")
  head = PixelHead(d_feat=d_feat, width=width, depth=depth, up=up, channels=channels)
  params = {
      "proj": {"kernel": _dense_kernel(t["proj.weight"]), "bias": jnp.asarray(t["proj.bias"])},
      "lat_embed": {"kernel": _dense_kernel(t["lat_embed.weight"]),
                    "bias": jnp.asarray(t["lat_embed.bias"])},
      "norm_out": {"scale": jnp.asarray(t["norm_out.weight"]),
                   "bias": jnp.asarray(t["norm_out.bias"])},
      "out": {"kernel": _dense_kernel(t["out.weight"]), "bias": jnp.asarray(t["out.bias"])},
  }
  for i in range(depth - up):
    params[f"blocks_{i}"] = _conv_block_params(t, f"blocks.{i}", width)
  if up:
    w_out = width // 2
    params["up_norm"] = {"scale": jnp.asarray(t["up_norm.weight"]),
                         "bias": jnp.asarray(t["up_norm.bias"])}
    params["up_proj"] = {"kernel": _dense_kernel(t["up_proj.weight"]),
                         "bias": jnp.asarray(t["up_proj.bias"])}
    for i in range(up):
      params[f"up_blocks_{i}"] = _conv_block_params(t, f"up_blocks.{i}", w_out)
  if dtype != jnp.float32:
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=dtype)
                                    if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a,
                                    params)
  info = {
      "width": width, "depth": depth, "up": up, "channels": channels,
      "d_feat": d_feat, "patch": PATCH // 2 if up else PATCH,
      "sigma_ranges": parse_sigma_ranges(meta.get("sigma")),
      "meta": dict(meta), "path": str(path),
  }
  return head, params, info


def decode(head, params, feats, lat, dtype=jnp.float32):
  """Features (gh, gw, d_feat) + latent (C, h, w) -> uint8 RGB (H, W, 3)."""
  img = head.apply({"params": params}, jnp.asarray(feats, dtype), lat_patches(jnp.asarray(lat, dtype)))
  img = jnp.clip(img * 0.5 + 0.5, 0.0, 1.0)  # [-1,1] -> [0,1]
  arr = np.asarray(jnp.transpose(img, (1, 2, 0)))
  return (arr * 255.0).round().astype(np.uint8)
