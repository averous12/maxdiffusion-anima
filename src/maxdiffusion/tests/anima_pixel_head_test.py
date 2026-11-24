"""Pixel-head port tests.

The last test compares this Flax port against the reference torch implementation
(`library/anima/pixel_head.py` from sorryhyun/anima_lora, vendored in the
ComfyUI-Anima-PixelDecoder node pack) on real released weights. A port that is merely
"close" is not enough: a wrong depthwise kernel orientation or a wrong pixel-shuffle
channel order still produces a plausible-looking but incorrect image.

Run directly:  python anima_pixel_head_test.py
The weights test needs ANIMA_PIXEL_HEAD_FILE (or ~/ph_ref/base.safetensors) and torch.
"""

import importlib.util
import math
import os
import sys

import numpy as np

from maxdiffusion.models import anima_pixel_head_flax as ph


def test_lat_patches_matches_reference_order():
  """(C,h,w) -> (h/2,w/2,4C) with the reference's axis order [h, w, c, i, j]."""
  c, h, w = 16, 8, 6
  lat = np.arange(c * h * w, dtype=np.float32).reshape(c, h, w)
  got = np.asarray(ph.lat_patches(lat))
  assert got.shape == (h // 2, w // 2, 4 * c), got.shape
  for i in range(h // 2):
    for j in range(w // 2):
      want = lat[:, i * 2:i * 2 + 2, j * 2:j * 2 + 2]
      want = want.reshape(c, 2, 2)
      # channel-major, then the 2x2 patch in row-major order
      assert np.array_equal(got[i, j], want.reshape(-1)), (i, j)


def test_pixel_shuffle_matches_torch():
  import torch
  import torch.nn.functional as F

  x = np.random.default_rng(0).standard_normal((1, 4, 5, 12)).astype(np.float32)
  want = F.pixel_shuffle(torch.from_numpy(x), 2).numpy()          # (1,1,10,24)
  got = np.asarray(ph.pixel_shuffle2(np.transpose(x, (0, 2, 3, 1))))  # NHWC
  got = np.transpose(got, (0, 3, 1, 2))
  assert got.shape == want.shape, (got.shape, want.shape)
  assert np.allclose(got, want, atol=1e-6), np.abs(got - want).max()


def test_head_shape_from_metadata():
  path = _weights_path()
  if path is None:
    print("SKIP head_shape: no pixel head file")
    return
  d_feat, width, depth, up, channels, meta, tensors = ph.head_shape(path)
  assert d_feat == 2048, d_feat
  assert width == 1024, width
  assert depth == 3 and up == 1, (depth, up)
  assert channels == 3, channels
  assert ph.parse_sigma_ranges(meta.get("sigma"))[0][0] > 0


def test_forward_output_geometry():
  """A random small head must produce patch*2 per token, i.e. full image resolution."""
  head = ph.PixelHead(d_feat=32, width=16, depth=2, up=1, channels=3)
  rng = np.random.default_rng(1)
  f = rng.standard_normal((4, 6, 32)).astype(np.float32)
  lat = rng.standard_normal((16, 8, 12)).astype(np.float32)
  params = head.init(__import__("jax").random.key(0), f, ph.lat_patches(lat))["params"]
  img = head.apply({"params": params}, f, ph.lat_patches(lat))
  assert img.shape == (3, 4 * 16, 6 * 16), img.shape


# --- parity against the reference torch implementation -----------------------

REF_SRC = os.environ.get("ANIMA_PIXEL_REF") or os.path.expanduser("~/ph_ref/pixel_head.py")


def _weights_path():
  p = os.environ.get("ANIMA_PIXEL_HEAD_FILE") or os.path.expanduser("~/ph_ref/base.safetensors")
  return p if os.path.isfile(p) else None


def _load_reference_module():
  if not os.path.isfile(REF_SRC):
    return None
  spec = importlib.util.spec_from_file_location("pixel_head_ref", REF_SRC)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_parity_with_reference_implementation():
  path = _weights_path()
  ref = _load_reference_module()
  if path is None or ref is None:
    print("SKIP parity: needs a pixel head file and the reference pixel_head.py")
    return
  import torch

  torch_ref = ref.load_pixel_head(path, device="cpu")
  head, params, info = ph.load_pixel_head(path)
  rng = np.random.default_rng(7)
  gh, gw = 6, 5
  f = (rng.standard_normal((gh, gw, info["d_feat"])) * 0.5).astype(np.float32)
  lat = rng.standard_normal((16, gh * 2, gw * 2)).astype(np.float32)

  with torch.no_grad():
    want = torch_ref(
        torch.from_numpy(f),
        torch.from_numpy(np.asarray(ref.lat_patches(torch.from_numpy(lat)))),
    ).numpy()
  got = np.asarray(head.apply({"params": params}, f, ph.lat_patches(lat)))

  assert got.shape == want.shape, (got.shape, want.shape)
  diff = np.abs(got - want)
  print(f"parity: shape {got.shape}, max abs diff {diff.max():.3e}, "
        f"mean {diff.mean():.3e}, p99 {np.percentile(diff, 99):.3e}")
  assert diff.max() < 1e-5, f"max abs diff {diff.max():.3e}"


def main():
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  failed = 0
  for t in tests:
    try:
      t()
      print(f"PASS {t.__name__}")
    except Exception as e:
      failed += 1
      print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
  print(f"PIXEL_HEAD_TESTS {len(tests) - failed}/{len(tests)}")
  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
