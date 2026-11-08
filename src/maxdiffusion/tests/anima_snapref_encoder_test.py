"""Regression tests for the two SnapRef bugs the real server caught on CPU.

Both failures were shape bugs that only appear at real sizes, so they are pinned here at the
level where they were caused:

  1. the Anima/Wan VAE encodes to channels-LAST (B, T, H, W, C); the sampler, the
     transformer's patchify and the K/V cache all work in (B, C, T, H, W). Feeding the raw
     encoder output made the reference pass patchify to 16x too many tokens
     (15552 instead of 972 at a 0.25 MP reference).
  2. the reference bias must cover the reference keys *plus* the target keys, and the target
     key count is the latent grid divided by the 2x2 patch area -- not the latent pixel count
     (4096, not 16384, at 1024px), or the attention add cannot broadcast.

Run: python -c "import anima_snapref_encoder_test as t; [getattr(t,n)() for n in dir(t) if n.startswith('test_')]"
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from flax import nnx  # noqa: E402

from maxdiffusion.models import anima_snapref_flax as sf  # noqa: E402


class _StubPosterior:
  """Stands in for the VAE's DiagonalGaussianDistribution."""

  def __init__(self, z):
    self._z = z

  def mode(self):
    return self._z


class _Empty(nnx.Module):
  """Placeholder container: AutoencoderKLWanCache only walks these to count conv layers."""


class _StubEncoder(nnx.Module):
  """A VAE whose encode() returns channels-LAST latents, exactly like the real one."""

  def __init__(self, z):
    super().__init__()
    self.z = z
    self.encoder = _Empty()
    self.decoder = _Empty()

  def encode(self, x, cache=None, return_dict=False):
    return (_StubPosterior(self.z),)


def _channels_last_case():
  """(1,3,1,64,64) pixels -> (1,1,8,8,16) latents, as measured on the real VAE."""
  z = np.arange(1 * 1 * 8 * 8 * 16, dtype=np.float32).reshape(1, 1, 8, 8, 16)
  return _StubEncoder(z), np.ones((16,), np.float32), np.ones((16,), np.float32)


def test_encoder_output_is_transposed_to_channels_first():
  vae, mean, std = _channels_last_case()
  out, _ = sf.encode_reference_latent(vae, np.zeros((1, 3, 1, 64, 64), np.float32), mean, std)
  assert out.shape == (1, 16, 1, 8, 8), f"expected (1,16,1,8,8), got {out.shape}"


def test_transpose_preserves_values_in_the_right_order():
  vae, _, _ = _channels_last_case()
  one = np.ones((16,), np.float32)
  zero = np.zeros((16,), np.float32)
  out, _ = sf.encode_reference_latent(vae, np.zeros((1, 3, 1, 64, 64), np.float32), zero, one)
  # with mean 0 / std 1 the output is the raw transposed tensor
  src = np.asarray(vae.z)
  got = np.asarray(out)
  for c in (0, 7, 15):
    assert np.array_equal(got[0, c, 0], src[0, 0, :, :, c]), f"channel {c} misplaced"


def test_channels_first_input_is_left_alone():
  z = np.zeros((1, 16, 1, 8, 8), np.float32)
  vae = _StubEncoder(z)
  out, _ = sf.encode_reference_latent(vae, np.zeros((1, 3, 1, 64, 64), np.float32),
                                      np.zeros((16,), np.float32), np.ones((16,), np.float32))
  assert out.shape == (1, 16, 1, 8, 8), f"shape drifted: {out.shape}"


def test_norm_is_applied_on_the_channel_axis():
  """Per-channel mean/std must hit axis 1 after the transpose, not the last axis."""
  vae, _, _ = _channels_last_case()
  mean = np.arange(16, dtype=np.float32)
  out, _ = sf.encode_reference_latent(vae, np.zeros((1, 3, 1, 64, 64), np.float32),
                                      mean, np.ones((16,), np.float32))
  src = np.asarray(vae.z)
  got = np.asarray(out)
  for c in (0, 5, 15):
    expect = src[0, 0, 0, 0, c] - mean[c]
    assert abs(float(got[0, c, 0, 0, 0]) - float(expect)) < 1e-5, \
      f"channel {c}: got {float(got[0, c, 0, 0, 0])}, expected {float(expect)}"


def test_bias_length_is_reference_plus_target_tokens():
  bias = sf.key_biases([972], [1.0], 4096)
  assert bias.shape == (972 + 4096,), f"bias length {bias.shape}"


def test_bias_covers_the_latent_grid_divided_by_patch_area():
  """1024px -> 128x128 latent -> 2x2 patch -> 4096 target tokens, not 16384."""
  lat_hw = (128, 128)
  patch = 2
  target = (lat_hw[0] // patch) * (lat_hw[1] // patch)
  bias = sf.key_biases([972], [1.0], target)
  assert bias.shape == (972 + 4096,), f"bias length {bias.shape}"
  assert bias.shape[0] != 972 + lat_hw[0] * lat_hw[1], "the raw latent grid must not be used"


def test_reference_rope_length_matches_reference_tokens():
  """The RoPE table and the reference token count must agree, or attention cannot broadcast."""
  for (pw, ph) in [(432, 576), (512, 512), (768, 1024)]:
    rh, rw = max(1, ph // 16), max(1, pw // 16)
    cos, sin = sf.reference_rope((64, 64), (rh, rw), 10)
    assert cos.shape[0] == rh * rw, f"{pw}x{ph}: rope {cos.shape[0]} vs tokens {rh * rw}"
    assert cos.shape[1] == 128, f"rope width {cos.shape[1]}"


if __name__ == "__main__":
  names = [n for n in dir(sys.modules[__name__]) if n.startswith("test_")]
  bad = 0
  for n in sorted(names):
    try:
      globals()[n]()
      print("PASS", n)
    except Exception as exc:  # noqa: BLE001
      bad += 1
      print("FAIL", n, type(exc).__name__, str(exc)[:160])
  print(f"FAILED {bad} of {len(names)}")
  sys.exit(1 if bad else 0)
