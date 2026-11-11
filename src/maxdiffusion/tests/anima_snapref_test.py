"""SnapRef adapter tests.

What is actually checked, and against what:

1. **No regression**: the edited transformer is compared *bit-for-bit* against the pristine
   `HEAD` copy of the same file, for a plain forward (scan and unrolled), with no SnapRef
   arguments. Any change to the default path shows up as a difference here.
2. **Independent oracle**: the SnAPRef attention (prepended reference keys + additive bias)
   is compared against a from-scratch numpy implementation of the semantics the ComfyUI node
   documents (`_masked_attention`): softmax over [ref keys + bias; target keys].
3. Adapter loading, RoPE positions, mode embedding, guidance/APG algebra.
"""
import importlib.util
import math
import os
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SNAPREF = os.environ.get("SNAPREF_PATH", "E:/Workspace/colab/snapref_ref/snapref_v1a.safetensors")

sys.path.insert(0, os.path.join(REPO, "src"))

from maxdiffusion.models import anima_snapref_flax as sf   # noqa: E402
from maxdiffusion.models.anima_cosmos_flax import (FlaxAnimaCosmosTransformer, _Attention,  # noqa: E402
                                                   cosmos_rope, cosmos_rope_from_positions)

TINY = dict(in_channels=4, out_channels=4, heads=2, head_dim=36, layers=3, context_dim=4,
            adaln_dim=4, patch_size=(1, 2, 2))


def _pristine_module():
  """The HEAD revision of anima_cosmos_flax.py, imported as its own module."""
  src = subprocess.run(["git", "-C", REPO, "show", "HEAD:src/maxdiffusion/models/anima_cosmos_flax.py"],
                       capture_output=True, check=True).stdout.decode()
  path = os.path.join(tempfile.mkdtemp(prefix="snapref_pristine_"), "anima_cosmos_pristine.py")
  with open(path, "w", encoding="utf-8") as fh:
    fh.write(src)
  spec = importlib.util.spec_from_file_location("anima_cosmos_pristine", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _tiny_inputs():
  return (jnp.asarray(np.random.default_rng(0).standard_normal((1, 4, 1, 4, 4)), dtype=jnp.float32),
          jnp.asarray([0.3], dtype=jnp.float32),
          jnp.asarray(np.random.default_rng(1).standard_normal((1, 5, 4)), dtype=jnp.float32),
          jnp.ones((1, 5), dtype=jnp.float32))          # attention mask is [B, S_ctx]


def test_mode_embedding_keeps_the_timestep_stream_2d():
  """t_embed_add must not add an axis: the whole block stack breaks if it does."""
  hs, ts, ctx, mask = _tiny_inputs()
  tr = FlaxAnimaCosmosTransformer(scan_blocks=True, **TINY)
  params = tr.init(jax.random.PRNGKey(11), hs, ts, ctx, mask)
  plain = tr.apply(params, hs, ts, ctx, mask)
  add = tr.apply(params, hs, ts, ctx, mask, t_embed_add=jnp.asarray(np.linspace(0, 0.1, TINY["heads"] * TINY["head_dim"]), jnp.float32))
  assert plain.shape == add.shape == (1, 4, 1, 4, 4)
  assert float(jnp.abs(add - plain).max()) > 1e-6, "mode embedding had no effect"
  zero = tr.apply(params, hs, ts, ctx, mask, t_embed_add=jnp.zeros((TINY["heads"] * TINY["head_dim"],), jnp.float32))
  assert jnp.array_equal(zero, plain), "a zero mode embedding must be a no-op"


def test_reference_rope_changes_with_ref_t():
  a = sf.reference_rope((8, 8), (8, 8), 10)
  b = sf.reference_rope((8, 8), (8, 8), 0)
  assert any(float(jnp.abs(x - y).max()) > 1e-6 for x, y in zip(a, b)), "ref_t must move the reference"
  # head_dim 128 -> the 64-wide frequency vector is [t(22), h(21), w(21)] and is then
  # duplicated to 128, so the temporal entries are 0:22 and 64:86; the rest is spatial.
  assert not jnp.array_equal(a[0][:, :22], b[0][:, :22]), "temporal part ignored ref_t"
  assert jnp.array_equal(a[0][:, 22:64], b[0][:, 22:64]), "spatial part must not depend on ref_t"


def test_plain_forward_is_bit_identical_to_head():
  pristine = _pristine_module()
  hs, ts, ctx, mask = _tiny_inputs()
  for scan in (True, False):
    a = FlaxAnimaCosmosTransformer(scan_blocks=scan, **TINY)
    b = pristine.FlaxAnimaCosmosTransformer(scan_blocks=scan, **TINY)
    pa = a.init(jax.random.PRNGKey(7), hs, ts, ctx, mask)
    pb = b.init(jax.random.PRNGKey(7), hs, ts, ctx, mask)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: jnp.array_equal(x, y), pa, pb)), \
        "edited transformer changed its initialisation"
    oa = a.apply(pa, hs, ts, ctx, mask)
    ob = b.apply(pb, hs, ts, ctx, mask)
    assert oa.shape == ob.shape
    assert jnp.array_equal(oa, ob), f"scan={scan}: default forward is not bit-identical (max {float(jnp.abs(oa-ob).max())})"


def test_scan_accepts_three_concatenated_reference_records():
  """Three reference records must preserve the scan layer axis through target attention."""
  hs, ts, ctx, mask = _tiny_inputs()
  tr = FlaxAnimaCosmosTransformer(scan_blocks=True, **TINY)
  params = tr.init(jax.random.PRNGKey(17), hs, ts, ctx, mask)
  records = [(jnp.zeros((TINY["layers"], 1, 2, TINY["heads"], TINY["head_dim"])),
              jnp.zeros((TINY["layers"], 1, 2, TINY["heads"], TINY["head_dim"])))
             for _ in range(3)]
  cached = sf.concat_reference_kv(records)
  bias = jnp.zeros((6 + 4,), dtype=jnp.float32)
  out = tr.apply(params, hs, ts, ctx, mask, ref_kv=cached, ref_bias=bias)
  assert out.shape == hs.shape
  assert bool(jnp.isfinite(out).all()), "multi-reference scan produced non-finite output"


def test_reference_kv_matches_numpy_oracle():
  """The Flax attention with cached reference K/V == the node's masked-attention semantics."""
  hidden, heads, head_dim = 72, 2, 36
  attn = _Attention(hidden, heads)
  rng = np.random.default_rng(3)
  x = jnp.asarray(rng.standard_normal((1, 4, hidden)), dtype=jnp.float32)
  ref = jnp.asarray(rng.standard_normal((1, 3, hidden)).astype(np.float32) * 0.7)
  params = attn.init(jax.random.PRNGKey(1), x)
  shapes = jax.eval_shape(lambda: attn.init(jax.random.PRNGKey(1), x))
  p = jax.tree_util.tree_map(lambda s: jnp.asarray(rng.standard_normal(s.shape) * 0.3, jnp.float32), shapes)

  k_r = jnp.asarray(rng.standard_normal((1, 3, heads, head_dim)), jnp.float32)
  v_r = jnp.asarray(rng.standard_normal((1, 3, heads, head_dim)), jnp.float32)
  bias = np.asarray([0.4, -0.6, 0.0] + [0.0] * 4, dtype=np.float32)

  got = attn.apply(p, x, ref_kv=(k_r, v_r), ref_bias=jnp.asarray(bias))
  want = _numpy_oracle(p, x, k_r, v_r, bias, hidden, heads, head_dim)
  err = float(np.abs(np.asarray(got) - want).max())
  assert err < 1e-4, f"reference K/V attention mismatch: {err}"

  # the bias has to make a difference, and the reference keys have to be included at all
  plain = np.asarray(attn.apply(p, x))
  assert float(np.abs(np.asarray(got) - plain).max()) > 1e-3, "reference keys had no effect"
  zerob = np.asarray(attn.apply(p, x, ref_kv=(k_r, v_r), ref_bias=jnp.zeros_like(jnp.asarray(bias))))
  assert float(np.abs(zerob - plain).max()) > 1e-3, "zero bias dropped the reference keys"


def _numpy_oracle(p, x, k_r, v_r, bias, hidden, heads, head_dim):
  """From-scratch numpy attention: softmax over [ref keys + bias; target keys].

  Mirrors ComfyUI's _masked_attention: the reference keys are concatenated *first* and get
  an additive bias on their scores; one softmax runs over the extended key set.
  """
  def d(prefix, key):
    return np.asarray(p["params"][f"{prefix}"][key]["kernel"], np.float32)

  def proj(dense, arr):
    return arr @ np.asarray(p["params"][dense]["kernel"], np.float32)

  def rms(arr, weight, eps=1e-6):
    a = arr.astype(np.float32)
    return a * (1.0 / np.sqrt(np.mean(a * a, -1, keepdims=True) + eps)) * np.asarray(weight, np.float32)

  q = proj("to_q", np.asarray(x)).reshape(1, 4, heads, head_dim)
  k = proj("to_k", np.asarray(x)).reshape(1, 4, heads, head_dim)
  v = proj("to_v", np.asarray(x)).reshape(1, 4, heads, head_dim)
  q = rms(q, p["params"]["norm_q"])
  k = rms(k, p["params"]["norm_k"])
  kk = np.concatenate([np.asarray(k_r), k], axis=1)
  vv = np.concatenate([np.asarray(v_r), v], axis=1)
  scores = np.einsum("bqhd,bkhd->bhqk", q, kk) / math.sqrt(head_dim)
  scores = scores + bias[None, None, None, :]
  w = np.exp(scores - scores.max(axis=-1, keepdims=True))
  w = w / w.sum(axis=-1, keepdims=True)
  y = np.einsum("bhqk,bkhd->bqhd", w, vv).reshape(1, 4, heads * head_dim)
  return y @ np.asarray(p["params"]["to_out"]["kernel"], np.float32)


def test_rope_positions_match_grid_form():
  """Explicit positions must reproduce the verified grid form at position zero."""
  t_pos = np.zeros(4 * 6, dtype=np.int32)
  h_pos = np.repeat(np.arange(4), 6)
  w_pos = np.tile(np.arange(6), 4)
  a = cosmos_rope_from_positions(jnp.asarray(t_pos), jnp.asarray(h_pos), jnp.asarray(w_pos), 128, (1.0, 4.0, 4.0))
  b = cosmos_rope(24, 128, jnp.float32, (1.0, 4.0, 4.0), (1, 4, 6))
  for got, want in zip(a, b):
    assert float(jnp.abs(got - want).max()) < 1e-5, "explicit-position RoPE diverges from the grid form"


def test_reference_token_axes_are_centred():
  # same size as the target: centred means offset 0, temporal position = ref_t
  t, h, w = sf.ref_token_axes((8, 8), (8, 8), 10)
  assert set(np.unique(t)) == {10}
  assert (h.min(), h.max(), w.min(), w.max()) == (0, 7, 0, 7)
  # half-size reference: centred -> offset 2
  t, h, w = sf.ref_token_axes((8, 8), (4, 4), 10)
  assert (h.min(), h.max()) == (2, 5)
  assert (w.min(), w.max()) == (2, 5)
  assert len(h) == 16


def test_load_adapter_real_file():
  if not os.path.exists(SNAPREF):
    print("skip: adapter file not present")
    return
  ad = sf.load_snapref(SNAPREF, num_layers=28)
  assert ad.meta["format"] == sf.SNAPREF_FORMAT
  assert ad.ref_t == 10 and tuple(ad.modes) == ("cs", "char")
  assert ad.mode_embed is not None and ad.mode_embed.shape == (2, 2048)
  assert set(ad.stream_names()) == {"ref", "tgt"}, ad.stream_names()
  ref, tgt = ad.lora_tree("ref"), ad.lora_tree("tgt")
  assert set(ref) == {"attn1", "attn2", "ff_in", "ff_out"}
  assert set(ref["attn1"]) == {"to_q", "to_k", "to_v", "to_out"}
  assert set(ref["attn2"]) == {"to_q", "to_out"}
  assert set(tgt) == {"attn1"} and set(tgt["attn1"]) == {"to_q", "to_k", "to_v", "to_out"}
  A, B = ref["attn1"]["to_q"]
  assert A.shape == (28, 2048, 128) and B.shape == (28, 128, 2048), (A.shape, B.shape)
  A2, B2 = tgt["attn1"]["to_q"]
  assert A2.shape == (28, 2048, 64) and B2.shape == (28, 64, 2048), (A2.shape, B2.shape)
  Af, Bf = ref["ff_in"]["self"]
  assert Af.shape == (28, 2048, 128) and Bf.shape == (28, 128, 8192), (Af.shape, Bf.shape)
  # scale folding: alpha/rank must be inside A
  assert sf.load_snapref(SNAPREF, num_layers=28, expect_layers=28).layers == 28


def test_lora_delta_matches_torch_convention():
  """(x @ A) @ B must equal torch's F.linear(F.linear(x, down), up) * (alpha/rank)."""
  if not os.path.exists(SNAPREF):
    print("skip: adapter file not present")
    return
  from safetensors import safe_open
  with safe_open(SNAPREF, framework="np") as f:
    down = f.get_tensor("anima_ref_kv.ref.blocks.0.self_attn.q_proj.lora_down").astype(np.float32)
    up = f.get_tensor("anima_ref_kv.ref.blocks.0.self_attn.q_proj.lora_up").astype(np.float32)
    alpha = float(f.get_tensor("anima_ref_kv.ref.blocks.0.self_attn.q_proj.alpha")[()])
  ad = sf.load_snapref(SNAPREF, num_layers=28)
  A = np.asarray(ad.lora_tree("ref")["attn1"]["to_q"][0][0])
  B = np.asarray(ad.lora_tree("ref")["attn1"]["to_q"][1][0])
  rng = np.random.default_rng(5)
  x = rng.standard_normal((2, down.shape[1])).astype(np.float32)
  want = (x @ down.T) @ up.T * (alpha / down.shape[0])
  got = (x @ A) @ B
  err = float(np.abs(got - want).max() / max(1e-6, np.abs(want).max()))
  assert err < 1e-4, f"LoRA delta differs from the torch convention: rel {err}"


def test_guidance_and_correction_algebra():
  assert sf.uncond_bias_for_fidelity(1.0) is None
  assert sf.uncond_bias_for_fidelity(0.5) == math.log(0.5)
  assert sf.uncond_bias_for_fidelity(0.5, ref_in_uncond=True) is None
  assert sf.uncond_bias_for_fidelity(0.5, ref_guidance_on=True) is None
  # extra call runs while sigma >= 1 - ref_end (ComfyUI flow percent_to_sigma)
  assert sf.guidance_active(0.9, 1.0) and not sf.guidance_active(0.9, 0.05)
  assert sf.guidance_active(0.5, 0.6) and not sf.guidance_active(0.2, 0.6)

  key = jax.random.PRNGKey(0)
  cond = jax.random.normal(key, (1, 4, 2, 2))
  unref = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 2, 2))
  base = jax.random.normal(jax.random.PRNGKey(2), (1, 4, 2, 2))
  g = sf.ref_guidance_combine(cond, unref, base, 4.0, 2.0)
  want = base + 2.0 * (unref - base) + 4.0 * (cond - unref)
  assert float(jnp.abs(g - want).max()) == 0.0

  pos = jax.random.normal(jax.random.PRNGKey(3), (1, 8, 2, 2)).astype(jnp.float32)
  guided = pos + 0.5 * jax.random.normal(jax.random.PRNGKey(4), pos.shape)
  out = sf.snapref_correct(guided, pos, "apg")
  flat = pos.astype(jnp.float64).reshape(1, -1)
  unit = (flat / jnp.linalg.norm(flat, axis=1, keepdims=True)).reshape(pos.shape)
  dot = jnp.sum((out - pos).astype(jnp.float64) * unit, axis=(1, 2, 3))
  assert float(jnp.abs(dot).max()) < 1e-6, "APG update is not orthogonal to the cond prediction"
  assert sf.snapref_correct(guided, pos, "none") is guided
  assert float(jnp.abs(sf.snapref_correct(guided, pos, "rescale", 0.0) - guided).max()) == 0.0


def test_reference_sizes_and_bias_layout():
  # The recorder returns (stacked_k, stacked_v), each with scan axis 0.
  recs = [(jnp.zeros((28, 1, 6, 2, 4)), jnp.zeros((28, 1, 6, 2, 4))),
          (jnp.zeros((28, 1, 5, 2, 4)), jnp.zeros((28, 1, 5, 2, 4))),
          (jnp.zeros((28, 1, 4, 2, 4)), jnp.zeros((28, 1, 4, 2, 4)))]
  kv = sf.concat_reference_kv(recs)
  assert kv[0].shape == (28, 1, 15, 2, 4)
  assert kv[1].shape == (28, 1, 15, 2, 4)
  assert sf.reference_sizes(recs) == [6, 5, 4]
  b = sf.key_biases([6, 5, 4], sf.reference_biases(0.5, [1.0, 2.0, 0.5]), 4)
  assert b.shape == (19,)
  assert np.allclose(b[:6], 0.5)
  assert np.allclose(b[6:11], 0.5 + math.log(2.0))
  assert np.allclose(b[11:15], 0.5 + math.log(0.5))
  assert np.allclose(b[15:], 0.0)
  assert sf.reference_biases(0.0, [1.0]) == [0.0]
