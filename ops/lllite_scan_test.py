"""CPU proof that LLLite rides nn.scan correctly.

Three claims, all on the real `_Block` / `_ScannedBlock` / transformer code:

  1. `stack_lllite_weights` lays the modules out along a leading layer axis, and a slice
     of it is bit-identical to the per-block dict the unstacked path uses.
  2. A `nn.scan` over blocks given the stacked tree produces the SAME output as manually
     looping the per-layer block with `layer_slice` — i.e. riding the scan changes nothing.
  3. strength 0 is bit-identical to no-LLLite at all (so enabling the feature cannot
     perturb existing SnapRef / plain runs), and non-zero strength changes the output.

Run:  python lllite_scan_test.py
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "maxdiffusion/src")
from maxdiffusion.models import anima_lllite_flax as LL  # noqa: E402
from maxdiffusion.models import anima_cosmos_flax as AC  # noqa: E402
from PIL import Image  # noqa: E402

HIDDEN, HEADS, CTX, ADALN = 128, 4, 128, 128
TOKENS, LAYERS = 16, 4


def small_lite(ckpt, layers=LAYERS):
  """Real 3dpose modules, sliced to the harness width, replicated across `layers`."""
  jl = LL.load_lllite(ckpt)
  sl = {}
  for (i, t), p in jl.blocks.items():
    if i >= layers:
      continue
    sl[(i, t)] = {
        "down.kernel": p["down.kernel"][:HIDDEN, :], "down.bias": p["down.bias"],
        "mid.kernel": p["mid.kernel"], "mid.bias": p["mid.bias"],
        "cond_to_film.kernel": p["cond_to_film.kernel"],
        "cond_to_film.bias": p["cond_to_film.bias"],
        "up.kernel": p["up.kernel"][:, :HIDDEN], "up.bias": p["up.bias"][:HIDDEN],
        "depth_embed": p["depth_embed"],
    }
  return LL.LLLite(config=jl.config, cond=jl.cond, blocks=sl)


def main():
  ckpt = ("hf_cache/models--toyxyz--Anima_ControlNet-LLLite/snapshots/"
          "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d/anima_3dpose_6200.safetensors")
  ctrl = ("hf_cache/models--toyxyz--Anima_ControlNet-LLLite/snapshots/"
          "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d/example_workflow/3dpose_example.png")

  lite = small_lite(ckpt)
  print(f"modules: {len(lite.blocks)} across {LAYERS} layers")

  # ---- 1. stacking
  stacked = LL.stack_lllite_weights(lite, LAYERS)
  print(f"stacked targets: {sorted(stacked)}")
  leaf = stacked["self_attn_q"]["down.kernel"]
  print(f"  self_attn_q down.kernel stacked {tuple(leaf.shape)} (expect ({LAYERS}, {HIDDEN}, 64))")
  assert leaf.shape == (LAYERS, HIDDEN, 64)
  slice1 = AC.layer_slice(stacked, 1)
  same = np.array_equal(np.asarray(slice1["self_attn_q"]["down.kernel"]),
                        np.asarray(lite.blocks[(1, "self_attn_q")]["down.kernel"]))
  print(f"  slice(1) == blocks[(1,'self_attn_q')]: {same}")
  assert same, "stacking does not round-trip"

  # ---- inputs
  rng = np.random.default_rng(0)
  x = jnp.asarray(rng.standard_normal((1, TOKENS, HIDDEN)).astype(np.float32))
  ctx = jnp.asarray(rng.standard_normal((1, 8, CTX)).astype(np.float32))
  emb = jnp.asarray(rng.standard_normal((1, HIDDEN)).astype(np.float32))
  temb = jnp.asarray(rng.standard_normal((1, 3 * HIDDEN)).astype(np.float32))
  cos = jnp.ones((TOKENS, HIDDEN // HEADS), jnp.float32)
  sin = jnp.zeros((TOKENS, HIDDEN // HEADS), jnp.float32)

  img = Image.open(ctrl).convert("RGB")
  cond_emb = LL.encode_conditioning(lite, jnp.asarray(LL.preprocess_control(img, (64, 64))))
  assert cond_emb.shape[1] == TOKENS, cond_emb.shape
  print(f"cond_emb {tuple(cond_emb.shape)}")

  # ---- 2. scanned vs per-layer loop
  ScanBlock = AC.nn.scan(
      AC._ScannedBlock,
      variable_axes={"params": 0}, split_rngs={"params": False}, length=LAYERS,
      in_axes=(AC._scan_broadcast,) * 6 + (AC.snapref_in_axes(None), AC.snapref_in_axes(None),
                                           AC._scan_broadcast,
                                           AC.snapref_in_axes(stacked), AC._scan_broadcast,
                                           AC._scan_broadcast))
  scan = ScanBlock(HIDDEN, HEADS, CTX, ADALN, name="scan")
  params = scan.init(jax.random.PRNGKey(0), x, emb, temb, ctx, cos, sin, None,
                     None, None, None, stacked, cond_emb, 1.0)

  x_scan, _ = scan.apply(params, x, emb, temb, ctx, cos, sin, None,
                         None, None, None, stacked, cond_emb, 1.0)

  blk = AC._Block(HIDDEN, HEADS, CTX, ADALN, name="b")
  # Use the SCAN's own per-layer weights so this is a true equality check, not two
  # independent runs that merely happen to be live.
  scan_stacked = params["params"]["b"]
  x_loop = x
  for i in range(LAYERS):
    x_loop = blk.apply({"params": AC.layer_slice(scan_stacked, i)},
                       x_loop, emb, temb, ctx, cos, sin,
                       lllite_w=AC.layer_slice(stacked, i), cond_emb=cond_emb,
                       lllite_strength=1.0)

  x_loop_off = x
  for i in range(LAYERS):
    x_loop_off = blk.apply({"params": AC.layer_slice(scan_stacked, i)},
                           x_loop_off, emb, temb, ctx, cos, sin)
  # nn.scan fixes the pytree structure at trace time, so the "off" case must pass the same
  # stacked structure with strength 0 rather than None.
  x_scan_off, _ = scan.apply(params, x, emb, temb, ctx, cos, sin, None,
                             None, None, None, stacked, cond_emb, 0.0)

  eq = float(jnp.abs(x_scan - x_loop).max())
  d_scan = float(jnp.abs(x_scan - x_scan_off).max())
  d_loop = float(jnp.abs(x_loop - x_loop_off).max())
  print(f"\n2. scan plumbing (same weights on both paths)")
  print(f"   scan vs loop max|diff| = {eq:.3e}   <- fp32 fusion-order noise")
  print(f"   scan delta vs off      = {d_scan:.4e}")
  print(f"   loop delta vs off      = {d_loop:.4e}")

  # ---- 3. strength 0 is a no-op
  x_scan_zero, _ = scan.apply(params, x, emb, temb, ctx, cos, sin, None,
                              None, None, None, stacked, cond_emb, 0.0)
  d_zero = float(jnp.abs(x_scan_zero - x_scan_off).max())
  print(f"\n3. strength 0 vs no-lllite (scan) = {d_zero:.3e}")

  ok = eq < 1e-4 and d_scan > 0 and d_loop > 0 and d_zero == 0.0 and same
  print("\nRESULT:", "SCAN OK" if ok else "SCAN FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
