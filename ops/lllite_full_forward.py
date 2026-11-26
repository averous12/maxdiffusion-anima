"""Full-transformer CPU check for the LLLite scan path.

The block-level test proved the scan plumbing; this drives the REAL
`FlaxAnimaCosmosTransformer.__call__` end to end (patchify, time embedding, rope, scan,
final layer) with LLLite attached, so a failure surfaces on CPU instead of burning TPU.

Asserts, on a 2-layer model with a tiny latent:
  1. it runs at all with lllite_w attached (the path the server/TPU uses)
  2. strength 0 is bit-identical to no LLLite
  3. non-zero strength changes the output
  4. the change grows with strength

Run:  python lllite_full_forward.py
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

LAYERS = 2
LAT_H = LAT_W = 8          # latent grid -> (8/2)^2 = 16 tokens
TOKENS = (LAT_H // 2) * (LAT_W // 2)
CTX_LEN = 8


def main():
  ckpt = ("hf_cache/models--toyxyz--Anima_ControlNet-LLLite/snapshots/"
          "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d/anima_3dpose_6200.safetensors")
  ctrl = ("hf_cache/models--toyxyz--Anima_ControlNet-LLLite/snapshots/"
          "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d/example_workflow/3dpose_example.png")

  jl = LL.load_lllite(ckpt)
  print(f"checkpoint: blocks={jl.config.block_count} model_dim={jl.config.model_dim}")
  if jl.config.model_dim != 2048:
    raise SystemExit(f"unexpected model_dim {jl.config.model_dim}")

  # real modules for the first LAYERS blocks, stacked along the layer axis
  subset = LL.LLLite(config=jl.config, cond=jl.cond,
                     blocks={(i, t): p for (i, t), p in jl.blocks.items() if i < LAYERS})
  lllite_w = LL.stack_lllite_weights(subset, LAYERS)
  print(f"stacked layers: {tuple(lllite_w['self_attn_q']['down.kernel'].shape)}")

  # control embedding must match the DiT token grid
  n = 8                          # control image = latent * 8 (VAE scale); /16 -> latent/2
  ctrl_hw = (LAT_H * n, LAT_W * n)
  img = Image.open(ctrl).convert("RGB")
  cond_emb = LL.encode_conditioning(jl, jnp.asarray(LL.preprocess_control(img, ctrl_hw)))
  print(f"cond_emb {tuple(cond_emb.shape)} (need {TOKENS} tokens)")
  if cond_emb.shape[1] != TOKENS:
    raise SystemExit(f"token mismatch {cond_emb.shape[1]} != {TOKENS}")

  tf = AC.FlaxAnimaCosmosTransformer(layers=LAYERS, scan_blocks=True)
  rng = np.random.default_rng(0)
  lat = jnp.asarray(rng.standard_normal((1, 16, 1, LAT_H, LAT_W)).astype(np.float32))
  ctx = jnp.asarray(rng.standard_normal((1, CTX_LEN, 1024)).astype(np.float32))
  pad = jnp.zeros((1, 1, LAT_H * 8, LAT_W * 8), jnp.float32)
  tvec = jnp.asarray([0.5], jnp.float32)

  params = tf.init(jax.random.PRNGKey(0), lat, tvec, ctx, None, pad)
  print(f"init ok; param leaves {len(jax.tree_util.tree_leaves(params))}")

  base = tf.apply(params, lat, tvec, ctx, None, pad)
  print(f"\n1. plain forward ok, out {tuple(base.shape)}")

  zero = tf.apply(params, lat, tvec, ctx, None, pad,
                  lllite_w=lllite_w, cond_emb=cond_emb, lllite_strength=0.0)
  d0 = float(jnp.abs(base - zero).max())
  print(f"2. strength 0 vs no-lllite: {d0:.3e}")

  prev = 0.0
  mono = True
  print("4. effect vs strength:")
  for s in (0.5, 1.0, 2.0):
    out = tf.apply(params, lat, tvec, ctx, None, pad,
                   lllite_w=lllite_w, cond_emb=cond_emb, lllite_strength=s)
    d = float(jnp.abs(out - base).max())
    fin = bool(jnp.isfinite(out).all())
    print(f"   strength {s:4.1f} -> max|delta| {d:.4e} finite={fin}")
    mono &= d > prev
    prev = d
  print(f"3. hook live at non-zero strength: {prev > 0}")

  ok = d0 == 0.0 and prev > 0 and mono
  print("\nRESULT:", "FULL FORWARD OK" if ok else "FULL FORWARD FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
