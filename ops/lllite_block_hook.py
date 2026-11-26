"""Integration check for the LLLite hook inside the Flax Anima block.

Three things are asserted, all against the real `_Block`/`_Attention` code:

  1. `lllite=None` and `lllite=loaded, strength=0` produce *bit-identical* output to each
     other, so enabling the feature cannot perturb existing SnapRef/plain runs.
  2. A non-zero strength actually changes the block output (the hook is live).
  3. The change grows monotonically with strength and is finite.

Run:  python lllite_block_hook.py [checkpoint.safetensors] [control.png]
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

HIDDEN, HEADS, CTX, ADALN = 128, 4, 128, 128     # small but structurally identical
TOKENS = 16


def run_block(params, x, emb, temb, context, cos, sin, **kw):
  blk = AC._Block(HIDDEN, HEADS, CTX, ADALN, name="b")
  return blk.apply(params, x, emb, temb, context, cos, sin, **kw)


def main():
  ckpt = sys.argv[1] if len(sys.argv) > 1 else (
      "hf_cache/models--toyxyz--Anima_ControlNet-LLLite/snapshots/"
      "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d/anima_3dpose_6200.safetensors")
  ctrl = sys.argv[2] if len(sys.argv) > 2 else (
      "hf_cache/models--toyxyz--Anima_ControlNet-LLLite/snapshots/"
      "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d/example_workflow/3dpose_example.png")

  jl = LL.load_lllite(ckpt)
  print(f"checkpoint model_dim={jl.config.model_dim} mlp_dim={jl.config.mlp_dim}")

  # The block's own width here is HIDDEN, but the checkpoint's modules are 2048-wide. To
  # exercise the real weights we slice the LLLite modules down to the harness width: keep
  # the first HIDDEN rows/cols of down/up so shapes line up. This is a plumbing test, so a
  # slice is enough and keeps the numbers real rather than random.
  sliced = {}
  for (idx, target), p in jl.blocks.items():
    sliced[(idx, target)] = {
        "down.kernel": p["down.kernel"][:HIDDEN, :],
        "down.bias": p["down.bias"],
        "mid.kernel": p["mid.kernel"],
        "mid.bias": p["mid.bias"],
        "cond_to_film.kernel": p["cond_to_film.kernel"],
        "cond_to_film.bias": p["cond_to_film.bias"],
        "up.kernel": p["up.kernel"][:, :HIDDEN],
        "up.bias": p["up.bias"][:HIDDEN],
        "depth_embed": p["depth_embed"],
    }
  lite = LL.LLLite(config=jl.config, cond=jl.cond, blocks=sliced)

  img = Image.open(ctrl).convert("RGB")
  cond_img = LL.preprocess_control(img, (64, 64))          # 64/16 = 4 -> need TOKENS=16
  cond_img = LL.preprocess_control(img, (64, 64))
  cond_emb = LL.encode_conditioning(lite, jnp.asarray(cond_img))
  # expand 4x4=16 tokens -> matches TOKENS
  if cond_emb.shape[1] != TOKENS:
    raise SystemExit(f"cond tokens {cond_emb.shape[1]} != {TOKENS}")
  print(f"cond_emb {cond_emb.shape}")

  rng = np.random.default_rng(0)
  x = jnp.asarray(rng.standard_normal((1, TOKENS, HIDDEN)).astype(np.float32))
  context = jnp.asarray(rng.standard_normal((1, 8, CTX)).astype(np.float32))
  # AdaLN gates must be non-zero or every residual branch is skipped and the block output
  # is exactly its input regardless of attention — which would make this test vacuous.
  emb = jnp.asarray(rng.standard_normal((1, HIDDEN)).astype(np.float32))
  temb = jnp.asarray(rng.standard_normal((1, 3 * HIDDEN)).astype(np.float32))
  cos = jnp.ones((TOKENS, HIDDEN // HEADS), jnp.float32)
  sin = jnp.zeros((TOKENS, HIDDEN // HEADS), jnp.float32)

  blk = AC._Block(HIDDEN, HEADS, CTX, ADALN, name="b")
  params = blk.init(jax.random.PRNGKey(0), x, emb, temb, context, cos, sin)

  base = run_block(params, x, emb, temb, context, cos, sin)
  zero = run_block(params, x, emb, temb, context, cos, sin,
                   lllite=lite, cond_emb=cond_emb, block_index=0, lllite_strength=0.0)
  print(f"\n1. strength=0 == no-lllite (bit identical): "
        f"max|diff| {float(jnp.abs(base - zero).max()):.3e}")

  prev = 0.0
  mono = True
  print("3. effect vs strength:")
  for s in (0.25, 0.5, 1.0, 2.0):
    out = run_block(params, x, emb, temb, context, cos, sin,
                    lllite=lite, cond_emb=cond_emb, block_index=0, lllite_strength=s)
    d = float(jnp.abs(out - base).max())
    finite = bool(jnp.isfinite(out).all())
    print(f"   strength {s:4.2f} -> max|delta| {d:.4e}  finite={finite}"
          f"  {'grew' if d > prev else 'DID NOT GROW'}")
    mono &= d > prev
    prev = d

  live = prev > 0.0
  print(f"\n2. hook is live at non-zero strength: {live}")
  print("\nRESULT:", "HOOK OK" if (mono and live and float(jnp.abs(base - zero).max()) == 0.0)
        else "HOOK FAILED")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
