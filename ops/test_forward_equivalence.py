"""Prove the refactor did not change inference.

The refactor rests on one claim: consolidating four jitted bodies into one
`_transformer_forward` plus adapters is numerically identical to calling
`transformer.apply` directly with the same kwargs. This checks it on a real
FlaxAnimaCosmosTransformer driven by the **real** LLLite checkpoint, not by reading the diff.

For each case the reference is a direct `apply(..., **kwargs)`; the subject is the adapter.
They must agree **bitwise**, because the adapter's only job is to build identical kwargs.

Deliberately out of scope, because a hand-built fixture would test the fixture rather than
the refactor:
  * LoRA     -- leaves need a leading layer axis for the scan
  * SnapRef  -- ref_kv/ref_bias must match the recorder module's internal cache layout
Both have their per-mode kwarg mapping covered by ops/test_forward_adapters.py.

Run from the workspace root:
  PYTHONPATH=maxdiffusion/src python maxdiffusion/ops/test_forward_equivalence.py
"""
from __future__ import annotations

import ast
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # .../maxdiffusion
WORKSPACE = os.path.dirname(ROOT)            # .../colab
sys.path.insert(0, os.path.join(ROOT, "src"))

from maxdiffusion.models import anima_lllite_flax as LL  # noqa: E402
from maxdiffusion.models.anima_cosmos_flax import FlaxAnimaCosmosTransformer  # noqa: E402

WANTED = ("_transformer_forward", "transformer_forward", "lllite_forward")
PIPELINE = os.path.join(ROOT, "src", "maxdiffusion", "runtime", "pipeline.py")
CKPT = os.path.join(WORKSPACE, "hf_cache", "models--toyxyz--Anima_ControlNet-LLLite",
                    "snapshots", "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d",
                    "anima_3dpose_6200.safetensors")

LAT = 4                    # -> (4/2)^2 = 4 tokens
CTX = 4
LAYERS = 1


def load_adapters(transformer):
  src = open(PIPELINE, encoding="utf-8").read()
  tree = ast.parse(src)
  chunks = []
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANTED:
      node.decorator_list = []
      chunks.append(ast.get_source_segment(src, node))
  ns = {"jax": type("j", (), {"jit": staticmethod(lambda f: f)})(), "transformer": transformer}
  exec(compile("\n\n".join(chunks), "<adapters>", "exec"), ns)
  return ns


def main():
  if not os.path.exists(CKPT):
    print(f"SKIP: checkpoint not found at {CKPT}")
    return 2

  tf = FlaxAnimaCosmosTransformer(layers=LAYERS, scan_blocks=True)
  rng = np.random.default_rng(7)
  lat = jnp.asarray(rng.standard_normal((1, 16, 1, LAT, LAT)).astype(np.float32))
  ctx = jnp.asarray(rng.standard_normal((1, CTX, 1024)).astype(np.float32))
  pad = jnp.zeros((1, 1, LAT * 8, LAT * 8), jnp.float32)
  tv = jnp.asarray([0.4], jnp.float32)
  variables = tf.init(jax.random.PRNGKey(1), lat, tv, ctx, None, pad)
  params = variables["params"]      # adapters take the inner dict; apply() wants the wrapper

  # real LLLite module weights for the first block, stacked over the single layer
  jl = LL.load_lllite(CKPT)
  subset = LL.LLLite(config=jl.config, cond=jl.cond,
                     blocks={(i, t): p for (i, t), p in jl.blocks.items() if i == 0})
  lllite_w = LL.stack_lllite_weights(subset, LAYERS)
  cond_emb = jnp.zeros((1, (LAT // 2) ** 2, 64), jnp.float32)
  print("real lllite weights: layers "
        f"{tuple(lllite_w['self_attn_q']['down.kernel'].shape)[0]}, "
        f"cond_emb {tuple(cond_emb.shape)}")

  ns = load_adapters(tf)

  def ref(**kw):
    return tf.apply({"params": params}, lat, tv, ctx, None, pad, **kw)

  CASES = [
    ("plain", lambda: ns["transformer_forward"](params, lat, tv, ctx, pad), {}),
    ("lllite s=0", lambda: ns["lllite_forward"](params, lat, tv, ctx, pad,
                                                lllite_w, cond_emb, 0.0),
     {"lllite_w": lllite_w, "cond_emb": cond_emb, "lllite_strength": 0.0}),
    ("lllite s=1", lambda: ns["lllite_forward"](params, lat, tv, ctx, pad,
                                                lllite_w, cond_emb, 1.0),
     {"lllite_w": lllite_w, "cond_emb": cond_emb, "lllite_strength": 1.0}),
  ]

  ok = True
  plain_out = None
  for label, call, kw in CASES:
    want = np.asarray(ref(**kw))
    got = np.asarray(call())
    same = np.array_equal(got, want)
    finite = bool(np.isfinite(got).all())
    if label == "plain":
      plain_out = got
    if not (same and finite):
      d = float(np.abs(got - want).max())
      print(f"  FAIL  {label:12s} bitwise={same} max|diff|={d:.3e} finite={finite}")
      ok = False
    else:
      print(f"  PASS  {label:12s} bitwise identical to direct apply")

  # a no-op control must be a true no-op, and a live control must actually do something
  s0 = np.asarray(ns["lllite_forward"](params, lat, tv, ctx, pad, lllite_w, cond_emb, 0.0))
  s1 = np.asarray(ns["lllite_forward"](params, lat, tv, ctx, pad, lllite_w, cond_emb, 1.0))
  delta0 = float(np.abs(s0 - plain_out).max())
  live = float(np.abs(s1 - plain_out).max())
  print(f"\n  strength 0 vs plain : max|diff| = {delta0:.3e}  (expect 0)")
  print(f"  strength 1 vs plain : max|diff| = {live:.3e}  (expect > 0, hook live)")
  if delta0 != 0.0:
    print("  FAIL  strength 0 is not inert")
    ok = False
  if live == 0.0:
    print("  FAIL  strength 1 has no effect -- control is dead")
    ok = False

  print("\nRESULT:", "INFERENCE UNCHANGED" if ok else "INFERENCE CHANGED")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
