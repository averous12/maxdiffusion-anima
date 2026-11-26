"""Prove the consolidated forward forwards exactly what the four old bodies forwarded.

Phase 2 collapsed four near-identical jitted bodies (plain, snapref, lllite, lllite+snapref)
into one `_transformer_forward` plus thin adapters. The risk is not arithmetic -- it is kwarg
mapping: if an adapter leaks `None` into the apply() call, jax.jit sees a different pytree
structure and the mode silently gets a different executable (or a slow retrace), and if it
drops a real argument the feature disappears.

So this test stubs `jax.jit` and `transformer`, extracts the real source out of pipeline.py
(which executes on import and therefore cannot be imported), and asserts the exact kwargs
each adapter produces.

Run:  PYTHONPATH=src python ops/test_forward_adapters.py
"""
from __future__ import annotations

import ast
import os
import sys

WANTED = ("_transformer_forward", "transformer_forward", "snapref_target_forward",
          "lllite_forward", "lllite_snapref_target_forward")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # .../maxdiffusion
PIPELINE = os.path.join(ROOT, "src", "maxdiffusion", "runtime", "pipeline.py")


class _StubTransformer:
  """Records the kwargs of each apply() call instead of running a model."""

  def __init__(self):
    self.calls = []

  def apply(self, params, lat, t_vec, ctx, extra, pad, **kwargs):
    self.calls.append(kwargs)
    return "OUT"


def _load_adapters():
  """Exec just the five relevant defs, with jax.jit and transformer stubbed."""
  src = open(PIPELINE, encoding="utf-8").read()
  tree = ast.parse(src)
  chunks = []
  for node in tree.body:
    name = getattr(node, "name", None)
    if name in WANTED and isinstance(node, ast.FunctionDef):
      # keep decorators off: jax.jit is stubbed to identity anyway
      node.decorator_list = []
      chunks.append(ast.get_source_segment(src, node))
  missing = set(WANTED) - {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
  if missing:
    raise SystemExit(f"defs missing from {PIPELINE}: {sorted(missing)}")

  stub = _StubTransformer()
  ns = {"jax": type("j", (), {"jit": staticmethod(lambda f: f)})(), "transformer": stub}
  exec(compile("\n\n".join(chunks), "<adapters>", "exec"), ns)
  return ns, stub


def main():
  ns, stub = _load_adapters()
  L, K, B = {"lora": 1}, {"kv": 2}, {"bias": 3}
  W, E = {"self_attn_q": 4}, {"emb": 5}
  A = (None, None, None)

  CASES = [
    # (label, call, expected kwargs)
    ("plain", lambda: ns["transformer_forward"](None, "lat", "t", "ctx", "pad"), {}),
    ("snapref lora only",
     lambda: ns["snapref_target_forward"](None, "lat", "t", "ctx", "pad", L, None, None),
     {"lora": L}),
    ("snapref lora+ref",
     lambda: ns["snapref_target_forward"](None, "lat", "t", "ctx", "pad", L, K, B),
     {"lora": L, "ref_kv": K, "ref_bias": B}),
    ("snapref ref only",
     lambda: ns["snapref_target_forward"](None, "lat", "t", "ctx", "pad", None, K, B),
     {"ref_kv": K, "ref_bias": B}),
    ("lllite",
     lambda: ns["lllite_forward"](None, "lat", "t", "ctx", "pad", W, E, 2.0),
     {"lllite_w": W, "cond_emb": E, "lllite_strength": 2.0}),
    ("lllite strength 0 still passes",
     lambda: ns["lllite_forward"](None, "lat", "t", "ctx", "pad", W, E, 0.0),
     {"lllite_w": W, "cond_emb": E, "lllite_strength": 0.0}),
    ("lllite+snapref",
     lambda: ns["lllite_snapref_target_forward"](None, "lat", "t", "ctx", "pad",
                                                 L, K, B, W, E, 3.0),
     {"lora": L, "ref_kv": K, "ref_bias": B,
      "lllite_w": W, "cond_emb": E, "lllite_strength": 3.0}),
  ]

  ok = True
  for label, call, expected in CASES:
    stub.calls.clear()
    out = call()
    got = stub.calls[0] if stub.calls else None
    if got != expected:
      print(f"  FAIL  {label:32s} kwargs {sorted(got or {})} != {sorted(expected)}")
      ok = False
      continue
    if out != "OUT":
      print(f"  FAIL  {label:32s} did not return the forward result")
      ok = False
      continue
    leaked = [k for k, v in got.items() if v is None]
    if leaked:
      print(f"  FAIL  {label:32s} leaked None for {leaked} -- would change the trace")
      ok = False
      continue
    print(f"  PASS  {label:32s} -> apply({', '.join(sorted(got)) or 'no kwargs'})")

  # the whole point: exactly one jitted body remains among the four modes
  src = open(PIPELINE, encoding="utf-8").read()
  tree = ast.parse(src)
  decorated = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.decorator_list]
  print(f"\njitted functions in the runtime: {decorated}")
  for name in ("transformer_forward", "snapref_target_forward", "lllite_forward",
               "lllite_snapref_target_forward"):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    if node.decorator_list:
      print(f"  FAIL  {name} is still separately jitted")
      ok = False
  if "_transformer_forward" not in decorated:
    print("  FAIL  _transformer_forward is not jitted")
    ok = False

  print("\nRESULT:", "FORWARD ADAPTERS OK" if ok else "FORWARD ADAPTERS FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
