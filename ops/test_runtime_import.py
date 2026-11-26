"""Runtime contract test: importing the runtime must not boot it.

This is the regression guard for phases 0-1c. Before them the server was a code string that
booted on import; now `import maxdiffusion.runtime.pipeline` is supposed to cost imports and
nothing else. If a future edit moves a `snapshot_download` or a model construction back to
module scope, every test in this repo would start downloading 6.6 GB again -- silently, just
slowly. This catches that.

It also pins the surface the refactor promised: boot/serve/handle_request exist and are
callable, the control-mode table is intact, and the runtime has exactly the four jitted
functions the phase-2 consolidation left.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # .../maxdiffusion
sys.path.insert(0, os.path.join(ROOT, "src"))

MODEL_ATTRS = ("transformer", "vae", "conditioner", "t_params", "qwen3_model", "cond_params")


def main():
  # point every "/content" artefact at a scratch dir so we can prove import writes nothing
  tmp = tempfile.mkdtemp(prefix="anima_import_")
  os.environ["ANIMA_CONTENT_ROOT"] = tmp
  os.environ.setdefault("ANIMA_HF_CACHE", os.path.join(tmp, "hf"))

  t0 = time.perf_counter()
  import maxdiffusion.runtime.pipeline as P          # noqa: E402
  secs = time.perf_counter() - t0

  ok = True
  def check(label, cond, detail=""):
    nonlocal ok
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {detail}" if detail else ""))
    if not cond:
      ok = False

  print(f"import took {secs:.1f}s")

  # 1. no boot side effects
  present = [a for a in MODEL_ATTRS if hasattr(P, a)]
  check("no model state after import", not present, f"found {present}" if present else "")
  log_file = os.path.join(tmp, "anima_server.log")
  check("import did not open the server log", not os.path.exists(log_file))
  check("import did not create the request file",
        not os.path.exists(os.path.join(tmp, "anima_request.json")))

  # 2. the promised surface
  for name in ("boot", "serve", "handle_request"):
    check(f"{name}() is callable", callable(getattr(P, name, None)))
  check("control modes intact", len(P.SUPPORTED_CONTROL_COMBOS) == 6,
        f"{len(P.SUPPORTED_CONTROL_COMBOS)} combos")
  check("regional+lllite still refused",
        "regional+lllite" not in P.SUPPORTED_CONTROL_COMBOS)

  # 3. the four jitted functions phase 2 left behind
  src = open(os.path.join(ROOT, "src", "maxdiffusion", "runtime", "pipeline.py"),
             encoding="utf-8").read()
  tree = ast.parse(src)
  jitted = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.decorator_list]
  check("exactly 4 jitted functions", len(jitted) == 4, str(jitted))
  check("_transformer_forward is the shared body", "_transformer_forward" in jitted)

  # 4. the loop is a function, not module-level
  check("no module-level while loop",
        not any(isinstance(n, ast.While) for n in tree.body))

  print(f"\nRESULT: {'RUNTIME CONTRACT OK' if ok else 'RUNTIME CONTRACT FAILED'}")
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
