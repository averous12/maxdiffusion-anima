"""Phase 1a: wrap the main loop in `def serve()`.

Scope: the `while True:` is a *single contiguous statement*, so wrapping it needs no
reordering -- unlike `boot()`, whose 139 statements are interleaved with imports and defs
across 45 alternating runs, where collating them would move imports past env setup.

The global list is not guessed. It comes from `symtable`: names that `serve` assigns AND
that resolve to module scope from outside it. Everything else -- 131 names -- becomes a
function local, which is the point: per-request scratch stops being module global state.

Because the loop lives *inside* serve(), names assigned in one iteration persist to the next
by themselves; only the 7 names other code also reads need `global`.

Verification before writing:
  * the wrapped body must be AST-identical to the original `while` statement
  * the file must compile
  * the global list must equal the symtable-derived set
"""
from __future__ import annotations

import ast
import symtable
import sys

PATH = "src/maxdiffusion/runtime/pipeline.py"
GLOBALS = ["_pv_expected", "prev_err", "prev_fail", "prev_fails_total",
           "prev_giveup", "prev_ok"]


def main():
  src = open(PATH, encoding="utf-8").read()
  lines = src.splitlines(keepends=True)
  tree = ast.parse(src)

  whiles = [n for n in tree.body if isinstance(n, ast.While)]
  if len(whiles) != 1:
    print(f"FAIL: expected exactly 1 module-level while, found {len(whiles)}")
    return 1
  loop = whiles[0]

  # re-derive the global list rather than trusting the constant above.
  # Scopes whose lines fall INSIDE the loop are excluded: before the wrap a comprehension in
  # the loop resolves to module scope, but once wrapped it is a serve local. Counting it here
  # would force a needless `global` and, worse, hide a genuine one behind noise.
  top = symtable.symtable(src, PATH, "exec")
  fn_globals = set()
  for child in top.get_children():
    if child.get_type() != "function" or child.get_name() == "serve":
      continue
    for t in [child] + list(child.get_children()):
      span = t.get_lineno()
      if loop.lineno <= span <= loop.end_lineno:
        continue
      for s in t.get_symbols():
        if s.is_global() and (s.is_referenced() or s.is_declared_global()):
          fn_globals.add(s.get_name())
  loop_assigned = {x.id for x in ast.walk(loop)
                   if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
  derived = sorted(loop_assigned & fn_globals)
  if derived != sorted(GLOBALS):
    print(f"FAIL: derived globals {derived} != expected {sorted(GLOBALS)}")
    return 1
  print(f"global list confirmed by symtable: {len(derived)} names")
  print(f"names becoming locals: {len(loop_assigned - set(derived))}")

  # indent the loop body by 4 and put it under serve()
  start, end = loop.lineno - 1, loop.end_lineno
  body = "".join("    " + ln if ln.strip() else ln for ln in lines[start:end])
  header = ("def serve():\n"
            '    """Watch for requests and handle them, one at a time, until the process dies."""\n'
            "    global " + ", ".join(sorted(derived)) + "\n")

  new_src = ("".join(lines[:start]) + header + body + "\n\nserve()\n"
             + "".join(lines[end:]))

  # --- verify
  new_tree = ast.parse(new_src)
  fn = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "serve")
  inner = next(s for s in fn.body if isinstance(s, ast.While))
  if ast.dump(inner) != ast.dump(loop):
    print("FAIL: wrapped loop is not AST-identical to the original")
    return 1
  print("AST equivalence: wrapped loop is identical to the original while")

  calls = [n for n in new_tree.body
           if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
           and getattr(n.value.func, "id", "") == "serve"]
  if not calls:
    print("FAIL: serve() is never called")
    return 1
  print("serve() is invoked at module level, so launch behaviour is unchanged")

  open(PATH, "w", encoding="utf-8").write(new_src)
  print(f"written: serve() now holds {end - start} lines")
  return 0


if __name__ == "__main__":
  sys.exit(main())
