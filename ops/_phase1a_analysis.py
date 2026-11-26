"""Phase 1a analysis (read-only): what would `boot()` / `serve()` actually need as globals?

Uses `symtable`, not an AST guess. The earlier estimate counted `i`, `k`, `p` as "read
outside the block" when those are loop variables inside unrelated functions -- a fake global.
`symtable` reports whether a name genuinely resolves to module scope, so the numbers here are
exact rather than conservative.

The decomposition being tested:
    def boot():   ...        # statements before the main loop
    def serve():  while True: ...
    if __name__ == "__main__": boot(); serve()

Key property to confirm: the whole `while` lives inside `serve()`, so per-request names are
locals that persist across iterations by themselves -- they do not need `global`.
"""
from __future__ import annotations

import ast
import symtable
import sys

PATH = "src/maxdiffusion/runtime/pipeline.py"
LOOP_LINE = 1456          # first line of the main `while True:`


def assigned_names(nodes):
  out = set()
  for n in nodes:
    for x in ast.walk(n):
      if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
        out.add(x.id)
  return out


def global_reads(table):
  """Names this scope resolves to module scope (true globals, not its own locals)."""
  out = set()
  for sym in table.get_symbols():
    if sym.is_global() and sym.is_referenced():
      out.add(sym.get_name())
  return out


def walk_all(table):
  yield table
  for child in table.get_children():
    yield from walk_all(child)


def main():
  src = open(PATH, encoding="utf-8").read()
  tree = ast.parse(src)
  top = symtable.symtable(src, PATH, "exec")

  execs = [n for n in tree.body
           if not isinstance(n, (ast.Import, ast.ImportFrom,
                                 ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
  boot = [n for n in execs if n.end_lineno < LOOP_LINE]
  serve = [n for n in execs if n.lineno >= LOOP_LINE]
  boot_asg = assigned_names(boot)
  serve_asg = assigned_names(serve)

  # every function scope in the module, with the globals it truly reads
  fn_globals = {}
  for child in top.get_children():
    if child.get_type() == "function":
      g = set()
      for t in walk_all(child):
        # names declared global inside this function are also module scope
        g |= {s.get_name() for s in t.get_symbols() if s.is_global() and s.is_referenced()}
        g |= {s.get_name() for s in t.get_symbols() if s.is_global() and s.is_declared_global()}
      fn_globals[child.get_name()] = g

  union_fn_globals = set()
  for g in fn_globals.values():
    union_fn_globals |= g

  print(f"module-level statements: boot={len(boot)} serve={len(serve)}")
  print(f"names assigned:         boot={len(boot_asg)} serve={len(serve_asg)}")
  print(f"function scopes found:  {len(fn_globals)}")
  print()

  # --- names boot must keep global: those any function truly reads as a module global
  boot_needs = sorted(boot_asg & union_fn_globals)
  print(f"boot() must declare global: {len(boot_needs)}")
  for i in range(0, len(boot_needs), 6):
    print("   ", ", ".join(boot_needs[i:i + 6]))
  print()

  # --- names serve must keep global: only those assigned in serve AND read by other code
  # `serve` is itself one of the function scopes; exclude it from "outside" reads
  outside_serve_globals = set()
  for name, g in fn_globals.items():
    if name == "serve":
      continue
    outside_serve_globals |= g
  serve_needs = sorted(serve_asg & outside_serve_globals)
  print(f"serve() must declare global: {len(serve_needs)}")
  print("   ", serve_needs if serve_needs else "(none -- per-request state stays local)")
  print()

  temps = sorted(serve_asg - set(serve_needs))
  print(f"per-request names that become function locals: {len(temps)}")
  print("   ", ", ".join(temps[:18]), "...")
  print()

  # --- cross-iteration state sanity: these persist because the loop is INSIDE serve()
  loop_node = next(n for n in tree.body if isinstance(n, ast.While))
  persist = sorted(assigned_names(loop_node) & set(serve_needs))
  print(f"cross-iteration names needing global: {persist if persist else '(none)'}")
  print()
  print(f"VERDICT: boot needs {len(boot_needs)} globals, serve needs {len(serve_needs)}. "
        f"{len(temps)} globals become locals.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
