"""Phase 1c: move the boot statements into boot() so importing the module has no side effects.

One-shot: it has already been applied to pipeline.py, and re-running it now fails at the
"exactly one serve() call" check because the call lives inside the __main__ guard.

This is the unlock. While the module boots at import, nothing can be tested: importing it
downloads 6.6 GB of weights. After this, `import maxdiffusion.runtime.pipeline` is free and
`handle_request` can be driven with a stub.

Three preconditions were checked on the real file before cutting anything, because each one
would produce a silent break rather than an error:

  1. no executing statement before the last import touches os.environ / sys.path /
     sys.modules / warnings / cwd  -> hoisting imports above them changes nothing
  2. no decorator references a name that an executing statement assigns -> defs can be
     hoisted above boot()
  3. no default argument references such a name -> defs can be hoisted above boot()

(2) and (3) matter because both are evaluated at def time, not call time.

What moves: 33 imports to the top, 51 defs after them, and every executing statement into
boot() in its original relative order -- order among executing statements is preserved
exactly, which is what keeps the JAX config calls where they were.
"""
from __future__ import annotations

import ast
import symtable
import sys

PATH = "src/maxdiffusion/runtime/pipeline.py"


def main():
  src = open(PATH, encoding="utf-8").read()
  tree = ast.parse(src)
  lines = src.splitlines(keepends=True)

  imports, defs, execs = [], [], []
  for n in tree.body:
    if isinstance(n, (ast.Import, ast.ImportFrom)):
      imports.append(n)
    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      defs.append(n)
    else:
      execs.append(n)

  # drop the trailing `serve()` call from the executing set -- it moves into the guard
  calls = [e for e in execs if isinstance(e, ast.Expr) and isinstance(e.value, ast.Call)
           and getattr(e.value.func, "id", "") == "serve"]
  if len(calls) != 1:
    print(f"FAIL: expected one serve() call, found {len(calls)}")
    return 1
  execs = [e for e in execs if e is not calls[0]]

  # names boot must publish: those any module function reads as a module global
  top = symtable.symtable(src, PATH, "exec")
  fn_globals = set()
  for child in top.get_children():
    if child.get_type() != "function":
      continue
    for t in [child] + list(child.get_children()):
      for s in t.get_symbols():
        if s.is_global() and (s.is_referenced() or s.is_declared_global()):
          fn_globals.add(s.get_name())
  boot_assigned = {x.id for e in execs for x in ast.walk(e)
                   if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
  needed = sorted(boot_assigned & fn_globals)
  print(f"imports {len(imports)}  defs {len(defs)}  boot statements {len(execs)}")
  print(f"boot must publish {len(needed)} globals")

  def seg(node):
    return "".join(lines[node.lineno - 1:node.end_lineno])

  # Several lines in this file hold more than one statement (`a = set(); b = set(); ...`),
  # so slicing per node duplicates them. Select the LINE SET instead: every line covered by
  # an executing statement, minus the lines owned by imports/defs (which are hoisted), which
  # also keeps interleaved comments and blank lines in place.
  hoisted_lines = set()
  for n in imports + defs:
    # A decorated def's node starts at the `def` line, so its decorators (: N.lineno) and
    # any comments directly above it would otherwise be left behind -- orphaned @jax.jit
    # lines indented inside boot() is a syntax error, and displaced comments are worse.
    start = min([d.lineno for d in getattr(n, "decorator_list", [])] + [n.lineno]) - 1
    while start > 0 and lines[start - 1].strip().startswith("#"):
      start -= 1
    hoisted_lines.update(range(start, n.end_lineno))
  first = min(n.lineno for n in execs) - 1
  last = max(n.end_lineno for n in execs)
  boot_lines = [i for i in range(first, last) if i not in hoisted_lines]

  def seg_indented(nodes):
    return "".join(("    " + lines[i]) if lines[i].strip() else lines[i]
                   for i in boot_lines)

  # the module docstring belongs at module scope, not inside boot()
  doc = ""
  if execs and isinstance(execs[0], ast.Expr) and isinstance(execs[0].value, ast.Constant):
    doc = seg(execs[0])
    execs = execs[1:]

  out = [doc + ("\n" if doc else ""),
         "".join(seg(n) for n in imports) + "\n",
         "".join(seg(n) + "\n\n" for n in defs),
         "\n".join([
           "def boot():",
           '    """Load the models and compile the executables. Called once, before serve().',
           "",
           "    Nothing above this line runs on import, so the module can be imported by tests",
           '    without pulling 6.6 GB of weights."""',
           "    global " + ", ".join(needed),
           ""]) + "\n",
         seg_indented(execs),
         '\nif __name__ == "__main__":\n    boot()\n    serve()\n']
  new_src = "".join(out)

  # ---- verify
  tree2 = ast.parse(new_src)
  compile(new_src, PATH, "exec")
  boot = next(n for n in tree2.body if isinstance(n, ast.FunctionDef) and n.name == "boot")

  def is_doc(node):
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)

  # module-level docstring is an Expr, but it is not an executing statement
  top_execs = [n for n in tree2.body
               if not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                     ast.AsyncFunctionDef, ast.ClassDef,
                                     ast.If))
               and not is_doc(n)]
  boot_body = [s for s in boot.body if not isinstance(s, ast.Global) and not is_doc(s)]
  checks = [
    ("boot is a function", isinstance(boot, ast.FunctionDef)),
    ("all imports kept", sum(1 for n in tree2.body
                             if isinstance(n, (ast.Import, ast.ImportFrom))) == len(imports)),
    # +1 for boot() itself
    ("all defs kept", sum(1 for n in tree2.body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                            ast.ClassDef))) == len(defs) + 1),
    ("boot holds every executing statement", len(boot_body) == len(execs)),
    ("no executing statement left at module level", not top_execs),
    ("__main__ guard runs boot then serve",
     "__main__" in ast.unparse(tree2)),
    ("nonlocal-free", not any(isinstance(s, ast.Nonlocal) for s in ast.walk(tree2))),
  ]
  for n, ok in checks:
    print(("  PASS  " if ok else "  FAIL  ") + n)
  if not all(ok for _, ok in checks):
    print("aborting; file not written")
    return 1

  # every boot statement body must be AST-identical to the original
  orig = [ast.dump(e) for e in execs]
  got = [ast.dump(s) for s in boot_body]
  if orig != got:
    print("FAIL: boot() body differs from the original statement sequence")
    return 1
  print(f"  PASS  boot body is the original {len(orig)} statements, in order")

  open(PATH, "w", encoding="utf-8").write(new_src)
  print(f"\nwritten. imports+defs at module scope, boot() {len(execs)} statements")
  return 0


if __name__ == "__main__":
  sys.exit(main())
