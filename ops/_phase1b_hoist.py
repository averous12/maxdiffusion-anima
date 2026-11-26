"""Phase 1b (cont.): hoist handle_request out of serve() to module scope.

Nesting it inside serve() left serve() at 553 lines -- the body had moved but not the bulk.
Hoisting fixes that, and buys something more useful than tidiness: handle_request becomes a
module-level, independently addressable unit instead of a closure only reachable by running
the server.

The cost is one name. Nested, it declared `nonlocal last_mtime`; at module scope that has no
enclosing binding, so `last_mtime` becomes a module global and serve() must declare it
`global` too. That is honest: last_mtime genuinely is cross-request state that both the loop
importer and the request path touch.
"""
from __future__ import annotations

import ast
import sys

PATH = "src/maxdiffusion/runtime/pipeline.py"


def main():
  src = open(PATH, encoding="utf-8").read()
  lines = src.splitlines(keepends=True)
  tree = ast.parse(src)

  serve = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "serve")
  hr = next(n for n in ast.walk(serve)
            if isinstance(n, ast.FunctionDef) and n.name == "handle_request")
  if not (serve.lineno < hr.lineno <= serve.end_lineno):
    print("FAIL: handle_request is not nested in serve")
    return 1

  # extract and dedent by 4 (nested at 4 -> module at 0)
  body = lines[hr.lineno - 1:hr.end_lineno]
  dedented = []
  for ln in body:
    dedented.append(ln[4:] if ln.startswith("    ") else ln)

  # nonlocal last_mtime -> fold last_mtime into the global declaration
  out_body = []
  for ln in dedented:
    if ln.strip().startswith("nonlocal "):
      names = [x.strip() for x in ln.strip()[len("nonlocal "):].split(",")]
      if names != ["last_mtime"]:
        print(f"FAIL: unexpected nonlocal list {names}")
        return 1
      continue                      # folded into the global line below
    if ln.strip().startswith("global "):
      names = [x.strip() for x in ln.strip()[len("global "):].split(",")]
      if "last_mtime" not in names:
        names.append("last_mtime")
      ln = "    global " + ", ".join(names) + "\n"
    out_body.append(ln)

  # serve() must now see last_mtime as module scope
  serve_head = lines[serve.lineno - 1:hr.lineno - 1]
  fixed_head = []
  for ln in serve_head:
    if ln.strip().startswith("global ") and "last_mtime" not in ln:
      names = [x.strip() for x in ln.strip()[len("global "):].split(",")]
      names.append("last_mtime")
      ln = "    global " + ", ".join(names) + "\n"
    fixed_head.append(ln)

  new = (lines[:serve.lineno - 1]
         + out_body + ["\n"]
         + fixed_head
         + lines[hr.end_lineno:serve.end_lineno + 1]
         + lines[serve.end_lineno + 1:])

  out = "".join(new)

  # ---- verify
  tree2 = ast.parse(out)
  compile(out, PATH, "exec")            # nonlocal-without-binding would raise here
  mod_fns = [n.name for n in tree2.body if isinstance(n, ast.FunctionDef)]
  serve2 = next(n for n in tree2.body if isinstance(n, ast.FunctionDef) and n.name == "serve")
  hr2 = next(n for n in tree2.body if isinstance(n, ast.FunctionDef)
             and n.name == "handle_request")
  loop2 = next(s for s in serve2.body if isinstance(s, ast.While))

  checks = [
    ("handle_request is module level", "handle_request" in mod_fns),
    ("no longer nested in serve",
     not any(isinstance(s, ast.FunctionDef) and s.name == "handle_request"
             for s in ast.walk(serve2))),
    ("serve() declares last_mtime global",
     "last_mtime" in next(ast.unparse(s) for s in serve2.body if isinstance(s, ast.Global))),
    ("handle_request declares last_mtime global",
     "last_mtime" in next(ast.unparse(s) for s in hr2.body if isinstance(s, ast.Global))),
    ("no nonlocal left", not any(isinstance(s, ast.Nonlocal) for s in ast.walk(hr2))),
    ("try preserved", any(isinstance(s, ast.Try) for s in ast.walk(hr2))),
    ("serve() still calls it", "handle_request(d, gen_id)" in ast.unparse(loop2)),
    ("serve() shrunk", (serve2.end_lineno - serve2.lineno + 1) < 200),
  ]
  for n, ok in checks:
    print(("  PASS  " if ok else "  FAIL  ") + n)
  if not all(ok for _, ok in checks):
    print("aborting; file not written")
    return 1

  open(PATH, "w", encoding="utf-8").write(out)
  print(f"\nwritten. serve() {serve2.end_lineno - serve2.lineno + 1} lines, "
        f"handle_request {hr2.end_lineno - hr2.lineno + 1} lines, both module level")
  return 0


if __name__ == "__main__":
  sys.exit(main())
