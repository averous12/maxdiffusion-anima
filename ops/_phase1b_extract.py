"""Phase 1b: extract the generation block into handle_request(d, gen_id).

Measured before cutting:
  * the block is a single Try at indent 8, so a nested `def` takes it with zero
    re-indentation -- a pure move
  * it writes back only `last_mtime` (nonlocal) plus the six preview trackers serve already
    declares global
  * it holds exactly one `continue` targeting the main loop (the interruption path), which
    sits AFTER `last_mtime = os.path.getmtime(REQ_PATH)` -- it refreshes the mtime itself and
    then skips the loop's trailing refresh

That continue is why this is not a pure move. A bare `return` would let the trailing refresh
run and change behaviour under a race, so handle_request returns a flag:
    False -> interruption already refreshed last_mtime; the loop must not touch it
    True  -> normal completion; the loop performs the trailing refresh

Assembly slices the original text, so the block is moved rather than retyped.
"""
from __future__ import annotations

import ast
import sys

PATH = "src/maxdiffusion/runtime/pipeline.py"
GLOBALS = ["_pv_expected", "prev_err", "prev_fail", "prev_fails_total",
           "prev_giveup", "prev_ok"]


def main():
  src = open(PATH, encoding="utf-8").read()
  lines = src.splitlines(keepends=True)
  tree = ast.parse(src)

  serve = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "serve")
  loop = next(s for s in serve.body if isinstance(s, ast.While))
  big = max([s for s in loop.body if isinstance(s, ast.Try)],
            key=lambda s: s.end_lineno - s.lineno)
  idx = [i for i, s in enumerate(loop.body) if s is big][0]
  after = loop.body[idx + 1:]
  if len(after) != 1 or not isinstance(after[0], ast.Assign):
    print(f"FAIL: expected 1 statement after the try, found {len(after)}")
    return 1
  trailing = after[0]

  conts = []

  def collect(node, in_inner):
    for c in ast.iter_child_nodes(node):
      inner = in_inner or isinstance(c, (ast.For, ast.While, ast.AsyncFor))
      if isinstance(c, ast.Continue) and not in_inner:
        conts.append(c.lineno)
      collect(c, inner)

  collect(big, False)
  if len(conts) != 1:
    print(f"FAIL: expected 1 loop-targeting continue, found {conts}")
    return 1
  cont_line = conts[0]

  # ---- slice the original text
  head = "".join(lines[:loop.lineno - 1])                 # everything before the while
  while_line = lines[loop.lineno - 1]
  between = "".join(lines[loop.lineno:big.lineno - 1])    # loop statements before the try
  block = lines[big.lineno - 1:big.end_lineno]            # the try, at indent 8
  after_loop = "".join(lines[loop.end_lineno:])           # `serve()` call etc.

  # continue -> return False, keeping its indent; the mtime refresh above it is untouched
  ind = len(lines[cont_line - 1]) - len(lines[cont_line - 1].lstrip())
  block[cont_line - (big.lineno - 1) - 1] = " " * ind + "return False\n"

  header = ("    def handle_request(d, gen_id):\n"
            '        """Prepare, denoise, decode and record one request.\n\n'
            "        Returns False when an interruption already refreshed last_mtime, so the\n"
            '        caller must not refresh it again; True otherwise."""\n'
            "        global " + ", ".join(GLOBALS) + "\n"
            "        nonlocal last_mtime\n")

  call = ("        if handle_request(d, gen_id):\n"
          "            last_mtime = os.path.getmtime(REQ_PATH)\n")

  out = (head + header + "".join(block) + "        return True\n"
         + while_line + between + call + after_loop)

  # ---- verify
  tree2 = ast.parse(out)
  serve2 = next(n for n in tree2.body if isinstance(n, ast.FunctionDef) and n.name == "serve")
  hr = next(n for n in ast.walk(serve2)
            if isinstance(n, ast.FunctionDef) and n.name == "handle_request")
  try2 = next(s for s in hr.body if isinstance(s, ast.Try))
  rets = [s for s in hr.body if isinstance(s, ast.Return)]
  loop2 = next(s for s in serve2.body if isinstance(s, ast.While))

  hr_returns = [s for s in ast.walk(hr) if isinstance(s, ast.Return)]
  returns_true = any(isinstance(r.value, ast.Constant) and r.value.value is True
                     for r in hr_returns)
  returns_false = any(isinstance(r.value, ast.Constant) and r.value.value is False
                      for r in hr_returns)
  loop_tries = [s for s in ast.walk(loop2) if isinstance(s, ast.Try)]
  checks = [
    ("try body preserved", len(try2.body) == len(big.body)),
    ("handler preserved", len(try2.handlers) == len(big.handlers)),
    ("finally preserved", len(try2.finalbody) == len(big.finalbody)),
    ("returns True and False", returns_true and returns_false),
    ("global declared", any(isinstance(s, ast.Global) for s in hr.body)),
    ("nonlocal declared", any(isinstance(s, ast.Nonlocal) for s in hr.body)),
    # the loop keeps its small mtime try; the 460-line generation try must be gone
    ("generation try removed from the loop",
     not any(len(t.body) == len(big.body) for t in loop_tries)),
    ("call present in the loop", "handle_request(d, gen_id)" in ast.unparse(loop2)),
    ("trailing refresh now conditional",
     "if handle_request" in ast.unparse(loop2)),
  ]
  for n, ok in checks:
    print(("  PASS  " if ok else "  FAIL  ") + n)
  if not all(ok for _, ok in checks):
    print("aborting; file not written")
    return 1

  compile(out, PATH, "exec")          # catches nonlocal-without-binding
  open(PATH, "w", encoding="utf-8").write(out)
  print(f"\nwritten. handle_request holds {try2.end_lineno - try2.lineno + 1} lines; "
        f"loop body is now {len(loop2.body)} statements")
  return 0


if __name__ == "__main__":
  sys.exit(main())
