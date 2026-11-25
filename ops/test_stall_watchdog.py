"""Exercise the stall watchdog from the shipped server source.

The server embeds its program in a string (scripts/anima_aesthetic_server.py), so the
watchdog cannot be imported normally. This extracts the real `_stall_watchdog` source
from that string and runs it against stubs, which tests the code that actually ships
rather than a re-implementation.

Run: python ops/test_stall_watchdog.py
"""

import ast
import os
import sys
import time

SERVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "anima_aesthetic_server.py")


def _embedded_program():
  tree = ast.parse(open(SERVER, encoding="utf-8").read())
  for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "code" for t in node.targets):
      return ast.literal_eval(node.value)
  raise AssertionError("embedded server program not found")


def _load_watchdog():
  prog = _embedded_program()
  tree = ast.parse(prog)
  fn = next((n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == "_stall_watchdog"), None)
  if fn is None:
    raise AssertionError("_stall_watchdog missing from the embedded program")
  src = ast.get_source_segment(prog, fn)

  exits, dumps, logs = [], [], []
  g = {
      "time": time, "_HEARTBEAT": {"t": time.time()}, "_RUN_ACTIVE": {"on": False},
      "STALL_IDLE_S": 0.0, "log": lambda m: logs.append(m),
      "os": type("O", (), {"_exit": staticmethod(lambda c: exits.append(c))}),
      "faulthandler": type("F", (), {"dump_traceback": staticmethod(lambda: dumps.append(1))}),
  }
  exec(compile(ast.Module(body=[fn], type_ignores=[]), "<watchdog>", "exec"), g)
  return g, src, exits, dumps, logs


def test_watchdog_exits_when_idle_stalled():
  g, src, exits, dumps, logs = _load_watchdog()
  # Pretend the loop has not iterated for a while and nothing is running.
  g["_HEARTBEAT"]["t"] = time.time() - 5
  g["STALL_IDLE_S"] = 0.01
  th = __import__("threading").Thread(target=g["_stall_watchdog"], daemon=True)
  th.start()
  th.join(timeout=13)   # the watchdog sleeps 10 s between checks
  assert exits == [1], f"watchdog did not exit(1): exits={exits} logs={logs}"
  assert dumps, "watchdog exited without dumping thread stacks"
  assert any("stalled" in m for m in logs), logs
  print("idle stall -> stack dump + exit(1)")


def test_watchdog_stays_quiet_during_a_run():
  g, src, exits, dumps, logs = _load_watchdog()
  g["_HEARTBEAT"]["t"] = time.time() - 5   # equally stale...
  g["_RUN_ACTIVE"]["on"] = True            # ...but a run is in flight
  g["STALL_IDLE_S"] = 0.01
  th = __import__("threading").Thread(target=g["_stall_watchdog"], daemon=True)
  th.start()
  th.join(timeout=13)
  assert exits == [], f"watchdog killed a live run: exits={exits}"
  print("long run in flight -> no exit")


def test_watchdog_source_has_both_guards():
  _, src, _, _, _ = _load_watchdog()
  assert 'not _RUN_ACTIVE["on"]' in src, src
  assert "os._exit(1)" in src, src
  assert "dump_traceback" in src, src
  print("source keeps the run-guard, the dump and the non-zero exit")


if __name__ == "__main__":
  tests = [test_watchdog_exits_when_idle_stalled, test_watchdog_stays_quiet_during_a_run,
           test_watchdog_source_has_both_guards]
  failed = 0
  for t in tests:
    try:
      t()
      print(f"PASS {t.__name__}")
    except AssertionError as e:
      failed += 1
      print(f"FAIL {t.__name__}: {e}")
  print(f"STALL_WATCHDOG {len(tests) - failed}/{len(tests)}")
  sys.exit(1 if failed else 0)
