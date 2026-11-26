"""Exercise the server's control-mode rules without loading a model or booting the runtime.

The rules used to be an inline `if regional and lllite: raise` buried in the request loop,
which meant the only way to check them was to run the whole TPU server. They now live in
`maxdiffusion.models.anima_control_modes`, so all seven combinations (six supported plus the
refused one) are testable on CPU in milliseconds.

Run:  PYTHONPATH=src python ops/test_control_modes.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from maxdiffusion.models.anima_control_modes import (  # noqa: E402
    SUPPORTED_CONTROL_COMBOS, ControlModeError, resolve_control_modes)

SENTINEL = object()          # stands in for an image; only `is not None` matters here

CASES = [
    # (label, request, expected mode or None-if-it-must-raise)
    ("plain (empty request)", {}, "plain"),
    ("regional", {"regional": [{"prompt": "sky"}]}, "regional"),
    ("regional via alias", {"regions": [{"prompt": "sky"}]}, "regional"),
    ("snapref", {"snapref": {"images": [SENTINEL]}}, "snapref"),
    ("lllite", {"lllite": {"image": SENTINEL}}, "lllite"),
    ("regional + snapref", {"regional": [{"prompt": "sky"}],
                            "snapref": {"images": [SENTINEL]}}, "regional+snapref"),
    ("lllite + snapref", {"lllite": {"image": SENTINEL},
                          "snapref": {"images": [SENTINEL]}}, "lllite+snapref"),
    # order must not matter
    ("declared snapref first", {"snapref": {"images": [SENTINEL]},
                                "lllite": {"image": SENTINEL}}, "lllite+snapref"),
    # must be refused
    ("regional + lllite", {"regional": [{"prompt": "sky"}],
                           "lllite": {"image": SENTINEL}}, None),
    ("all three", {"regional": [{"prompt": "sky"}], "snapref": {"images": [SENTINEL]},
                   "lllite": {"image": SENTINEL}}, None),
    # explicit opt-outs stay plain
    ("lllite disabled", {"lllite": {"image": SENTINEL, "enabled": False}}, "plain"),
    ("snapref disabled", {"snapref": {"images": [SENTINEL], "enabled": False}}, "plain"),
    ("regional empty list", {"regional": []}, "plain"),
    # a strength without a control map cannot steer anything
    ("lllite without image", {"lllite": {"strength": 2.0}}, "plain"),
    ("snapref without images", {"snapref": {"strength": 1.0}}, "plain"),
]


def main():
  print(f"supported combos: {len(SUPPORTED_CONTROL_COMBOS)}")
  for c in SUPPORTED_CONTROL_COMBOS:
    print(f"  {c}")

  # structural checks on the table itself
  ok = True
  if len(SUPPORTED_CONTROL_COMBOS) != 6:
    print(f"FAIL: expected 6 supported combos, got {len(SUPPORTED_CONTROL_COMBOS)}")
    ok = False
  if "regional+lllite" in SUPPORTED_CONTROL_COMBOS:
    print("FAIL: regional+lllite must not be supported")
    ok = False

  print()
  for label, req, expected in CASES:
    try:
      got = resolve_control_modes(req)
    except ControlModeError as e:
      if expected is None:
        print(f"  PASS  {label:26s} refused: {str(e).split('.')[0][:58]}")
      else:
        print(f"  FAIL  {label:26s} raised but expected {expected}: {e}")
        ok = False
      continue
    if expected is None:
      print(f"  FAIL  {label:26s} returned {got['mode']} but must be refused")
      ok = False
      continue
    if got["mode"] != expected:
      print(f"  FAIL  {label:26s} got {got['mode']} expected {expected}")
      ok = False
      continue
    # flags must agree with the mode name
    flags = {k for k in ("regional", "snapref", "lllite") if got[k]}
    derived = "+".join(sorted(flags)) or "plain"
    name_set = set(got["mode"].split("+")) if got["mode"] != "plain" else set()
    if flags != name_set:
      print(f"  FAIL  {label:26s} flags {sorted(flags)} disagree with mode {got['mode']}")
      ok = False
      continue
    print(f"  PASS  {label:26s} -> {got['mode']:18s} flags {derived}")

  # every produced mode must be a declared supported combo
  produced = set()
  for _, req, expected in CASES:
    if expected is None:
      continue
    produced.add(resolve_control_modes(req)["mode"])
  unknown = produced - set(SUPPORTED_CONTROL_COMBOS)
  if unknown:
    print(f"FAIL: resolver produced undeclared modes: {sorted(unknown)}")
    ok = False

  print(f"\ncases {len(CASES)}, modes exercised {len(produced)}/6")
  print("RESULT:", "CONTROL MODES OK" if ok else "CONTROL MODES FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
