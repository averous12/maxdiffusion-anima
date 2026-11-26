"""LLLite control images arrive as paths; prove the coercion makes them work.

The bug this covers: a request carries `lllite.image` as a *path* string, `preprocess_control`
did `Image.fromarray(np.asarray(image))`, and a path becomes a (1,1) '<U21' array -- so every
server-side LLLite request died with `TypeError: Cannot handle this data type: (1, 1), <U21>`
while the CPU probes, which passed arrays, passed.

Checks:
  1. a path and the equivalent PIL image produce byte-identical preprocessed arrays
  2. bytes do too
  3. the raw string still fails if it reaches preprocess_control (documents the original bug)
  4. a missing path raises FileNotFoundError rather than a PIL type error

Run from the workspace root:  python ops/test_lllite_control_path.py <control.png>
"""
import os
import sys

import numpy as np

CONTROL = sys.argv[1] if len(sys.argv) > 1 else "samples/pose4_control.png"
SIZE = (256, 256)


def main():
    sys.path.insert(0, "maxdiffusion/src")
    from PIL import Image

    import maxdiffusion.models.anima_lllite_flax as lllite
    from maxdiffusion.runtime.pipeline import _as_control_image

    if not os.path.exists(CONTROL):
        print(f"SKIP: control image not found: {CONTROL}")
        return 2

    ref = Image.open(CONTROL).convert("RGB")
    arr_pil = lllite.preprocess_control(ref, SIZE, aspect="letterbox")
    arr_path = lllite.preprocess_control(_as_control_image(CONTROL), SIZE, aspect="letterbox")
    same_path = np.array_equal(arr_pil, arr_path)
    print(f"{'PASS' if same_path else 'FAIL'}  path == PIL image "
          f"({arr_path.shape}, max|diff| {np.abs(arr_pil - arr_path).max():.3e})")

    with open(CONTROL, "rb") as f:
        raw = f.read()
    arr_bytes = lllite.preprocess_control(_as_control_image(raw), SIZE, aspect="letterbox")
    same_bytes = np.array_equal(arr_pil, arr_bytes)
    print(f"{'PASS' if same_bytes else 'FAIL'}  bytes == PIL image")
    print(f"      control path was {len(CONTROL)} chars ('{CONTROL}')")

    # 3. the original bug, still reproducible if the coercion is bypassed
    raised = None
    try:
        lllite.preprocess_control(CONTROL, SIZE, aspect="letterbox")
    except TypeError as e:
        raised = str(e)
    ok = raised is not None and "<U" in raised
    print(f"{'PASS' if ok else 'FAIL'}  raw string still raises the original PIL type error"
          f"{' -- ' + raised if raised else ''}")

    # 4. missing path names itself
    missing_ok = False
    try:
        _as_control_image("/content/definitely_not_here.png")
    except FileNotFoundError:
        missing_ok = True
    print(f"{'PASS' if missing_ok else 'FAIL'}  missing path raises FileNotFoundError")

    ok_all = same_path and same_bytes and ok and missing_ok
    print()
    print("RESULT: CONTROL PATH OK" if ok_all else "RESULT: CONTROL PATH FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
