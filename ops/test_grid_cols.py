"""Batch contact sheets must tile exactly when they can.

A 4-image batch rendered 3 wide: two cells were filled black, which reads as "two results are
missing" rather than "the grid is a rectangle". grid_cols() scored 3x2 (0.30) better than 2x2
(0.405) because each empty cell only cost 0.15. Holes now cost 0.6, so an exact tiling wins
whenever one exists, while a prime-ish count still gets a sensible shape.

Run from the workspace root.
"""
import sys

sys.path.insert(0, "maxdiffusion/src")


def main():
    import math

    # Import only the helper: pipeline pulls in jax, which is not needed to score layouts.
    src = open("maxdiffusion/src/maxdiffusion/runtime/pipeline.py", encoding="utf-8").read()
    start = src.index("def grid_cols(")
    end = src.index("def make_grid_u8(")
    ns = {"math": math}
    exec(compile(src[start:end], "grid_cols", "exec"), ns)
    grid_cols = ns["grid_cols"]

    fails = []
    expect = {1: 1, 2: 2, 3: 3, 4: 2, 6: 3, 8: 4, 9: 3}
    for n, cols in expect.items():
        rows = -(-n // cols)
        holes = rows * cols - n
        ok = cols == grid_cols(n, 1024, 1024) and holes == 0
        print(f"{'PASS' if ok else 'FAIL'}  n={n}: {cols}x{rows}, holes={holes} "
              f"(got {grid_cols(n, 1024, 1024)})")
        if not ok:
            fails.append(f"n={n}")

    # 5 has no exact rectangle; one hole is the best available and must not become a 1x5 strip.
    cols5 = grid_cols(5, 1024, 1024)
    holes5 = (-(-5 // cols5)) * cols5 - 5
    ok5 = holes5 <= 1 and cols5 > 1
    print(f"{'PASS' if ok5 else 'FAIL'}  n=5: {cols5} cols, holes={holes5} (best available)")
    if not ok5:
        fails.append("n=5")

    # landscape images: the shape preference still applies, exact tiling first
    for n in (4, 6):
        cols = grid_cols(n, 1536, 1024)
        print(f"      n={n} landscape -> {cols} cols")

    print()
    print("RESULT: GRID COLS OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
