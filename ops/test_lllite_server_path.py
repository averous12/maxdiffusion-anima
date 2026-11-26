"""Exercise the exact server-side LLLite path with the real checkpoint.

This is the sequence a request actually takes after the two bugs found on TPU:

  lllite.image (a path)  ->  _as_control_image  ->  preprocess_control  ->  encode_conditioning

Previously it died at step 1 (path became a (1,1) '<U21' array) and, once that was fixed, at the
last step (`'dict' object has no attribute 'config'`). Both are cheap to reproduce without any
model weights: the LLLite checkpoint is 51 MB and the encoder runs standalone.

Token count check: the encoder downsamples by 16, so 256 px -> (256/16)^2 = 256 tokens.

Run from the workspace root:
  LL=<path to anima_3dpose_6200.safetensors dir> python ops/test_lllite_server_path.py
"""
import os
import sys

LL_DIR = os.environ.get(
    "LL",
    "hf_cache/models--toyxyz--Anima_ControlNet-LLLite/snapshots/"
    "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d",
)
CONTROL = os.environ.get("CONTROL", "samples/pose4_control.png")


def main():
    sys.path.insert(0, "maxdiffusion/src")
    os.environ["ANIMA_LLLITE_DIR"] = LL_DIR
    import maxdiffusion.runtime.pipeline as pl

    # These helpers are written to run after boot(), which creates CROOT/LOG/caches. Prime the
    # minimum here so the encoder can be exercised without loading 6.6 GB of weights.
    pl.CROOT = os.environ.get("ANIMA_CROOT", os.getcwd())
    pl._LLLITE_CACHE = {}
    pl.LOG = open(os.devnull, "w")
    get_lllite, lllite_cond_emb = pl.get_lllite, pl.lllite_cond_emb

    if not os.path.isdir(LL_DIR):
        print(f"SKIP: no LLLite checkpoint at {LL_DIR}")
        return 2
    if not os.path.exists(CONTROL):
        print(f"SKIP: no control image at {CONTROL}")
        return 2

    fails = []

    ll = get_lllite("anima_3dpose_6200")
    has_obj = isinstance(ll, dict) and ll.get("lite") is not None
    print(f"{'PASS' if has_obj else 'FAIL'}  cache entry carries the LLLite object")
    if not has_obj:
        fails.append("cache entry")

    for size, want_tokens in ((256, 256), (512, 1024)):
        emb = lllite_cond_emb(ll, CONTROL, size, size)
        shape = tuple(emb.shape)
        ok = shape[0] == 1 and shape[1] == want_tokens and len(shape) == 3
        finite = bool((emb == emb).all())
        print(f"{'PASS' if ok and finite else 'FAIL'}  {size}px path -> shape {shape} "
              f"(expected (1, {want_tokens}, D)), finite={finite}")
        if not (ok and finite):
            fails.append(f"{size}px")

    # a missing file must name itself, not raise a PIL type error
    missing = False
    try:
        lllite_cond_emb(ll, "/content/nope_control.png", 256, 256)
    except FileNotFoundError:
        missing = True
    print(f"{'PASS' if missing else 'FAIL'}  missing control path raises FileNotFoundError")
    if not missing:
        fails.append("missing path")

    print()
    print("RESULT: SERVER LLLITE PATH OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
