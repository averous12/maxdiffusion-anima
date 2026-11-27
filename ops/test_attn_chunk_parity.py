"""Query-blocked attention must be numerically the same attention.

The change trades time for memory: the fp32 score tensor [B, heads, S, S+ref] is what makes
batch 4 impossible, and splitting the QUERIES divides it by the block count. That is only
legitimate if the maths is untouched, so this compares the blocked path against the original
single-shot einsums and reports the real difference rather than asserting a tolerance chosen to
pass.

Shapes are deliberately awkward: S not divisible by the block size exercises the padding path,
and both ref_bias (SnapRef) and a key mask are covered because they are applied inside the
score reduction.

Run from the workspace root.
"""
import sys

sys.path.insert(0, "maxdiffusion/src")


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", False)
    from maxdiffusion.models import anima_cosmos_flax as m

    fails = []
    rng = np.random.default_rng(0)

    def mk(b, s, k, h, d, bias, mask):
        q = jnp.asarray(rng.standard_normal((b, s, h, d)).astype(np.float32)) * 0.5
        kk = jnp.asarray(rng.standard_normal((b, k, h, d)).astype(np.float32)) * 0.5
        vv = jnp.asarray(rng.standard_normal((b, k, h, d)).astype(np.float32)) * 0.5
        rb = jnp.asarray(rng.standard_normal((k,)).astype(np.float32)) if bias else None
        ms = None
        if mask:
            ms = jnp.ones((b, k), dtype=bool)
            ms = ms.at[:, 1].set(False)          # one dropped key, so the mask really bites
        return q, kk, vv, rb, ms

    cases = [
        # name                B  S     K     H  D  bias  mask  chunk
        ("self S=64",         1, 64, 64, 2, 8, False, False, 16),
        ("self + ref_bias",   2, 64, 96, 2, 8, True,  False, 16),
        ("masked keys",       2, 64, 96, 2, 8, True,  True,  16),
        ("S not divisible",   1, 70, 100, 2, 8, True,  True,  16),
        ("S < chunk",         1, 12, 20, 2, 8, True,  False, 64),
        ("real geometry B1",  1, 256, 384, 4, 16, True, True, 64),
        ("real geometry B4",  4, 256, 384, 4, 16, True, True, 64),
    ]

    for name, b, s, k, h, d, bias, mask, chunk in cases:
        q, kk, vv, rb, ms = mk(b, s, k, h, d, bias, mask)
        ref = m._attend(q, kk, vv, ref_bias=rb, mask=ms, chunk=0)      # original path
        got = m._attend(q, kk, vv, ref_bias=rb, mask=ms, chunk=chunk)  # blocked
        same_shape = ref.shape == got.shape
        same_dtype = got.dtype == jnp.float32
        a = np.asarray(ref, dtype=np.float64)
        c = np.asarray(got, dtype=np.float64)
        max_abs = float(np.max(np.abs(a - c))) if same_shape else float("nan")
        scale = float(np.max(np.abs(a))) or 1.0
        max_rel = max_abs / scale
        # Principled bound, not a chosen tolerance: fp32 rounding is eps = 1.19e-7 at
        # magnitude 1, so anything within a few ulp is the same maths summed in a different
        # order. A real regression would be orders of magnitude larger.
        eps = float(np.finfo(np.float32).eps)
        ulp = max_abs / (eps * scale)
        bitwise = same_shape and np.array_equal(np.asarray(ref), np.asarray(got))
        # Bound is 16 ulp: blocking changes the matmul tiling, so the same sum is grouped
        # differently. That is not an approximation -- no term is dropped, no dtype is
        # narrowed, softmax is still fp32 over the full key axis -- but it is also not
        # guaranteed bit-identical, so this test reports the real figure instead of hiding it.
        ok = same_shape and same_dtype and ulp <= 16.0
        print(f"{'PASS' if ok else 'FAIL'}  {name:18} B={b} S={s} K={k} chunk={chunk:<4} "
              f"shape={same_shape} fp32={same_dtype} bitwise={bitwise} "
              f"max_abs={max_abs:.3e} rel={max_rel:.3e} ulp={ulp:.2f}")
        if not ok:
            fails.append(name)

    # A block size larger than S must fall back to exactly the original single shot.
    q, kk, vv, rb, ms = mk(1, 40, 60, 2, 8, True, True)
    same = np.array_equal(np.asarray(m._attend(q, kk, vv, rb, ms, chunk=0)),
                          np.asarray(m._attend(q, kk, vv, rb, ms, chunk=512)))
    print(f"{'PASS' if same else 'FAIL'}  chunk > S is bitwise identical to no chunk")
    if not same:
        fails.append("overlarge chunk")

    print()
    print("note: differences are summation-order only (fp32 eps = 1.19e-7); no term is dropped,")
    print("      no dtype is narrowed, and softmax stays fp32 over the full key axis.")
    print("RESULT: ATTENTION PARITY OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
