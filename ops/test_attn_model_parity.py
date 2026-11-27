"""The whole transformer must behave the same with blocked attention.

Kernel-level parity is necessary but not sufficient: the blocking sits inside _Attention, which
is reached through _Block / _ScannedBlock (a lax.scan over layers), after RoPE, AdaN modulation
and with SnapRef's prepended reference keys. This runs a tiny real transformer both ways and
compares the output, so a shape, scan or masking interaction would show up here and not on a
TPU run.

Run from the workspace root.
"""
import sys

sys.path.insert(0, "maxdiffusion/src")


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np

    from maxdiffusion.models import anima_cosmos_flax as m

    fails = []
    eps = float(np.finfo(np.float32).eps)

    def build(**kw):
        return m.FlaxAnimaCosmosTransformer(
            in_channels=16, out_channels=16, heads=2, head_dim=32, layers=2,
            context_dim=16, adaln_dim=16, patch_size=(1, 2, 2), **kw)

    def run(model, params, rng, b=1, t=1, h=8, w=8, ctx_len=6, ref=None, bias=None):
        key = jax.random.key(rng)
        x = jax.random.normal(key, (b, 16, t, h, w), dtype=jnp.float32) * 0.3
        ts = jnp.asarray([0.5] * b, dtype=jnp.float32)
        ctx = jax.random.normal(jax.random.key(rng + 1), (b, ctx_len, 16),
                                dtype=jnp.float32) * 0.3
        mask = jnp.ones((b, ctx_len), dtype=bool)
        kw = {}
        if ref is not None:
            kw["ref_kv"] = ref
            kw["ref_bias"] = bias
        return model.apply({"params": params}, x, ts, ctx, mask, **kw)

    # ---- plain path
    model = build()
    key = jax.random.key(0)
    params = model.init(key, jnp.zeros((1, 16, 1, 8, 8), jnp.float32),
                        jnp.zeros((1,), jnp.float32), jnp.zeros((1, 6, 16), jnp.float32),
                        jnp.ones((1, 6), dtype=bool))["params"]

    ref_out = None
    for chunk in (0, 4, 8):
        m.ATTN_QCHUNK = chunk
        try:
            out = run(model, params, 7)
        except Exception as e:
            print(f"FAIL  plain chunk={chunk}: {type(e).__name__}: {e}")
            fails.append(f"plain chunk {chunk}")
            continue
        if chunk == 0:
            ref_out = np.asarray(out, dtype=np.float64)
            shape_note = f"out {tuple(out.shape)} dtype {out.dtype}"
            ok = True
        else:
            a = ref_out
            c = np.asarray(out, dtype=np.float64)
            scale = float(np.max(np.abs(a))) or 1.0
            ulp = float(np.max(np.abs(a - c))) / (eps * scale)
            shape_note = f"max_abs={float(np.max(np.abs(a - c))):.3e} ulp={ulp:.2f}"
            ok = ulp <= 16.0
        print(f"{'PASS' if ok else 'FAIL'}  plain transformer chunk={chunk:<2} ({shape_note})")
        if not ok:
            fails.append(f"plain chunk {chunk}")

    # ---- SnapRef path: reference keys prepended, additive bias over the key axis.
    # ref_kv and ref_bias arrive as scan INPUTS with a leading per-layer axis, which is the
    # runtime's packed form; anything else gets sliced per layer and mis-broadcasts.
    LAYERS, S_REF, CTX = 2, 5, 6
    k1 = jax.random.normal(jax.random.key(3), (1, S_REF, 2, 32), dtype=jnp.float32) * 0.2
    v1 = jax.random.normal(jax.random.key(4), (1, S_REF, 2, 32), dtype=jnp.float32) * 0.2
    ref_k = jnp.stack([k1] * LAYERS)          # (layers, 1, S_ref, heads, head_dim)
    ref_v = jnp.stack([v1] * LAYERS)
    # ref_bias is closed over (not sliced per layer), so it stays a single [S_ref + S] vector
    bias = jnp.linspace(-0.2, 0.2, S_REF + 16, dtype=jnp.float32)
    try:
        m.ATTN_QCHUNK = 0
        base = np.asarray(run(model, params, 7, ref=(ref_k, ref_v), bias=bias), dtype=np.float64)
        m.ATTN_QCHUNK = 4
        blk = np.asarray(run(model, params, 7, ref=(ref_k, ref_v), bias=bias), dtype=np.float64)
        scale = float(np.max(np.abs(base))) or 1.0
        ulp = float(np.max(np.abs(base - blk))) / (eps * scale)
        ok = ulp <= 16.0
        print(f"{'PASS' if ok else 'FAIL'}  snapref path (ref keys + bias) "
              f"max_abs={float(np.max(np.abs(base - blk))):.3e} ulp={ulp:.2f}")
        if not ok:
            fails.append("snapref path")
    except Exception as e:
        print(f"FAIL  snapref path: {type(e).__name__}: {e}")
        fails.append("snapref path")

    print()
    print("RESULT: MODEL PARITY OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
