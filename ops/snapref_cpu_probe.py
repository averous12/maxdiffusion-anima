"""Local CPU probe for the SnapRef backend path — real weights, real timing.

Steps: load the Anima base transformer (single-file ComfyUI checkpoint), convert to the Flax
tree, load the SnapRef v1a adapter, then run

  1. a plain 1-step forward (the no-SnapRef path, for a baseline), and
  2. the SnapRef two-pass forward: one reference pass that records every layer's self-attention
     K/V, then one target forward that reads them.

Reports wall time, peak RSS and the effect of the adapter (target output must change when the
reference is attached; the reference pass must depend on the picture).
"""
import gc
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "src")
jax.config.update("jax_enable_x64", False)

from maxdiffusion.models import anima_snapref_flax as sf            # noqa: E402
from maxdiffusion.models.anima_cosmos_flax import (FlaxAnimaCosmosTransformer,   # noqa: E402
                                                   convert_anima_aesthetic_weights)

HF = os.environ.get("ANIMA_LOCAL_HF", "E:/Workspace/colab/hf_cache")
CKPT = os.environ.get("ANIMA_CKPT", "")
SNAPREF = os.environ.get("SNAPREF_PATH", "E:/Workspace/colab/snapref_ref/snapref_v1a.safetensors")


def rss_gb():
  """Peak RSS in GB (psutil if present, else ctypes on Windows)."""
  try:
    import psutil
    return psutil.Process().memory_info().rss / 1e9
  except Exception:
    pass
  try:
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
      _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                  ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                  ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                  ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                  ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                  ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                             ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 1e9
  except Exception:
    return float("nan")


def main():
  ckpt = CKPT or _find_ckpt()
  print(f"checkpoint: {ckpt}", flush=True)
  print(f"adapter:    {SNAPREF}", flush=True)

  t0 = time.perf_counter()
  transformer = FlaxAnimaCosmosTransformer(layers=28, scan_blocks=True)
  tv = jax.eval_shape(
      transformer.init, jax.random.key(2),
      jax.ShapeDtypeStruct((1, 16, 1, 16, 16), jnp.float32),
      jax.ShapeDtypeStruct((1,), jnp.float32),
      jax.ShapeDtypeStruct((1, 8, 1024), jnp.float32),
  )
  t_params = convert_anima_aesthetic_weights(ckpt, tv["params"], dtype=jnp.bfloat16, num_layers=28)
  del tv
  gc.collect()
  print(f"transformer converted in {time.perf_counter()-t0:.1f}s  rss {rss_gb():.2f} GB", flush=True)

  t0 = time.perf_counter()
  ad = sf.load_snapref(SNAPREF, num_layers=28)
  print(f"adapter loaded in {time.perf_counter()-t0:.1f}s  rss {rss_gb():.2f} GB", flush=True)
  print(ad.describe(), flush=True)

  H = W = 64                                   # latent 8x8 -> 16 target tokens (CPU-cheap)
  lat = jnp.asarray(np.random.default_rng(0).standard_normal((1, 16, 1, H // 8, W // 8)), jnp.float32)
  ctx = jnp.asarray(np.random.default_rng(1).standard_normal((1, 8, 1024)), jnp.float32)
  pad = jnp.zeros((1, 1, H, W), jnp.float32)

  @jax.jit
  def plain(p, x, t, c, pm):
    return transformer.apply({"params": p}, x, t, c, None, pm)

  @jax.jit
  def target(p, x, t, c, pm, lora, refkv, refbias):
    return transformer.apply({"params": p}, x, t, c, None, pm,
                             lora=lora, ref_kv=refkv, ref_bias=refbias)

  tgt_lora = ad.lora_tree("tgt")
  ref_lora = ad.lora_tree("ref")

  t0 = time.perf_counter()
  base = plain(t_params, lat, jnp.asarray([0.5]), ctx, pad)
  base.block_until_ready()
  print(f"plain forward (compile+run): {time.perf_counter()-t0:.1f}s  out {base.shape}", flush=True)

  # --- the reference pass: its own stream, sigma 0, ref_t position, mode offset
  ref_lat = jnp.asarray(np.random.default_rng(7).standard_normal((1, 16, 1, H // 8, W // 8)), jnp.float32)
  cos, sin = sf.reference_rope((H // 16, W // 16), (H // 16, W // 16), ad.ref_t)
  mode_vec = ad.mode_vector("cs")

  @jax.jit
  def refpass(p, x, c, pm):
    tr = FlaxAnimaCosmosTransformer(layers=28, scan_blocks=True, record_kv=True)
    return tr.apply({"params": p}, x, jnp.zeros((1,), jnp.float32), c, None, pm,
                    rope_override=(cos, sin), t_embed_add=mode_vec,
                    lora=ref_lora)

  t0 = time.perf_counter()
  out, kv = refpass(t_params, ref_lat, ctx, pad)
  kv[0][0].block_until_ready()
  print(f"reference pass (compile+run): {time.perf_counter()-t0:.1f}s  "
        f"(k,v) stacked {kv[0][0].shape}  per-layer k {kv[0][0][0].shape}  rss {rss_gb():.2f} GB", flush=True)

  sizes = [int(kv[0][0].shape[1])]
  bias = jnp.asarray(sf.key_biases(sizes, sf.reference_biases(0.0, [1.0]), 16))

  t0 = time.perf_counter()
  with_ref = target(t_params, lat, jnp.asarray([0.5]), ctx, pad, tgt_lora, kv, bias)
  with_ref.block_until_ready()
  print(f"target forward with reference (compile+run): {time.perf_counter()-t0:.1f}s", flush=True)

  t0 = time.perf_counter()
  tgt_only = target(t_params, lat, jnp.asarray([0.5]), ctx, pad, tgt_lora, None, None)
  tgt_only.block_until_ready()
  print(f"target forward, tgt LoRA only, no reference: {time.perf_counter()-t0:.1f}s", flush=True)

  d_plain = float(jnp.abs(tgt_only - base).max())
  d_ref = float(jnp.abs(with_ref - tgt_only).max())
  print(f"effect of tgt LoRA alone          : {d_plain:.6f} (must be > 0)")
  print(f"effect of adding the reference KV : {d_ref:.6f} (must be > 0)")
  assert d_plain > 0 and d_ref > 0, "adapter had no effect"

  # a different reference picture must change the target output
  kv2 = refpass(t_params, ref_lat * 0.5 + 0.25, ctx, pad)[1]
  alt = target(t_params, lat, jnp.asarray([0.5]), ctx, pad, tgt_lora, kv2, bias)
  alt.block_until_ready()
  print(f"different reference picture       : {float(jnp.abs(alt - with_ref).max()):.6f} (must be > 0)")
  assert float(jnp.abs(alt - with_ref).max()) > 0

  # strength bias must change the result too
  bias_s = jnp.asarray(sf.key_biases(sizes, sf.reference_biases(1.0, [1.0]), 16))
  strong = target(t_params, lat, jnp.asarray([0.5]), ctx, pad, tgt_lora, kv, bias_s)
  strong.block_until_ready()
  print(f"strength 0 -> 1                   : {float(jnp.abs(strong - with_ref).max()):.6f} (must be > 0)")
  assert float(jnp.abs(strong - with_ref).max()) > 0
  print("CPU PROBE OK", flush=True)


def _find_ckpt():
  root = os.path.join(HF, "models--circlestone-labs--Anima", "snapshots")
  for base, _d, files in os.walk(root):
    for f in files:
      if f == "anima-base-v1.0.safetensors":
        return os.path.join(base, f)
  raise FileNotFoundError("anima-base-v1.0.safetensors not found under " + root)


if __name__ == "__main__":
  main()
