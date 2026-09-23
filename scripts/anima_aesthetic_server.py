import os
import subprocess
import sys

repo = "/content/maxdiffusion"
venv = repo + "/.venv"
# Prefer the repo venv when it exists; otherwise fall back to the running
# interpreter so the server also works in sessions that pip-installed into the
# system Python (the E2E runner path).
if os.path.exists(venv + "/bin/python"):
    py = venv + "/bin/python"
    _path_prefix = venv + "/bin:"
else:
    py = sys.executable
    _path_prefix = ""
env = {**os.environ, "PATH": _path_prefix + "/usr/local/bin:" + os.environ["PATH"],
       "PYTHONPATH": repo + "/src", "HF_HOME": "/content/hf_cache",
       "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}
env.pop("UV_SYSTEM_PYTHON", None)
print("server interpreter:", py, flush=True)

code = r'''
import time

LOG = open("/content/anima_server.log", "a", buffering=1)
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")

def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0

def hbm_stats():
    try:
        ms = jax.devices()[0].memory_stats() or {}
        return {"in_use_gb": round(ms.get("bytes_in_use", 0) / 1e9, 2),
                "peak_gb": round(ms.get("peak_bytes_in_use", 0) / 1e9, 2),
                "limit_gb": round(ms.get("bytes_limit", 0) / 1e9, 2)}
    except Exception:
        return {}

log("=== SERVER boot: imports ===")
import os
import json
import numpy as np
import gc
import traceback
import faulthandler
# Diagnostic only: exit=False so the watchdog can never kill a long-lived server.
# It dumps the live stack of every thread once, which is what localizes a stall.
faulthandler.dump_traceback_later(1800, exit=False)
import jax
jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32")
import jax.numpy as jnp
log("matmul precision: BF16_BF16_F32 (bf16 operands, fp32 accumulate; activations stay fp32)")

from maxdiffusion.models.anima_cosmos_flax import FlaxAnimaCosmosTransformer, convert_anima_aesthetic_weights
from maxdiffusion.models.anima_text_conditioner_flax import (
    AnimaTextConditionerConfig, FlaxAnimaTextConditioner,
    convert_anima_aesthetic_adapter_weights)
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model
from maxdiffusion.models.qwen3_utils import load_and_convert_qwen3_weights
from maxdiffusion.models.qwen_image_vae_utils import load_qwen_image_vae
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
from maxdiffusion.schedulers.scheduling_flow_match_flax import (
    FlaxFlowMatchScheduler, FlowMatchSchedulerState)
from flax import nnx
from flax.traverse_util import flatten_dict as _flatten_dict
from huggingface_hub import snapshot_download

log(f"devices: {jax.devices()}")
log(f"HBM at boot: {hbm_stats()}")

# ---------------------------------------------------------------------------
# All weight conversion runs with CPU as the default device.
#
# The converters build their trees leaf-by-leaf with jnp.asarray. With the TPU as
# the default device each leaf becomes its own host->TPU transfer: measured 227.9s
# for the transformer instead of 60.6s. One explicit device_put at the end is far
# cheaper, and it keeps the HBM peak at 3.91 GB instead of 10.21 GB.
#
# There is deliberately NO msgpack parameter cache here. Measured on the real tree,
# flax_serialization.to_bytes + write costs 76s for the 6.6 GB stacked transformer
# tree (45.9s serialize + 30.1s write) while re-converting the whole thing takes
# ~10s, so caching is a net loss. The old VAE cache write was also a latent crash
# ("TypeError: can not serialize 'State' object") that killed the server outright
# whenever the cache was invalid.
# ---------------------------------------------------------------------------
_CPU = jax.devices("cpu")[0]

CACHE_DIR = "/content/anima_cache"  # retained for the HF cache only; no param cache
REQ_PATH = "/content/anima_request.json"
PROG_PATH = "/content/anima_progress.json"
SNAP_PATH = "/content/anima_snap.png"

def write_prog(**kw):
    try:
        tmp = PROG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"t": time.time(), "rss_gb": round(rss_gb(), 2), **kw}, f)
        os.replace(tmp, PROG_PATH)
    except Exception:
        pass

t_all = time.perf_counter()
snapshot_dir = snapshot_download("circlestone-labs/Anima-Base-v1.0-Diffusers", cache_dir="/content/hf_cache")
aesthetic_snapshot = snapshot_download("circlestone-labs/Anima", cache_dir="/content/hf_cache",
                                       allow_patterns=["split_files/diffusion_models/anima-aesthetic-v1.1.safetensors"])
import shutil
aesthetic_path = "/content/aesthetic_v1.1.safetensors"
_src = os.path.join(aesthetic_snapshot, "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors")
if not os.path.exists(aesthetic_path) or os.path.getsize(aesthetic_path) != os.path.getsize(_src):
    shutil.copyfile(_src, aesthetic_path)
log(f"base weights at {snapshot_dir}")
log(f"aesthetic transformer+conditioner: {aesthetic_path} ({os.path.getsize(aesthetic_path)/1e9:.2f} GB)")

log("=== SERVER: torch tokenizer + TPU Qwen3 load (once) ===")
t0 = time.perf_counter()
from transformers import AutoConfig as HFAutoConfig, AutoTokenizer
from tokenizers import Tokenizer as _RawTokenizer
tok = AutoTokenizer.from_pretrained("circlestone-labs/Anima-Base-v1.0-Diffusers", subfolder="tokenizer")
_t5 = _RawTokenizer.from_file(os.path.join(snapshot_dir, "t5_tokenizer", "tokenizer.json"))
_t5.enable_padding(pad_id=0, pad_token="<pad>", length=512)
_t5.enable_truncation(max_length=512)
pc = HFAutoConfig.from_pretrained("circlestone-labs/Anima-Base-v1.0-Diffusers", subfolder="text_encoder")
rope_theta = getattr(pc, "rope_theta", None) or pc.rope_parameters["rope_theta"]
qcfg = FlaxQwen3Config(vocab_size=pc.vocab_size, hidden_size=pc.hidden_size,
    intermediate_size=pc.intermediate_size, num_hidden_layers=pc.num_hidden_layers,
    num_attention_heads=pc.num_attention_heads, num_key_value_heads=pc.num_key_value_heads,
    head_dim=getattr(pc, "head_dim", pc.hidden_size // pc.num_attention_heads),
    rms_norm_eps=pc.rms_norm_eps, rope_theta=rope_theta,
    max_position_embeddings=pc.max_position_embeddings, dtype=jnp.bfloat16,
    max_layer_to_run=None, is_causal=True)
qwen3_model = FlaxQwen3Model(qcfg)
qv = jax.eval_shape(qwen3_model.init, jax.random.key(0),
                    jax.ShapeDtypeStruct((1, 512), jnp.int32),
                    jax.ShapeDtypeStruct((1, 512), jnp.int32))
with jax.default_device(_CPU):
    qwen3_params = load_and_convert_qwen3_weights(os.path.join(snapshot_dir, "text_encoder"), qv["params"], qcfg)
del qv
gc.collect()
qwen3_params = jax.device_put(qwen3_params, jax.devices()[0])
qwen3_jit = jax.jit(lambda p, ids, mask: qwen3_model.apply({"params": p}, ids, mask)[0])
log(f"Qwen3 TPU model ready in {time.perf_counter()-t0:.1f}s; {hbm_stats()}")

def encode_texts(prompt, neg):
    ids=[]; masks=[]
    for p in [prompt, neg]:
        ti=tok(p,padding="max_length",max_length=512,truncation=True,return_tensors="np")
        ids.append(ti.input_ids[0]); masks.append(ti.attention_mask[0])
    qids=jnp.asarray(np.stack(ids),dtype=jnp.int32)
    qmask=jnp.asarray(np.stack(masks),dtype=jnp.int32)
    qhidden=qwen3_jit(qwen3_params,qids,qmask)
    qhidden=(qhidden*qmask[...,None].astype(qhidden.dtype)).astype(jnp.float32)
    out=[]
    for b,p in enumerate([prompt,neg]):
        enc=_t5.encode(p)
        out.append((np.asarray(qhidden[b:b+1]), np.asarray(qmask[b:b+1]),
                    np.asarray([enc.ids],dtype=np.int32), np.asarray([enc.attention_mask],dtype=np.int32)))
    return out
log(f"text/Qwen3 TPU ready in {time.perf_counter()-t0:.1f}s")

log("=== SERVER: conditioner ===")
t0 = time.perf_counter()
cond_cfg = AnimaTextConditionerConfig(dtype=jnp.float32, param_dtype=jnp.float32)
conditioner = FlaxAnimaTextConditioner(cond_cfg)
cv = jax.eval_shape(conditioner.init, jax.random.key(1),
                    jax.ShapeDtypeStruct((1, 8, 1024), jnp.float32),
                    jax.ShapeDtypeStruct((1, 8), jnp.int32))
with jax.default_device(_CPU):
    cond_params = convert_anima_aesthetic_adapter_weights(aesthetic_path, cv["params"], dtype=jnp.float32)
del cv
gc.collect()
log(f"conditioner converted in {time.perf_counter()-t0:.1f}s")

log("=== SERVER: transformer (scan_blocks=True) ===")
t0 = time.perf_counter()
transformer = FlaxAnimaCosmosTransformer(layers=28, scan_blocks=True)
# Shape-only init: the converters read .shape, never values, so eval_shape gives the
# full parameter tree in ~0.2s with negligible RSS (vs ~40s and a multi-GB dummy tree).
tv = jax.eval_shape(
    transformer.init, jax.random.key(2),
    jax.ShapeDtypeStruct((1, 16, 1, 8, 8), jnp.float32),
    jax.ShapeDtypeStruct((1,), jnp.float32),
    jax.ShapeDtypeStruct((1, 8, 1024), jnp.float32),
)
t_shape = time.perf_counter() - t0
t0 = time.perf_counter()
try:
    with jax.default_device(_CPU):
        t_params = convert_anima_aesthetic_weights(aesthetic_path, tv["params"],
                                                   dtype=jnp.bfloat16, num_layers=28)
except Exception:
    log("transformer: conversion raised:\n" + traceback.format_exc())
    raise
t_conv = time.perf_counter() - t0
del tv
gc.collect()
log(f"transformer: eval_shape {t_shape:.2f}s + convert {t_conv:.1f}s")
t0 = time.perf_counter()
t_params = jax.device_put(t_params, jax.devices()[0])
log(f"transformer params -> TPU in {time.perf_counter()-t0:.1f}s; {hbm_stats()}")

log("=== SERVER: VAE ===")
t0 = time.perf_counter()
vae = AutoencoderKLWan(nnx.Rngs(0), dtype=jnp.bfloat16, weights_dtype=jnp.bfloat16)
_st = nnx.state(vae, nnx.Param)
_fs = dict(nnx.to_flat_state(_st))
_ft = {k: v.value for k, v in _fs.items()}
with jax.default_device(_CPU):
    _conv = load_qwen_image_vae(snapshot_dir, _ft)
    _cf = _flatten_dict(_conv)
    _cf_by_path = {"/".join(str(x) for x in k): v for k, v in _cf.items()}
    _nf, _miss = {}, []
    for _k, _vs in _fs.items():
        _p = "/".join(str(x) for x in _k)
        if _p in _cf_by_path:
            _nf[_k] = _vs.replace(jnp.asarray(_cf_by_path[_p], dtype=_vs.value.dtype))
        else:
            _miss.append(_p)
assert not _miss, f"VAE merge incomplete: {_miss[:8]}"
nnx.update(vae, nnx.State.from_flat_path(_nf))
del _ft, _conv, _cf, _cf_by_path, _nf, _st, _fs
gc.collect()
log(f"vae converted+merged in {time.perf_counter()-t0:.1f}s")

@jax.jit
def cond_forward(c_params, source_hidden, source_mask, target_ids, target_mask):
    return conditioner.apply({"params": c_params}, source_hidden_states=source_hidden,
                             target_input_ids=target_ids,
                             source_attention_mask=source_mask,
                             target_attention_mask=target_mask)

# Only the transformer forward is jitted. CFG (two calls + combine) stays eager: a nested
# jit over both calls duplicates a large HLO and was what previously blew up compile memory.
@jax.jit
def transformer_forward(tp, lat, t_vec, ctx, pad):
    return transformer.apply({"params": tp}, lat, t_vec, ctx, None, pad)

def cfg_prediction(latents, timestep, context, neg_context, pad, guidance):
    t_vec = jnp.broadcast_to(timestep / jnp.float32(1000.0), (latents.shape[0],)).astype(jnp.float32)
    nc = transformer_forward(t_params, latents, t_vec, context, pad)
    nu = transformer_forward(t_params, latents, t_vec, neg_context, pad)
    nc32 = nc.astype(jnp.float32)
    nu32 = nu.astype(jnp.float32)
    return nu32 + jnp.float32(guidance) * (nc32 - nu32)

def decode_latents(latents):
    """Full VAE decode of a latent, matching the pipeline's latent mean/std glue."""
    lmean = jnp.array(vae.latents_mean, dtype=latents.dtype).reshape(1, 16, 1, 1, 1)
    lstd = jnp.array(vae.latents_std, dtype=latents.dtype).reshape(1, 16, 1, 1, 1)
    z = latents / (1.0 / lstd) + lmean
    graphdef, state, rest = nnx.split(vae, nnx.Param, ...)
    merged = nnx.merge(graphdef, state, rest)
    video = merged.decode(z, AutoencoderKLWanCache(merged), return_dict=False)[0]
    video = jnp.clip(video / 2.0 + 0.5, 0.0, 1.0)
    video.block_until_ready()
    img = np.asarray(video)
    if img.ndim == 5:
        img = img[:, 0]
    if img.shape[-1] not in (1, 3):
        img = np.moveaxis(img, 1, -1)
    return (img[0] * 255.0).round().astype(np.uint8)

def save_u8(u8, path):
    from PIL import Image
    tmp = path + ".tmp.png"
    Image.fromarray(u8).save(tmp)
    os.replace(tmp, path)

log("=== SERVER: warming executables at production shape (1024px) ===")
try:
    t0 = time.perf_counter()
    _sched = FlaxFlowMatchScheduler()
    _st8 = FlowMatchSchedulerState.create()
    _sr = np.linspace(1.0, 1.0 / 4, 4).astype(np.float32)
    _st8 = _sched.set_timesteps(_st8, sigmas=jnp.asarray(_sr))
    _ts = np.asarray(_st8.timesteps); _sg = np.asarray(_st8.sigmas[:-1])
    # Warm the transformer at the real 1024px latent/pad shapes so the first user
    # generation does not pay the XLA compile.
    _lat = jnp.asarray(np.random.default_rng(0).standard_normal((1, 16, 1, 128, 128)).astype(np.float32))
    _pad = jnp.zeros((1, 1, 1024, 1024), dtype=jnp.float32)
    # These dtypes must match the real generation path exactly or XLA compiles a
    # SECOND executable for the first real request. They did not: the real context
    # is bf16 (cast after the conditioner) and the real latents are fp32. Warmup
    # therefore warmed an fp32-context/fp32-latent signature the server never uses,
    # and the first generation paid both compiles again (~124s vs 21s measured).
    _ctx = jnp.zeros((1, 512, 1024), dtype=jnp.bfloat16)
    for _i in range(2):
        _pred = cfg_prediction(_lat, jnp.asarray(np.float32(_ts[_i])), _ctx, _ctx, _pad, 4.0)
        _sn = _sg[_i + 1] if _i + 1 < 4 else 0.0
        _lat = _lat + jnp.asarray(np.float32(_sn - _sg[_i])) * _pred
    _lat.block_until_ready()
    del _lat, _pad, _ctx, _pred
    gc.collect()
    log(f"warmup: transformer forward compiled+ran at 1024px in {time.perf_counter()-t0:.1f}s; {hbm_stats()}")
    t0 = time.perf_counter()
    # fp32, not bf16: the real latents entering decode_latents are fp32 (they come
    # from an fp32 noise tensor), and decode_latents derives latents_mean/std from
    # the latent dtype, so a bf16 warmup compiles an executable the server never
    # calls -- the first real preview then recompiled the VAE (~50s, matching this
    # warmup's own compile time).
    _wz = jnp.zeros((1, 16, 1, 128, 128), dtype=jnp.float32)
    _wv = decode_latents(_wz)
    log(f"warmup: VAE decode compiled+ran at 1024px in {time.perf_counter()-t0:.1f}s; out {_wv.shape}; {hbm_stats()}")
    del _wz, _wv
    gc.collect()
    log(f"warmup done; executables compiled; host RSS {rss_gb():.2f} GB")
except Exception:
    log("warmup raised:\n" + traceback.format_exc())
    raise

log(f"SERVER READY total {(time.perf_counter()-t_all)/60:.1f} min; watching {REQ_PATH}")
write_prog(stage="ready", step=0, steps=0)

def default_req():
    return {"prompt": "masterpiece, best quality, 1girl",
            "negative_prompt": "worst quality, low quality, blurry",
            "height": 1024, "width": 1024, "steps": 30, "guidance": 4.0,
            # -1 = pick a random seed; the server resolves it and reports the value.
            "seed": -1, "preview_every": 5, "out": "/content/anima_perstep.png"}

if not os.path.exists(REQ_PATH):
    json.dump({**default_req(), "go": False}, open(REQ_PATH, "w"))

last_mtime = 0.0
while True:
    try:
        mt = os.path.getmtime(REQ_PATH)
    except FileNotFoundError:
        time.sleep(1.0)
        continue
    if mt == last_mtime:
        time.sleep(0.5)
        continue
    last_mtime = mt
    try:
        req = json.load(open(REQ_PATH))
    except Exception as e:
        log(f"bad request json: {e}")
        continue
    if not req.get("go"):
        continue
    d = default_req(); d.update({k: v for k, v in req.items() if k != "go"})
    json.dump({**d, "go": False, "status": "running"}, open(REQ_PATH, "w"))
    write_prog(stage="starting", step=0, steps=d.get("steps", 30))
    last_mtime = os.path.getmtime(REQ_PATH)
    try:
        PROMPT, NEG = d["prompt"], d["negative_prompt"]
        H, W, STEPS, GUIDANCE, SEED = d["height"], d["width"], d["steps"], d["guidance"], d["seed"]
        OUT = d["out"]
        PREV_EVERY = int(d.get("preview_every", 5) or 0)
        # H and W must be multiples of 16. The VAE downsamples by 8, and the transformer
        # patchifies the latent grid by 2, so the latent dimensions must stay even
        # (cosmos_patchify raises "Input dimensions must be divisible by patch_size"
        # otherwise -- a 1000px request gives a 125-wide latent grid). A non-multiple also
        # silently loses the remainder pixels (1108 -> 1104) and used to crash the mask
        # resize. Round to the nearest multiple of 16 and say so.
        _MULT = 16
        H0, W0 = int(H), int(W)
        H = max(_MULT, int(round(H0 / _MULT)) * _MULT)
        W = max(_MULT, int(round(W0 / _MULT)) * _MULT)
        if (H, W) != (H0, W0):
            log(f"requested {H0}x{W0} is not a multiple of {_MULT}; using {H}x{W} "
                f"(VAE stride 8, patchify stride 2)")
        # A negative seed means "pick one", so the UI can offer random by default.
        # The resolved value goes into the log and the progress file; without that a
        # random run cannot be reproduced.
        _seed_req = int(SEED)
        if _seed_req < 0:
            SEED = int(np.random.default_rng().integers(0, 2**31 - 1))
            log(f"seed {_seed_req} requested; using random seed {SEED}")
        g0 = time.perf_counter()
        write_prog(stage="text-encoding", step=0, steps=STEPS, seed=int(SEED))
        (qe, qm, t5ids, t5mask), (ne, nm, nt5ids, nt5mask) = encode_texts(PROMPT, NEG)
        write_prog(stage="conditioning", step=0, steps=STEPS, seed=int(SEED))
        context = cond_forward(cond_params, jnp.asarray(qe, dtype=jnp.float32), qm, t5ids, t5mask).astype(jnp.bfloat16)
        neg_context = cond_forward(cond_params, jnp.asarray(ne, dtype=jnp.float32), nm, nt5ids, nt5mask).astype(jnp.bfloat16)
        context.block_until_ready(); neg_context.block_until_ready()
        sched = FlaxFlowMatchScheduler()
        st = FlowMatchSchedulerState.create()
        sigmas_raw = np.linspace(1.0, 1.0 / STEPS, STEPS).astype(np.float32)
        st = sched.set_timesteps(st, sigmas=jnp.asarray(sigmas_raw))
        timesteps = np.asarray(st.timesteps)
        sigmas = np.asarray(st.sigmas[:-1])
        rng = np.random.default_rng(SEED)
        latents = jnp.asarray(rng.standard_normal((1, 16, 1, H // 8, W // 8)).astype(np.float32))
        pad = jnp.zeros((1, 1, H, W), dtype=jnp.float32)
        d0 = time.perf_counter()
        prev_total = 0.0
        for i in range(STEPS):
            pred = cfg_prediction(latents, jnp.asarray(np.float32(timesteps[i])),
                                  context, neg_context, pad, GUIDANCE)
            sigma_next = sigmas[i + 1] if i + 1 < STEPS else 0.0
            latents = latents + (jnp.asarray(np.float32(sigma_next - sigmas[i]))) * pred
            latents.block_until_ready()
            step_s = time.perf_counter() - d0
            if i % 2 == 0 or i == STEPS - 1:
                write_prog(stage="denoise", step=i + 1, steps=STEPS,
                           ms_per_step=round(step_s / (i + 1) * 1000.0, 1),
                           guidance=GUIDANCE, seed=int(SEED))
            if PREV_EVERY and ((i + 1) % PREV_EVERY == 0 or i == STEPS - 1):
                try:
                    p0 = time.perf_counter()
                    save_u8(decode_latents(latents), SNAP_PATH)
                    prev_total += time.perf_counter() - p0
                    write_prog(stage="denoise", step=i + 1, steps=STEPS,
                               preview_step=i + 1, preview_s=round(prev_total, 2),
                               ms_per_step=round(step_s / (i + 1) * 1000.0, 1),
                               guidance=GUIDANCE, seed=int(SEED))
                except Exception as _e:
                    log(f"preview decode failed at step {i+1}: {_e}")
        latents.block_until_ready()
        loop_s = time.perf_counter() - d0
        pre_s = d0 - g0
        # Preview decodes run inside the loop, so subtract them: otherwise "denoise"
        # silently includes ~50s of VAE work on the first generation (it did).
        denoise_s = loop_s - prev_total
        write_prog(stage="decode", step=STEPS, steps=STEPS)
        v0 = time.perf_counter()
        save_u8(decode_latents(latents), OUT)
        decode_s = time.perf_counter() - v0
        total_s = time.perf_counter() - g0
        log(f"GEN DONE {OUT} {H}x{W} seed {SEED} in {total_s:.1f}s (text+cond {pre_s:.1f}s, "
            f"denoise {denoise_s:.1f}s = {denoise_s / max(1, STEPS) * 1000.0:.0f} ms/step, "
            f"previews {prev_total:.1f}s, decode {decode_s:.1f}s) "
            f"= {60.0/total_s:.2f} images/min; {hbm_stats()}; RSS {rss_gb():.2f} GB")
        write_prog(stage="done", step=STEPS, steps=STEPS, out=OUT, seed=int(SEED),
                   elapsed_s=round(total_s, 1), pre_s=round(pre_s, 1),
                   denoise_s=round(denoise_s, 1), decode_s=round(decode_s, 1),
                   height=H, width=W, hbm=hbm_stats())
        json.dump({**d, "go": False, "status": "done", "elapsed_s": round(total_s, 1)},
                  open(REQ_PATH, "w"))
    except Exception:
        log("gen raised:\n" + traceback.format_exc())
        write_prog(stage="error")
        try:
            json.dump({**d, "go": False, "status": "error"}, open(REQ_PATH, "w"))
        except Exception:
            pass
    last_mtime = os.path.getmtime(REQ_PATH)
'''
r = subprocess.Popen([py, "-u", "-c", code], cwd=repo, env=env)
print("server pid", r.pid, flush=True)
