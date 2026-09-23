import os
import subprocess
import sys

repo = "/content/maxdiffusion"
venv = repo + "/.venv"
py = venv + "/bin/python"
env = {**os.environ, "PATH": venv + "/bin:/usr/local/bin:" + os.environ["PATH"],
       "PYTHONPATH": repo + "/src", "HF_HOME": "/content/hf_cache",
       "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}
env.pop("UV_SYSTEM_PYTHON", None)

code = r'''
import time

LOG = open("/content/anima_run.log", "a", buffering=1)
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")

log("=== STAGE 0: imports ===")
import os
import numpy as np
import gc
import traceback
import faulthandler
faulthandler.dump_traceback_later(600, exit=True)
import jax
jax.config.update("jax_default_matmul_precision", "bfloat16")
import jax.numpy as jnp
log("matmul precision: bfloat16 (single-change bf16 test; activations stay fp32)")
log(f"devices: {jax.devices()}")

from maxdiffusion.models.anima_cosmos_flax import FlaxAnimaCosmosTransformer, convert_anima_aesthetic_weights
from maxdiffusion.models.anima_text_conditioner_flax import (
    AnimaTextConditionerConfig, FlaxAnimaTextConditioner,
    load_and_convert_anima_text_conditioner_weights, convert_anima_aesthetic_adapter_weights)
from maxdiffusion.models.qwen_image_vae_utils import load_qwen_image_vae
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
from maxdiffusion.schedulers.scheduling_flow_match_flax import (
    FlaxFlowMatchScheduler, FlowMatchSchedulerState)
from flax import nnx
from flax import serialization as flax_serialization
from flax.traverse_util import flatten_dict as _flatten_dict
from huggingface_hub import snapshot_download

PROMPT = "masterpiece, best quality, 1girl, fern (sousou no frieren), sousou no frieren, @izei1337, purple hair, black robe, lips, sidelocks, feet out of frame, very long hair, puffy sleeves, white dress, butterfly on hand, eyelashes, simple background, closed mouth, mage staff, arm at side, straight hair, blush, solo, purple eyes, chromatic aberration, purple pupils, looking at viewer, hand up, standing, bug, robe, black background, signature, bright pupils, black coat, coat, long sleeves, blue butterfly, upturned eyes, wide sleeves, blunt bangs, from above, dress, blunt ends, long hair, purple butterfly, butterfly, tsurime, half updo"
NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"
H = W = 1024
STEPS = 30
GUIDANCE = 4.0
SEED = 0

t_all = time.perf_counter()
snapshot_dir = snapshot_download("circlestone-labs/Anima-Base-v1.0-Diffusers")
aesthetic_snapshot = snapshot_download("circlestone-labs/Anima", allow_patterns=["split_files/diffusion_models/anima-aesthetic-v1.1.safetensors"])
import shutil
shutil.copyfile(os.path.join(aesthetic_snapshot, "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors"), "/content/aesthetic_v1.1.safetensors")
log(f"weights at {snapshot_dir}; aesthetic staged")

log("=== STAGE 1: text encoding (torch CPU) ===")
t0 = time.perf_counter()
import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(snapshot_dir, "tokenizer"))
from tokenizers import Tokenizer as _RawTokenizer
_t5 = _RawTokenizer.from_file(os.path.join(snapshot_dir, "t5_tokenizer", "tokenizer.json"))
_t5.enable_padding(pad_id=0, pad_token="<pad>", length=512)
_t5.enable_truncation(max_length=512)
te = AutoModel.from_pretrained(os.path.join(snapshot_dir, "text_encoder"), torch_dtype=torch.float32).eval()
texts = []
with torch.no_grad():
    for p in [PROMPT, NEG]:
        ti = tok(p, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        emb = te(input_ids=ti.input_ids, attention_mask=ti.attention_mask).last_hidden_state
        emb = emb * ti.attention_mask.to(emb.dtype).unsqueeze(-1)
        _enc = _t5.encode(p)
        t5ids = np.asarray([_enc.ids], dtype=np.int32)
        t5mask = np.asarray([_enc.attention_mask], dtype=np.int32)
        texts.append((emb.cpu().numpy(), ti.attention_mask.cpu().numpy(), t5ids, t5mask))
del te
gc.collect()
(qe, qm, t5ids, t5mask), (ne, nm, nt5ids, nt5mask) = texts
log(f"text encoding done in {time.perf_counter()-t0:.1f}s; qe {qe.shape} {qe.dtype}")
np.save('/content/dump_qe.npy', qe); np.save('/content/dump_ne.npy', ne); np.save('/content/dump_qm.npy', qm); np.save('/content/dump_nm.npy', nm); np.save('/content/dump_t5.npy', t5ids); np.save('/content/dump_nt5.npy', nt5ids)

log("=== STAGE 2: weight conversion ===")
CACHE_DIR = "/content/anima_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
import hashlib
def _cache_valid(path, src, src_mtime):
    if not os.path.exists(path):
        return False
    try:
        import json
        meta = json.load(open(path + ".meta"))
        return meta.get("src_mtime") == src_mtime and meta.get("src_size") == os.path.getsize(src)
    except Exception:
        return False
def _cache_mark(path, src, src_mtime):
    import json
    json.dump({"src_mtime": src_mtime, "src_size": os.path.getsize(src)},
              open(path + ".meta", "w"))

t0 = time.perf_counter()
cond_cfg = AnimaTextConditionerConfig(dtype=jnp.float32, param_dtype=jnp.float32)
conditioner = FlaxAnimaTextConditioner(cond_cfg)
cv = conditioner.init(jax.random.key(1), jnp.zeros((1, 8, 1024), jnp.float32), np.zeros((1, 8), np.int32))
aesthetic_path = "/content/aesthetic_v1.1.safetensors"
aes_mtime = os.path.getmtime(aesthetic_path)
cond_cache = os.path.join(CACHE_DIR, "cond_params.msgpack")
if _cache_valid(cond_cache, aesthetic_path, aes_mtime):
    log("conditioner: loading from disk cache ...")
    with open(cond_cache, "rb") as f:
        cond_params = flax_serialization.from_bytes(cv["params"], f.read())
    log(f"conditioner loaded from cache in {time.perf_counter()-t0:.1f}s")
else:
    cond_params = convert_anima_aesthetic_adapter_weights(aesthetic_path, cv["params"], dtype=jnp.float32)
    with open(cond_cache, "wb") as f:
        f.write(flax_serialization.to_bytes(cond_params))
    _cache_mark(cond_cache, aesthetic_path, aes_mtime)
    log(f"conditioner converted in {time.perf_counter()-t0:.1f}s")
del cv
gc.collect()

t0 = time.perf_counter()
transformer = FlaxAnimaCosmosTransformer(layers=28)
log("transformer: init on dummy (1,16,1,8,8) shapes...")
tv = transformer.init(jax.random.key(2), jnp.zeros((1, 16, 1, 8, 8), jnp.bfloat16),
                      jnp.zeros((1,), jnp.bfloat16), jnp.zeros((1, 8, 1024), jnp.bfloat16))
log(f"transformer: init done in {time.perf_counter()-t0:.1f}s")
aesthetic_path = "/content/aesthetic_v1.1.safetensors"
log(f"transformer: converting 685 aesthetic keys from {aesthetic_path} ...")
log(f"transformer: file size {os.path.getsize(aesthetic_path)/1e9:.2f} GB, tv leaves {len(jax.tree_util.tree_leaves(tv['params']))}")
t_cache = os.path.join(CACHE_DIR, "transformer_params.msgpack")
if _cache_valid(t_cache, aesthetic_path, aes_mtime):
    log("transformer: loading from disk cache ...")
    with open(t_cache, "rb") as f:
        t_params = flax_serialization.from_bytes(tv["params"], f.read())
    log("transformer: loaded from cache")
else:
    try:
        t_params = convert_anima_aesthetic_weights(aesthetic_path, tv["params"], dtype=jnp.bfloat16)
    except Exception:
        log("transformer: conversion raised:\n" + traceback.format_exc())
        raise
    log("transformer: conversion done")
    with open(t_cache, "wb") as f:
        f.write(flax_serialization.to_bytes(t_params))
    _cache_mark(t_cache, aesthetic_path, aes_mtime)
del tv
gc.collect()
log(f"transformer converted in {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
vae = AutoencoderKLWan(nnx.Rngs(0), dtype=jnp.bfloat16, weights_dtype=jnp.bfloat16)
_st = nnx.state(vae, nnx.Param)
_fs = dict(nnx.to_flat_state(_st))
_ft = {k: v.value for k, v in _fs.items()}
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
# No msgpack cache: to_bytes+write on the transformer tree measured 76s vs ~10s to
# re-convert, and serializing an nnx.State here crashed outright
# ("TypeError: can not serialize 'State' object") whenever the cache was invalid.
del _ft, _conv, _cf, _cf_by_path, _nf, _st, _fs
gc.collect()
log(f"vae converted+merged in {time.perf_counter()-t0:.1f}s")

log("=== STAGE 3: encode conditioner (jax) ===")
try:
    @jax.jit
    def cond_forward(c_params, source_hidden, source_mask, target_ids, target_mask):
        return conditioner.apply({"params": c_params}, source_hidden_states=source_hidden,
                                 target_input_ids=target_ids,
                                 source_attention_mask=source_mask,
                                 target_attention_mask=target_mask)
    t0 = time.perf_counter()
    context = cond_forward(cond_params, jnp.asarray(qe, dtype=jnp.float32), qm, t5ids, t5mask).astype(jnp.bfloat16)
    neg_context = cond_forward(cond_params, jnp.asarray(ne, dtype=jnp.float32), nm, nt5ids, nt5mask).astype(jnp.bfloat16)
    context.block_until_ready(); neg_context.block_until_ready()
except Exception:
    log("conditioner encode raised:\n" + traceback.format_exc())
    raise
np.save('/content/dump_context.npy', np.asarray(context)); np.save('/content/dump_neg_context.npy', np.asarray(neg_context))
log(f"conditioning encoded in {time.perf_counter()-t0:.1f}s; ctx {context.shape}")
# free conditioner device params (encode done; keep host copy? we only need conv later inside jit scope? no)
del cond_params, conditioner
gc.collect()

log("=== STAGE 4: per-step denoise (single-step jit) ===")
sched = FlaxFlowMatchScheduler()
st = FlowMatchSchedulerState.create()
sigmas_raw = np.linspace(1.0, 1.0 / STEPS, STEPS).astype(np.float32)
st = sched.set_timesteps(st, sigmas=jnp.asarray(sigmas_raw))
timesteps = np.asarray(st.timesteps)
sigmas = np.asarray(st.sigmas[:-1])
np.save('/content/dump_sigmas.npy', sigmas); np.save('/content/dump_timesteps.npy', timesteps)
log(f"sigmas head {sigmas[:3]} tail {sigmas[-3:]}")

@jax.jit
def tf_step(t_params, latents, timestep, ctx, nctx, pad):
    t_vec = jnp.broadcast_to(timestep / jnp.float32(1000.0), (latents.shape[0],)).astype(jnp.float32)
    nc = transformer.apply({"params": t_params}, latents, t_vec, ctx, None, pad)
    nu = transformer.apply({"params": t_params}, latents, t_vec, nctx, None, pad)
    # Preserve the guidance delta before bf16 rounding amplifies small differences.
    pred = nu.astype(jnp.float32) + jnp.float32(GUIDANCE) * (nc.astype(jnp.float32) - nu.astype(jnp.float32))
    return nc, nu, pred.astype(jnp.float32)

rng = np.random.default_rng(SEED)
latents = jnp.asarray(rng.standard_normal((1, 16, 1, H // 8, W // 8)).astype(np.float32))
pad = jnp.asarray(np.zeros((1, 1, H, W), dtype=np.float32))
np.save('/content/dump_latent_initial.npy', np.asarray(latents))
t0 = time.perf_counter()
try:
    for i in range(STEPS):
        step_input = latents
        nc, nu, pred = tf_step(t_params, step_input, jnp.asarray(np.float32(timesteps[i])),
                               context, neg_context, pad)
        if i in (0, 9, 19, 29):
            nc.block_until_ready(); nu.block_until_ready(); pred.block_until_ready(); step_input.block_until_ready()
            np.save(f'/content/dump_latent_in_{i:02d}.npy', np.asarray(step_input))
            np.save(f'/content/dump_nc_{i:02d}.npy', np.asarray(nc))
            np.save(f'/content/dump_nu_{i:02d}.npy', np.asarray(nu))
            np.save(f'/content/dump_pred_{i:02d}.npy', np.asarray(pred))
            delta = np.asarray(nc, dtype=np.float32) - np.asarray(nu, dtype=np.float32)
            pp = np.asarray(pred, dtype=np.float32)
            log(f'step {i+1} stats nc_rms={np.sqrt(np.mean(np.asarray(nc)**2)):.6g} nu_rms={np.sqrt(np.mean(np.asarray(nu)**2)):.6g} delta_rms={np.sqrt(np.mean(delta**2)):.6g} pred_rms={np.sqrt(np.mean(pp**2)):.6g}')
        sigma_next = sigmas[i + 1] if i + 1 < STEPS else 0.0
        latents = latents + (jnp.asarray(np.float32(sigma_next - sigmas[i]))) * pred
        if i % 5 == 0 or i == STEPS - 1:
            latents.block_until_ready()
            log(f"step {i+1}/{STEPS} elapsed {time.perf_counter()-t0:.1f}s")
except Exception:
    log("denoise raised:\n" + traceback.format_exc())
    raise
latents.block_until_ready()
np.save('/content/dump_latent_final.npy', np.asarray(latents))
denoise_total = time.perf_counter() - t0
log(f"DENOISE DONE in {denoise_total:.1f}s = {denoise_total/STEPS*1000:.0f} ms/step")

log("=== STAGE 5: VAE decode ===")
t0 = time.perf_counter()
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
img_u8 = (img[0] * 255.0).round().astype(np.uint8)
from PIL import Image
Image.fromarray(img_u8).save("/content/anima_perstep.png")
log(f"IMAGE saved /content/anima_perstep.png in {time.perf_counter()-t0:.1f}s; img {img_u8.shape}")

log("=== STAGE 6: timed warm reps ===")
times = []
for rep in range(3):
    rng = np.random.default_rng(SEED + rep + 1)
    latents = jnp.asarray(rng.standard_normal((1, 16, 1, H // 8, W // 8)).astype(np.float32))
    t0 = time.perf_counter()
    for i in range(STEPS):
        _nc, _nu, pred = tf_step(t_params, latents, jnp.asarray(np.float32(timesteps[i])),
                                  context, neg_context, pad)
        sigma_next = sigmas[i + 1] if i + 1 < STEPS else 0.0
        latents = latents + (jnp.asarray(np.float32(sigma_next - sigmas[i]))) * pred
    latents.block_until_ready()
    dt = time.perf_counter() - t0
    times.append(dt)
    log(f"rep {rep+1}/3: {dt:.1f}s ({dt/STEPS*1000:.0f} ms/step)")
warm = min(times)
log(f"WARM {STEPS}-step CFG: {warm:.1f}s = {60.0/warm:.2f} images/min")
log(f"ALL_DONE total {(time.perf_counter()-t_all)/60:.1f} min")
'''
r = subprocess.run([py, "-u", "-c", code], cwd=repo, env=env,
                   capture_output=False, text=True, timeout=1750)
print("returncode", r.returncode, flush=True)