"""One-shot Anima E2E run on TPU: text -> conditioner -> transformer -> denoise -> VAE -> PNG.

Runs the full pipeline twice (cold + warm) so warm denoise throughput can be
reported separately from XLA compile time, and prints host RSS plus TPU HBM at
every stage. Use this to validate a change end to end; use
`scripts/anima_aesthetic_server.py` for interactive Gradio serving.
"""
import os
import subprocess
import sys

repo = "/content/maxdiffusion"
venv = repo + "/.venv"
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
print("interpreter:", py, flush=True)

code = r'''
import time, os, gc, shutil, traceback
import numpy as np
import jax
jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32")
import jax.numpy as jnp

LOG = open("/content/anima_scan_e2e.log", "a", buffering=1)
def rss():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg} | RSS={rss():.2f}GB"
    print(line, flush=True)
    LOG.write(line + "\n")

log("=== ANIMA SCAN E2E (scan_blocks=True, BF16_BF16_F32) ===")
log(f"devices: {jax.devices()}")
try:
    log(f"HBM before: {jax.devices()[0].memory_stats()}")
except Exception as e:
    log(f"HBM probe unavailable: {e}")

from maxdiffusion.models.anima_cosmos_flax import FlaxAnimaCosmosTransformer, convert_anima_aesthetic_weights
from maxdiffusion.models.anima_text_conditioner_flax import (
    AnimaTextConditionerConfig, FlaxAnimaTextConditioner, convert_anima_aesthetic_adapter_weights)
from maxdiffusion.models.qwen_image_vae_utils import load_qwen_image_vae
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
from maxdiffusion.schedulers.scheduling_flow_match_flax import (
    FlaxFlowMatchScheduler, FlowMatchSchedulerState)
from flax import nnx
from flax.traverse_util import flatten_dict as _flatten_dict
from huggingface_hub import snapshot_download

PROMPT = ("masterpiece, best quality, 1girl, fern (sousou no frieren), sousou no frieren, "
          "purple hair, black robe, lips, sidelocks, very long hair, puffy sleeves, white dress, "
          "butterfly on hand, eyelashes, simple background, closed mouth, mage staff, arm at side, "
          "straight hair, blush, solo, purple eyes, purple pupils, looking at viewer, hand up, "
          "standing, robe, black background, bright pupils, black coat, coat, long sleeves, "
          "blue butterfly, upturned eyes, wide sleeves, blunt bangs, from above, dress, long hair, "
          "purple butterfly, butterfly, half updo")
NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"
H = W = 1024
STEPS = 30
GUIDANCE = 4.0
SEED = 0

t_all = time.perf_counter()
snapshot_dir = snapshot_download("circlestone-labs/Anima-Base-v1.0-Diffusers", cache_dir="/content/hf_cache")
aesthetic_snapshot = snapshot_download("circlestone-labs/Anima", cache_dir="/content/hf_cache",
                                       allow_patterns=["split_files/diffusion_models/anima-aesthetic-v1.1.safetensors"])
CKPT = "/content/aesthetic_v1.1.safetensors"
_src = os.path.join(aesthetic_snapshot, "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors")
if not os.path.exists(CKPT) or os.path.getsize(CKPT) != os.path.getsize(_src):
    shutil.copyfile(_src, CKPT)
log(f"weights: base={snapshot_dir}")
log(f"aesthetic ckpt staged {os.path.getsize(CKPT)/1e9:.2f} GB")

log("STAGE 1: text encoding (torch CPU)")
t0 = time.perf_counter()
import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("circlestone-labs/Anima-Base-v1.0-Diffusers", subfolder="tokenizer")
from tokenizers import Tokenizer as _RawTokenizer
_t5 = _RawTokenizer.from_file(os.path.join(snapshot_dir, "t5_tokenizer", "tokenizer.json"))
_t5.enable_padding(pad_id=0, pad_token="<pad>", length=512)
_t5.enable_truncation(max_length=512)
te = AutoModel.from_pretrained("circlestone-labs/Anima-Base-v1.0-Diffusers", subfolder="text_encoder", torch_dtype=torch.float32).eval()
texts = []
with torch.no_grad():
    for p in [PROMPT, NEG]:
        ti = tok(p, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
        emb = te(input_ids=ti.input_ids, attention_mask=ti.attention_mask).last_hidden_state
        emb = emb * ti.attention_mask.to(emb.dtype).unsqueeze(-1)
        _enc = _t5.encode(p)
        texts.append((emb.cpu().numpy(), ti.attention_mask.cpu().numpy(),
                      np.asarray([_enc.ids], dtype=np.int32), np.asarray([_enc.attention_mask], dtype=np.int32)))
del te; gc.collect()
(qe, qm, t5ids, t5mask), (ne, nm, nt5ids, nt5mask) = texts
log(f"text encoding done in {time.perf_counter()-t0:.1f}s; qe {qe.shape}")

log("STAGE 2: conditioner")
t0 = time.perf_counter()
cond_cfg = AnimaTextConditionerConfig(dtype=jnp.float32, param_dtype=jnp.float32)
conditioner = FlaxAnimaTextConditioner(cond_cfg)
cv = jax.eval_shape(conditioner.init, jax.random.key(1),
                    jax.ShapeDtypeStruct((1, 8, 1024), jnp.float32),
                    jax.ShapeDtypeStruct((1, 8), jnp.int32))
cond_params = convert_anima_aesthetic_adapter_weights(CKPT, cv["params"], dtype=jnp.float32)
del cv; gc.collect()
log(f"conditioner converted in {time.perf_counter()-t0:.1f}s")

@jax.jit
def cond_forward(c_params, source_hidden, source_mask, target_ids, target_mask):
    return conditioner.apply({"params": c_params}, source_hidden, target_ids, source_mask, target_mask)

context = cond_forward(cond_params, jnp.asarray(qe, jnp.float32), jnp.asarray(qm, jnp.float32),
                       jnp.asarray(t5ids, jnp.int32), jnp.asarray(t5mask, jnp.int32))
neg_context = cond_forward(cond_params, jnp.asarray(ne, jnp.float32), jnp.asarray(nm, jnp.float32),
                           jnp.asarray(nt5ids, jnp.int32), jnp.asarray(nt5mask, jnp.int32))
context.block_until_ready(); neg_context.block_until_ready()
log(f"context {context.shape} {context.dtype}")
del qe, ne; gc.collect()

log("STAGE 3: transformer (scan_blocks=True)")
t0 = time.perf_counter()
transformer = FlaxAnimaCosmosTransformer(layers=28, scan_blocks=True)
shape_tree = jax.eval_shape(
    transformer.init, jax.random.key(2),
    jax.ShapeDtypeStruct((1, 16, 1, 8, 8), jnp.float32),
    jax.ShapeDtypeStruct((1,), jnp.float32),
    jax.ShapeDtypeStruct((1, 8, 1024), jnp.float32),
)
log(f"eval_shape tree in {time.perf_counter()-t0:.1f}s")
t0 = time.perf_counter()
t_params = convert_anima_aesthetic_weights(CKPT, shape_tree["params"], dtype=jnp.bfloat16, num_layers=28)
del shape_tree; gc.collect()
log(f"transformer converted in {time.perf_counter()-t0:.1f}s")
t_params = jax.device_put(t_params, jax.devices()[0])
log("params on TPU")

st = FlowMatchSchedulerState.create()
scheduler = FlaxFlowMatchScheduler(num_train_timesteps=1000, shift=3.0, dtype=jnp.float32)
st = scheduler.set_timesteps(st, STEPS, shift=3.0, sigmas=np.linspace(1.0, 1.0/STEPS, STEPS).tolist())
sigmas = np.asarray(st.sigmas)

@jax.jit
def transformer_forward(tp, lat, t, ctx):
    return transformer.apply({"params": tp}, lat, t, ctx, attention_mask=None, padding_mask=None)

def run_steps(seed):
    rng = np.random.default_rng(seed)
    latents = jnp.asarray(rng.standard_normal((1, 16, 1, H // 8, W // 8)).astype(np.float32))
    t0 = time.perf_counter()
    t_first = None
    for i in range(STEPS):
        sig = jnp.asarray([np.float32(sigmas[i])])
        sig_next = jnp.asarray([np.float32(sigmas[i + 1])])
        pc = transformer_forward(t_params, latents, sig, context)
        pu = transformer_forward(t_params, latents, sig, neg_context)
        pred = pu + GUIDANCE * (pc - pu)
        latents = latents + (sig_next - sig) * pred
        if i == 0:
            latents.block_until_ready()
            t_first = time.perf_counter() - t0
            log(f"  step 1/30 (compile included) {t_first:.1f}s")
        elif i in (9, 19, 29):
            latents.block_until_ready()
            a = np.asarray(latents)
            log(f"  step {i+1}/30 std={float(a.std()):.6f} elapsed={time.perf_counter()-t0:.1f}s")
    latents.block_until_ready()
    return latents, time.perf_counter() - t0, t_first

latents, t_total, t_first = run_steps(SEED)
log(f"RUN1 (cold): {t_total:.1f}s total, first step {t_first:.1f}s, {STEPS-1} warm steps {t_total-t_first:.1f}s ({(t_total-t_first)/(STEPS-1):.2f}s/step)")
np.save("/content/scan_latents_run1.npy", np.asarray(latents))

lat2, t2, tf2 = run_steps(SEED + 1)
log(f"RUN2 (warm): {t2:.1f}s total for {STEPS} steps, first step {tf2:.2f}s -> {t2/STEPS:.2f}s/step = {t2:.1f}s/image warm = {60.0/t2:.2f} images/min")

try:
    log(f"HBM after denoise: {jax.devices()[0].memory_stats()}")
except Exception as e:
    log(f"HBM probe unavailable: {e}")

log("STAGE 4: VAE decode")
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
del _ft, _conv, _cf, _cf_by_path, _nf, _st, _fs
gc.collect()
log(f"vae converted+merged in {time.perf_counter()-t0:.1f}s")

def decode_and_save(lat, out_path):
    lmean = jnp.array(vae.latents_mean, dtype=lat.dtype).reshape(1, 16, 1, 1, 1)
    lstd = jnp.array(vae.latents_std, dtype=lat.dtype).reshape(1, 16, 1, 1, 1)
    z = lat / (1.0 / lstd) + lmean
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
    Image.fromarray(img_u8).save(out_path)
    return img_u8.shape

t0 = time.perf_counter()
shape = decode_and_save(latents, "/content/anima_scan_run1.png")
log(f"VAE decode RUN1 {time.perf_counter()-t0:.1f}s -> /content/anima_scan_run1.png {shape}")
t0 = time.perf_counter()
shape2 = decode_and_save(lat2, "/content/anima_scan_run2.png")
log(f"VAE decode RUN2 (warm) {time.perf_counter()-t0:.1f}s -> /content/anima_scan_run2.png {shape2}")

log(f"=== E2E DONE total {time.perf_counter()-t_all:.1f}s ===")
'''

with open("/tmp/scan_e2e.py", "w") as f:
    f.write(code)

r = subprocess.run([py, "/tmp/scan_e2e.py"], env=env, capture_output=True, text=True, timeout=3600)
print("STDOUT:", r.stdout[-14000:], flush=True)
print("STDERR:", r.stderr[-3000:], flush=True)
print("RC:", r.returncode)
