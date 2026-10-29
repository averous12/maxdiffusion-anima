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
from maxdiffusion.models.anima_lora_flax import load_and_merge_anima_loras
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

# --- prompt control begin -----------------------------------------------------
# Weighting AND scheduling come from maxdiffusion.anima_prompt_control, which
# implements the ComfyUI core weighting syntax and the asagi4/comfyui-prompt-control
# scheduling grammar. The old inline parser only understood "(text:weight)" and
# could not schedule, so it is gone.
import maxdiffusion.anima_prompt_control as prompt_control

PROMPT_STYLE = (os.environ.get("ANIMA_PROMPT_STYLE", "comfy").strip().lower()
                or "comfy")
# T5 is our text encoder, and prompt-control documents BREAK as a no-op for T5;
# CAT is the operator that actually splits/concatenates conditioning.

def token_weights(offsets, cw, text):
    """Per-token weight = mean char weight over the token's span (leading spaces ignored)."""
    tw = np.ones(len(offsets), np.float32)
    n = len(cw)
    for k, (a, b) in enumerate(offsets):
        a, b = int(a), min(int(b), n)
        while a < b and text[a].isspace():
            a += 1
        if b > a:
            tw[k] = float(cw[a:b].mean())
    return tw
# --- prompt control end ---

def _encode_text_list(pairs):
    """[(clean_text, per_char_weights)] -> (qhidden, qmask, t5ids, t5mask), batched.

    The weight application is the same as before: the conditioner RMS-normalises
    K, so plain per-token scaling would barely change attention. Instead move each
    weighted token away from (w>1) or toward (w<1) the prompt's mean embedding:
    (z - mu) * w + mu. Tokens with weight exactly 1.0 are untouched, so an
    unweighted prompt is bit-identical to the unweighted path.
    """
    ids, masks, wts, clean = [], [], [], []
    kw = dict(padding="max_length", max_length=512, truncation=True, return_tensors="np")
    for text, cw in pairs:
        clean.append(text)
        tw = np.ones(512, np.float32)
        if len(cw) and (cw != 1.0).any():
            try:
                ti = tok(text, return_offsets_mapping=True, **kw)
                tw = token_weights(ti["offset_mapping"][0], cw, text)
            except Exception as e:  # slow tokenizer has no offsets: weights ignored, not fatal
                log(f"prompt weights ignored ({type(e).__name__}: {e})")
                ti = tok(text, **kw)
        else:
            ti = tok(text, **kw)
        ids.append(ti.input_ids[0]); masks.append(ti.attention_mask[0]); wts.append(tw)

    qids = jnp.asarray(np.stack(ids), dtype=jnp.int32)
    qmask = jnp.asarray(np.stack(masks), dtype=jnp.int32)
    qhidden = qwen3_jit(qwen3_params, qids, qmask)
    qhidden = (qhidden * qmask[..., None].astype(qhidden.dtype)).astype(jnp.float32)
    tw = np.stack(wts)
    if (tw != 1.0).any():
        w = jnp.asarray(tw)[..., None]
        valid = qmask[..., None].astype(jnp.float32)
        mu = qhidden.sum(axis=1, keepdims=True) / jnp.maximum(valid.sum(axis=1, keepdims=True), 1.0)
        qhidden = jnp.where(w != 1.0, (qhidden - mu) * w + mu, qhidden) * valid

    out = []
    for b in range(len(clean)):
        enc = _t5.encode(clean[b])
        out.append((np.asarray(qhidden[b:b + 1]), np.asarray(qmask[b:b + 1]),
                    np.asarray([enc.ids], dtype=np.int32),
                    np.asarray([enc.attention_mask], dtype=np.int32)))
    return out


def encode_texts(prompt, neg, steps=30, style=None, tags=None, seed=0):
    """Schedule-aware encoding.

    Returns (positive, negative) where each is (contexts, step_state, planned):
      contexts   : list of (1, 512, 1024) bf16 conditioning tensors, one per
                   distinct prompt state the schedule passes through
      step_state : index into `contexts` for each denoise step
      planned    : the planner record, for logging

    Combination semantics:
      * AVG(w) → every term is encoded separately and the contexts are averaged
        as w*acc + (1-w)*next, which is what ConditioningAverage does.
      * AND / CAT / plain → prompt-control concatenates tensors, but our context
        window is a FIXED 512 tokens shared with the T5 encoder, so instead the
        terms are packed into that single window (term, space, term, ...). Same
        effect for prompts of ordinary length, and it keeps the compiled shape
        stable. Long multi-term prompts truncate at 512 like any other prompt.
    """
    style = (style or PROMPT_STYLE).lower()
    if style not in prompt_control.WEIGHT_MODES:
        log(f"unknown prompt style {style!r}, falling back to comfy")
        style = "comfy"

    def build(text):
        planned = prompt_control.plan(text or "", steps, weight_mode=style,
                                      tags=tags, seed=seed)
        contexts = []
        for result in planned["states"]:
            pairs = [prompt_control._segments_to_text(term) for term in result["terms"]]
            pairs = [(prompt_control.apply_escapes(t), cw) for t, cw in pairs]
            pairs = [(t, cw) for t, cw in pairs if t]
            if not pairs:
                pairs = [("", np.zeros(0, np.float32))]
            ops = result["ops"]

            if len(pairs) > 1 and ops and all(o[0] == "AVG" for o in ops):
                ctxs = []
                for t, cw in pairs:
                    qh, qm, ti, tm = _encode_text_list([(t, cw)])[0]
                    c = cond_forward(cond_params, jnp.asarray(qh, dtype=jnp.float32),
                                     qm, ti, tm).astype(jnp.bfloat16)
                    c.block_until_ready()
                    ctxs.append(c.astype(jnp.float32))
                acc = ctxs[0]
                for k, (_, w) in enumerate(ops):
                    nxt = ctxs[k + 1]
                    acc = acc * float(w) + nxt * (1.0 - float(w))
                ctx = acc.astype(jnp.bfloat16)
            else:
                buf, wbuf = [], []
                for t, cw in pairs:
                    if buf:
                        buf.append(" "); wbuf.append(1.0)
                    buf.append(t)
                    wbuf.extend(cw.tolist() if len(cw) == len(t) else [1.0] * len(t))
                full = "".join(buf)
                cw_full = np.asarray(wbuf, np.float32)
                qh, qm, ti, tm = _encode_text_list([(full, cw_full)])[0]
                ctx = cond_forward(cond_params, jnp.asarray(qh, dtype=jnp.float32),
                                   qm, ti, tm).astype(jnp.bfloat16)
                ctx.block_until_ready()
            contexts.append(ctx)
        return contexts, planned["step_state"], planned

    pos_ctxs, pos_map, pos_plan = build(prompt)
    neg_ctxs, neg_map, neg_plan = build(neg)
    log(f"prompt control [{style}]: pos {prompt_control.describe(pos_plan)} | "
        f"neg {prompt_control.describe(neg_plan)}")
    return (pos_ctxs, pos_map, pos_plan), (neg_ctxs, neg_map, neg_plan)
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

# --- Optional LoRA merge (must happen on CPU before device_put) ---
# ANIMA_LORAS is a JSON list of {"path": ..., "scale": ...} merged in one pass.
# ANIMA_LORA_PATH / ANIMA_LORA_SCALE remain supported for a single LoRA.
def _lora_specs():
    raw = os.environ.get("ANIMA_LORAS", "").strip()
    specs = []
    if raw:
        try:
            for item in json.loads(raw):
                p = str(item.get("path", "")).strip()
                if p:
                    specs.append((p, float(item.get("scale", 1.0))))
        except Exception as e:
            log(f"ANIMA_LORAS is not valid JSON ({type(e).__name__}: {e}); ignoring it")
    if not specs:
        p = os.environ.get("ANIMA_LORA_PATH", "").strip()
        if p:
            specs.append((p, float(os.environ.get("ANIMA_LORA_SCALE", "1.0"))))
    return specs

lora_specs = [(p, s) for p, s in _lora_specs() if os.path.exists(p)]
missing = [(p, s) for p, s in _lora_specs() if not os.path.exists(p)]
for p, _ in missing:
    log(f"WARNING: LoRA path not found, skipping: {p}")

if lora_specs:
    log("merging %d LoRA(s): %s" % (len(lora_specs),
        ", ".join(f"{os.path.basename(p)}@{s}" for p, s in lora_specs)))
    t_lora0 = time.perf_counter()
    try:
        t_params, lora_stats = load_and_merge_anima_loras(
            t_params, lora_specs, dtype=jnp.bfloat16)
        log(f"LoRA merged: {lora_stats['merged']} layer-targets, "
            f"{lora_stats['skipped']} skipped | format={lora_stats.get('format')} "
            f"targets={lora_stats.get('targets')} "
            f"convention={lora_stats.get('convention')} "
            f"in {time.perf_counter()-t_lora0:.2f}s")
        # A partially applied LoRA still produces a plausible image, so an
        # unexpected skip count must be loud rather than silent.
        if lora_stats.get("skipped"):
            log(f"WARNING: {lora_stats['skipped']} layer-targets were skipped; "
                f"the LoRA is only partially applied")
        for _d in lora_stats.get("details", [])[:10]:
            log(f"  lora: {_d}")
    except Exception:
        log("LoRA merge failed:\n" + traceback.format_exc())
        raise
else:
    log("no LoRA configured (ANIMA_LORAS / ANIMA_LORA_PATH) or file not found")

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
    """Decode a (B,16,1,h,w) latent batch -> list of B uint8 HxWx3 arrays.

    Each sample is decoded on its own, so the (1,...)-shaped VAE executable that warmup
    compiled is reused for every batch size: no per-B recompile, and decode memory stays flat.
    """
    lmean = jnp.array(vae.latents_mean, dtype=latents.dtype).reshape(1, 16, 1, 1, 1)
    lstd = jnp.array(vae.latents_std, dtype=latents.dtype).reshape(1, 16, 1, 1, 1)
    graphdef, state, rest = nnx.split(vae, nnx.Param, ...)
    out = []
    for b in range(latents.shape[0]):
        z = latents[b:b + 1] / (1.0 / lstd) + lmean
        merged = nnx.merge(graphdef, state, rest)
        video = merged.decode(z, AutoencoderKLWanCache(merged), return_dict=False)[0]
        video = jnp.clip(video / 2.0 + 0.5, 0.0, 1.0)
        video.block_until_ready()
        img = np.asarray(video)
        if img.ndim == 5:
            img = img[:, 0]
        if img.shape[-1] not in (1, 3):
            img = np.moveaxis(img, 1, -1)
        out.append((img[0] * 255.0).round().astype(np.uint8))
    return out

def grid_cols(n, w, h):
    """Column count whose grid is closest to a 3:2 landscape, with a small penalty per empty cell."""
    best = None
    for cols in range(1, n + 1):
        rows = -(-n // cols)
        score = abs(math.log((cols * w) / (rows * h) / 1.5)) + 0.15 * (rows * cols - n)
        if best is None or score < best[0]:
            best = (score, cols)
    return best[1]

def make_grid_u8(imgs):
    if len(imgs) == 1:
        return imgs[0]
    h, w = imgs[0].shape[:2]
    cols = grid_cols(len(imgs), w, h)
    rows = -(-len(imgs) // cols)
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas

def save_u8(u8, path, step=None, steps=None, ms_per_step=None, guidance=None):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(u8)
    # Overlay progress text so the live preview is self-describing: without it the
    # browser can look frozen when a preview is followed by a very similar preview.
    try:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except Exception:
            font = ImageFont.load_default()
        text = f"step {step or '?'}/{steps or '?'}, {ms_per_step or '?'} ms/step, CFG {guidance or '?'}"
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    except Exception:
        pass
    tmp = path + ".tmp.png"
    img.save(tmp)
    os.replace(tmp, path)

# (batch, H, W) shapes the transformer has compiled for, (H, W) shapes the VAE has, and the VAE
# shapes whose PREVIEW decode failed during the CURRENT request. A new shape means a one-time XLA
# compile, which the UI is told about so it does not look frozen. _vae_failed is reset per request
# on purpose: it exists so the give-up path stops re-advertising a compile for a shape that just
# failed, NOT to blacklist that shape for the server's whole life -- a later request at the same
# shape pays the same real compile and must still say so.
_tf_compiled = set(); _vae_compiled = set(); _vae_failed = set()

# Preview bookkeeping, at module level, with _pv_state() defined here rather than inside the
# request body. The request loop is itself module level, so EVERY progress record can carry the
# state -- including "starting" and the transformer "compiling" record, which fire before the
# denoise loop, and "decode"/"error", which fire after it. A closure defined inside the request
# body would raise NameError in the request's own error handler whenever a request died before
# that def -- exactly the case where the client most needs to know whether previews had failed.
#
# preview_fails counts CONSECUTIVE failures (it drives the give-up policy) and resets on success;
# preview_fails_total is cumulative, so a recovered transient failure still shows in "done".
prev_ok = 0
prev_fail = 0
prev_fails_total = 0
prev_err = ""
prev_giveup = False
_pv_expected = 0


def _pv_state():
    return {"preview_ok": prev_ok, "preview_fails": prev_fail,
            "preview_fails_total": prev_fails_total, "preview_total": _pv_expected,
            "preview_gaveup": prev_giveup, "preview_error": prev_err or None}
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
    _tf_compiled.add((1, 1024, 1024))
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
    log(f"warmup: VAE decode compiled+ran at 1024px in {time.perf_counter()-t0:.1f}s; out {_wv[0].shape}; {hbm_stats()}")
    _vae_compiled.add((1024, 1024))
    del _wz, _wv
    gc.collect()
    log(f"warmup done; executables compiled; host RSS {rss_gb():.2f} GB")
except Exception:
    log("warmup raised:\n" + traceback.format_exc())
    raise

log(f"SERVER READY total {(time.perf_counter()-t_all)/60:.1f} min; watching {REQ_PATH}")
write_prog(stage="ready", step=0, steps=0, gen_id="warmup")

def default_req():
    return {"prompt": "masterpiece, best quality, 1girl",
            "negative_prompt": "worst quality, low quality, blurry",
            "height": 1024, "width": 1024, "steps": 30, "guidance": 4.0,
            # -1 = pick a random seed; the server resolves it and reports the value.
            "seed": -1, "preview_every": 5, "out": "/content/anima_perstep.png"}

if not os.path.exists(REQ_PATH):
    json.dump({**default_req(), "go": False}, open(REQ_PATH, "w"))

last_mtime = 0.0
last_gen_id = ""          # id of the last request we actually ran
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
    # Every progress record carries the id of the request that produced it. Without it a
    # client cannot tell its own result from a previous run's: the UI polled the progress
    # file, found the leftover stage="done" record, and returned the previous image
    # instantly instead of waiting for the request it had just submitted.
    gen_id = str(d.get("gen_id") or "")
    if gen_id and gen_id == last_gen_id:
        # Already ran this exact request (a duplicate write or a re-touched file).
        log(f"ignoring request gen_id {gen_id}: already handled")
        json.dump({**d, "go": False, "status": "duplicate"}, open(REQ_PATH, "w"))
        last_mtime = os.path.getmtime(REQ_PATH)
        continue
    last_gen_id = gen_id
    json.dump({**d, "go": False, "status": "running"}, open(REQ_PATH, "w"))
    # Reset before the FIRST record of this request, so a client polling "starting" can
    # never see the previous request's preview_gaveup / preview_error.
    prev_ok = 0
    prev_fail = 0
    prev_fails_total = 0
    prev_err = ""
    prev_giveup = False
    _pv_expected = 0
    _vae_failed = set()
    write_prog(stage="starting", step=0, steps=d.get("steps", 30), gen_id=gen_id,
               **_pv_state())
    last_mtime = os.path.getmtime(REQ_PATH)
    try:
        # One prompt, one negative, N images. Older clients sent lists: use the first entry.
        _p = d.get("prompt", "")
        if isinstance(_p, (list, tuple)):
            _p = _p[0] if _p else ""
        _n = d.get("negative_prompt", "")
        if isinstance(_n, (list, tuple)):
            _n = _n[0] if _n else ""
        BATCH = max(1, min(8, int(d.get("batch_count", 1) or 1)))
        H, W, STEPS, GUIDANCE, SEED = d["height"], d["width"], d["steps"], d["guidance"], d["seed"]
        OUT = d["out"]
        PREV_EVERY = int(d.get("preview_every", 5) or 0)
        # Compute the expected preview count up front and log it, so "off" is explicit in the
        # server log rather than a silent truthiness fall-through (preview_every=0 and a
        # missing/None value both mean "no previews", which is invisible to the client).
        _pv_expected = sum(1 for _i in range(int(STEPS))
                           if PREV_EVERY and ((_i + 1) % PREV_EVERY == 0 or _i == STEPS - 1))
        log(f"preview_every={PREV_EVERY} -> {_pv_expected} VAE previews "
            f"{'(OFF: no live preview will be written)' if not PREV_EVERY else ''}")
        _MULT = 16
        H0, W0 = int(H), int(W)
        H = max(_MULT, int(round(H0 / _MULT)) * _MULT)
        W = max(_MULT, int(round(W0 / _MULT)) * _MULT)
        if (H, W) != (H0, W0):
            log(f"requested {H0}x{W0} is not a multiple of {_MULT}; using {H}x{W} "
                f"(VAE stride 8, patchify stride 2)")
        _seed_req = int(SEED)
        if _seed_req < 0:
            SEED = int(np.random.default_rng().integers(0, 2**31 - 1 - 8))
            log(f"seed {_seed_req} requested; using random seed {SEED}")
        # Image k uses seed SEED + k, so any image of a batch can be reproduced on its own.
        seeds = [int(SEED) + k for k in range(BATCH)]
        try:
            os.remove(SNAP_PATH)  # never let a client show the previous run's preview
        except FileNotFoundError:
            pass
        g0 = time.perf_counter()

        write_prog(stage="text-encoding", step=0, steps=STEPS, seed=int(SEED), seeds=seeds,
                   gen_id=gen_id, batch_total=BATCH)
        # Prompt-control scheduling: every distinct state the schedule passes through is
        # encoded once up front, then the denoise loop indexes the right one per step.
        # All states share the same (1,512,1024) shape, so switching costs no recompile.
        tags = d.get("tags") or None
        style = d.get("prompt_style") or PROMPT_STYLE
        seed_i = int(SEED) if int(SEED) >= 0 else 0
        (pos_ctxs, pos_map, pos_plan), (neg_ctxs, neg_map, neg_plan) = encode_texts(
            _p, _n, steps=STEPS, style=style, tags=tags, seed=seed_i)
        if BATCH > 1:
            pos_ctxs = [jnp.repeat(c, BATCH, axis=0) for c in pos_ctxs]
            neg_ctxs = [jnp.repeat(c, BATCH, axis=0) for c in neg_ctxs]
        write_prog(stage="text-encoded", step=0, steps=STEPS, seed=int(SEED), seeds=seeds,
                   gen_id=gen_id, batch_total=BATCH,
                   prompt_states=max(len(pos_ctxs), len(neg_ctxs)),
                   prompt_switches=[round(b, 3) for b in
                                    sorted(set(pos_plan["breakpoints"]) |
                                           set(neg_plan["breakpoints"])) if 0 < b < 1])

        sched = FlaxFlowMatchScheduler()
        st = FlowMatchSchedulerState.create()
        sigmas_raw = np.linspace(1.0, 1.0 / STEPS, STEPS).astype(np.float32)
        st = sched.set_timesteps(st, sigmas=jnp.asarray(sigmas_raw))
        timesteps = np.asarray(st.timesteps)
        sigmas = np.asarray(st.sigmas[:-1])

        # All BATCH images share every transformer call: latents (B,16,1,h,w), context
        # (B,512,1024), pad (B,1,H,W). The padding mask is concatenated channel-wise inside the
        # transformer, so its batch dim must equal the latents' batch dim.
        latents = jnp.asarray(np.concatenate(
            [np.random.default_rng(s).standard_normal((1, 16, 1, H // 8, W // 8)).astype(np.float32)
             for s in seeds], axis=0))
        pad = jnp.zeros((BATCH, 1, H, W), dtype=jnp.float32)
        common = dict(steps=STEPS, seed=int(SEED), seeds=seeds, gen_id=gen_id, batch_total=BATCH)
        if (BATCH, H, W) not in _tf_compiled:
            log(f"new transformer shape batch={BATCH} {H}x{W}: XLA compile on first step")
            write_prog(stage="compiling", step=0, note=f"transformer {BATCH}x{W}x{H}",
                       **common, **_pv_state())
        d0 = time.perf_counter()
        prev_total = 0.0
        first_s = 0.0
        for i in range(STEPS):
            context = pos_ctxs[pos_map[i]]
            neg_context = neg_ctxs[neg_map[i]]
            pred = cfg_prediction(latents, jnp.asarray(np.float32(timesteps[i])),
                                  context, neg_context, pad, GUIDANCE)
            sigma_next = sigmas[i + 1] if i + 1 < STEPS else 0.0
            latents = latents + (jnp.asarray(np.float32(sigma_next - sigmas[i]))) * pred
            latents.block_until_ready()
            step_s = time.perf_counter() - d0 - prev_total
            if i == 0:
                first_s = step_s
                _tf_compiled.add((BATCH, H, W))
            # ms/step excludes step 1 (which carries any XLA compile) and preview decodes.
            ms = round((step_s - first_s) / i * 1000.0, 1) if i > 0 else round(step_s * 1000.0, 1)
            if i % 2 == 0 or i == STEPS - 1:
                write_prog(stage="denoise", step=i + 1, ms_per_step=ms, guidance=GUIDANCE,
                           **common, **_pv_state())
            if PREV_EVERY and not prev_giveup and ((i + 1) % PREV_EVERY == 0 or i == STEPS - 1):
                # A failed preview must never be silent: the old handler logged one line and
                # carried on, so the UI showed snap=None forever behind a healthy status line
                # and a normal "done". Every outcome is now counted and published to the
                # progress JSON (preview_ok / preview_fails / preview_error / preview_gaveup).
                # _pv_stage records which half failed, so a full disk or a PIL/grid error is
                # not reported to the client as a VAE decode failure. Guessing it from the
                # traceback text would mislabel it the moment a frame mentions "decode".
                _pv_t0 = time.perf_counter()
                _pv_stage = "decode"
                try:
                    if (H, W) not in _vae_compiled and (H, W) not in _vae_failed:
                        write_prog(stage="compiling", step=i + 1, note=f"VAE decode {W}x{H}",
                                   **common, **_pv_state())
                    _imgs = decode_latents(latents)
                    _pv_stage = "preview write"
                    save_u8(make_grid_u8(_imgs), SNAP_PATH, step=i + 1,
                            steps=STEPS, ms_per_step=ms, guidance=GUIDANCE)
                    _vae_compiled.add((H, W))
                    # ALL preview wall time, success or failure, is charged to prev_total.
                    # Otherwise a failed attempt's ~50s XLA compile lands in denoise_s and
                    # inflates ms_per_step, the figure the perf docs quote.
                    prev_total += time.perf_counter() - _pv_t0
                    prev_ok += 1
                    prev_fail = 0
                    prev_err = ""      # a recovered preview must not keep reporting failure
                    write_prog(stage="denoise", step=i + 1, preview_step=i + 1,
                               preview_s=round(prev_total, 2), ms_per_step=ms,
                               guidance=GUIDANCE, **common, **_pv_state())
                except Exception as _e:
                    prev_total += time.perf_counter() - _pv_t0
                    prev_fail += 1
                    prev_fails_total += 1
                    prev_err = f"{type(_e).__name__}: {str(_e)[:200]}"
                    _vae_failed.add((H, W))
                    log(f"preview failed at step {i+1} during {_pv_stage} "
                        f"({type(_e).__name__}): {_e}\n" + traceback.format_exc())
                    if prev_fail >= 2:
                        prev_giveup = True
                        log(f"previews disabled after {prev_fail} consecutive failures; "
                            f"denoise continues to completion")
                    write_prog(stage="denoise", step=i + 1, ms_per_step=ms,
                               guidance=GUIDANCE, **common, **_pv_state())
        latents.block_until_ready()
        denoise_s = time.perf_counter() - d0 - prev_total
        write_prog(stage="decode", step=STEPS, **common, **_pv_state())
        v0 = time.perf_counter()
        u8s = decode_latents(latents)
        _vae_compiled.add((H, W))
        stem, ext = os.path.splitext(OUT)
        ext = ext or ".png"
        if BATCH == 1:
            images = [OUT]
            save_u8(u8s[0], OUT)
        else:
            images = []
            for k, u in enumerate(u8s, start=1):
                per = f"{stem}_batch{k}{ext}"
                save_u8(u, per)
                images.append(per)
            save_u8(make_grid_u8(u8s), OUT)
            log(f"batch grid saved to {OUT}")
        decode_s = time.perf_counter() - v0
        total_s = time.perf_counter() - g0
        log(f"GEN DONE {OUT} {H}x{W} batch {BATCH} seeds {seeds[0]}..{seeds[-1]} "
            f"denoise {denoise_s:.1f}s decode {decode_s:.1f}s total {total_s:.1f}s")
        write_prog(stage="done", step=STEPS, out=OUT, grid=OUT, images=images,
                   elapsed_s=round(total_s, 1), denoise_s=round(denoise_s, 1),
                   decode_s=round(decode_s, 1), height=H, width=W, hbm=hbm_stats(),
                   **common, **_pv_state())
        json.dump({**d, "go": False, "status": "done", "elapsed_s": round(total_s, 1)},
                  open(REQ_PATH, "w"))
    except Exception as _err:
        log("gen raised:\n" + traceback.format_exc())
        # The preview state must survive into the error record too: a run that OOMs during
        # the final decode used to lose the fact that previews had already failed.
        write_prog(stage="error", gen_id=gen_id, error=f"{type(_err).__name__}: {str(_err)[:300]}",
                   **_pv_state())
        try:
            json.dump({**d, "go": False, "status": "error"}, open(REQ_PATH, "w"))
        except Exception:
            pass
    last_mtime = os.path.getmtime(REQ_PATH)
'''
r = subprocess.Popen([py, "-u", "-c", code], cwd=repo, env=env)
print("server pid", r.pid, flush=True)
