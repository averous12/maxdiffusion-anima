import os
import subprocess
import sys

repo = os.environ.get("ANIMA_REPO", "/content/maxdiffusion")
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
       "PYTHONPATH": repo + "/src", "HF_HOME": os.environ.get("HF_HOME", "/content/hf_cache"),
       "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}
env.pop("UV_SYSTEM_PYTHON", None)
print("server interpreter:", py, flush=True)

code = r'''
import os
import time

# Content root. Colab keeps the historical /content; ANIMA_CONTENT_ROOT lets the same
# script run (and be tested) on a normal machine without touching any other default.
CROOT = os.environ.get("ANIMA_CONTENT_ROOT", "/content")
LOG = open(os.path.join(CROOT, "anima_server.log"), "a", buffering=1)
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
    # Non-Linux fallback (local CPU runs of this same script): keep the field meaningful.
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
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
import math
import numpy as np
import gc
import traceback
import faulthandler
# Diagnostic only: exit=False so the watchdog can never kill a long-lived server.
# It dumps the live stack of every thread once, which is what localizes a stall.
faulthandler.dump_traceback_later(int(os.environ.get("ANIMA_SERVER_TIMEOUT_S", "1800")), exit=False)
import jax
jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32")

# Persistent XLA executable cache: the 52s VAE + 16s transformer warmup
# compiles are pure compile cost, keyed on SHAPES not weight values, so a
# server reboot (LoRA Apply, watchdog restart) can load them instead of
# recompiling. The cache key includes JAX/libtpu versions + topology, so a
# fresh VM allocation recompiles once — that is expected and cheap-ish.
# Drive is used ONLY when actually mounted (cold_start / anima_job do not
# mount it, and an unmounted makedirs would silently create a dead local
# directory under /content); otherwise fall back to the VM-local path, which
# still spans server reboots within this VM's life.
XLA_CACHE = os.environ.get("ANIMA_XLA_CACHE", "")
# One Drive tree for both caches: /content/drive/MyDrive/anima_cache/{hf,xla}.
# This default only exists so a forgotten env var cannot silently lose the
# cache; the agenda sets ANIMA_XLA_CACHE explicitly. The pre-unification
# /content/drive/MyDrive/anima_xla_cache stays readable as a fallback so
# entries written before the tree was unified are not thrown away.
DRIVE_XLA = "/content/drive/MyDrive/anima_cache/xla"
LEGACY_DRIVE_XLA = "/content/drive/MyDrive/anima_xla_cache"
if not XLA_CACHE:
    if os.path.isdir("/content/drive/MyDrive"):
        XLA_CACHE = DRIVE_XLA
    elif os.path.isdir(LEGACY_DRIVE_XLA):
        XLA_CACHE = LEGACY_DRIVE_XLA
    else:
        XLA_CACHE = "/content/anima_xla_cache"
try:
    os.makedirs(XLA_CACHE, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", XLA_CACHE)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    _persistent = XLA_CACHE.startswith("/content/drive/")
    log(f"XLA cache: {XLA_CACHE} (drive={_persistent} persistent={_persistent})")
    if not _persistent:
        # Loud on purpose: a silent fallback here means the ~68 s of warmup
        # compiles is paid every single session and nobody notices.
        log("XLA cache WARNING: NOT persistent -- this directory lives on the VM "
            "and is lost at the next allocation; set ANIMA_XLA_CACHE="
            + DRIVE_XLA + " to keep it")
except Exception as e:
    log(f"XLA cache disabled ({type(e).__name__}: {e}); compiling every boot")
    XLA_CACHE = ""
import jax.numpy as jnp
log("matmul precision: BF16_BF16_F32 (bf16 operands, fp32 accumulate; activations stay fp32)")

from maxdiffusion.models.anima_cosmos_flax import FlaxAnimaCosmosTransformer, convert_anima_aesthetic_weights
from maxdiffusion.models.anima_text_conditioner_flax import (
    AnimaTextConditionerConfig, FlaxAnimaTextConditioner,
    convert_anima_aesthetic_adapter_weights)
from maxdiffusion.models.anima_lora_flax import load_and_merge_anima_loras
from maxdiffusion.models import anima_snapref_flax as anima_snapref
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
# The project's Colab venv installs a TPU-only jax build, where jax.devices("cpu")
# raises "Unknown backend cpu". Fall back to the default device so the host-side
# weight conversion still runs (it just happens on the TPU device instead).
try:
    _CPU = jax.devices("cpu")[0]
except (RuntimeError, ValueError):
    _CPU = jax.devices()[0]
    log(f"no CPU backend in this jax build; host conversion uses {_CPU.device_kind}")

CACHE_DIR = os.path.join(CROOT, "anima_cache")  # retained for the HF cache only; no param cache
REQ_PATH = os.path.join(CROOT, "anima_request.json")
PROG_PATH = os.path.join(CROOT, "anima_progress.json")
SNAP_PATH = os.path.join(CROOT, "anima_snap.png")
# The client creates this file to cancel the running generation.
INTERRUPT_PATH = os.path.join(CROOT, "anima_interrupt")

def write_prog(**kw):
    try:
        tmp = PROG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"t": time.time(), "rss_gb": round(rss_gb(), 2), **kw}, f)
        os.replace(tmp, PROG_PATH)
    except Exception:
        pass

t_all = time.perf_counter()
# Weight cache: the two snapshot_download calls below used to hardcode
# /content/hf_cache, which ignored HF_HOME entirely -- so weights pre-fetched
# into Drive were never used and every session paid the ~5.6 GB download again.
# ANIMA_HF_CACHE wins, then HF_HOME, then the old VM-local path.
HF_CACHE = (os.environ.get("ANIMA_HF_CACHE") or os.environ.get("HF_HOME")
            or "/content/hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
log(f"weight cache (HF): {HF_CACHE} (persistent={HF_CACHE.startswith('/content/drive/')})")
# Weight sources are overridable from the notebook's "links" cell via env; empty
# = the standard Anima aesthetic family. Each is an HF repo id that
# snapshot_download / from_pretrained can resolve (the base repo must carry the
# tokenizer / text_encoder / t5_tokenizer subfolders the loader reads).
BASE_REPO = os.environ.get("ANIMA_BASE_REPO", "circlestone-labs/Anima-Base-v1.0-Diffusers").strip()
AESTHETIC_REPO = os.environ.get("ANIMA_AESTHETIC_REPO", "circlestone-labs/Anima").strip()
AESTHETIC_FILE = os.environ.get("ANIMA_AESTHETIC_FILE",
                                "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors").strip()
log(f"weight sources: base={BASE_REPO} | aesthetic={AESTHETIC_REPO}/{AESTHETIC_FILE}")
snapshot_dir = snapshot_download(BASE_REPO, cache_dir=HF_CACHE)
aesthetic_snapshot = snapshot_download(AESTHETIC_REPO, cache_dir=HF_CACHE,
                                       allow_patterns=[AESTHETIC_FILE])
import shutil
aesthetic_path = os.path.join(CROOT, os.path.basename(AESTHETIC_FILE))
_src = os.path.join(aesthetic_snapshot, AESTHETIC_FILE)
if not os.path.exists(aesthetic_path) or os.path.getsize(aesthetic_path) != os.path.getsize(_src):
    shutil.copyfile(_src, aesthetic_path)
log(f"base weights at {snapshot_dir}")
log(f"aesthetic transformer+conditioner: {aesthetic_path} ({os.path.getsize(aesthetic_path)/1e9:.2f} GB)")

log("=== SERVER: torch tokenizer + TPU Qwen3 load (once) ===")
t0 = time.perf_counter()
from transformers import AutoConfig as HFAutoConfig, AutoTokenizer
from tokenizers import Tokenizer as _RawTokenizer
tok = AutoTokenizer.from_pretrained(BASE_REPO, subfolder="tokenizer")
_t5 = _RawTokenizer.from_file(os.path.join(snapshot_dir, "t5_tokenizer", "tokenizer.json"))
_t5.enable_padding(pad_id=0, pad_token="<pad>", length=512)
_t5.enable_truncation(max_length=512)
pc = HFAutoConfig.from_pretrained(BASE_REPO, subfolder="text_encoder")
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
def swap_loras(specs):
    """Rebuild t_params from the pristine as-booted weights with `specs` merged.

    No restart, no torch->jax re-conversion, no VAE reload, no warmup, no XLA
    compile: the transformer's executable is keyed on shapes, not values, so the
    new weight arrays reuse it. Measured ~1.1 s for a real 665 MB CivitAI LoRA
    against 366 s for a reboot.

    `specs` is [] to go back to the bare base model.
    """
    global t_params, _LORA_SPECS
    t0 = time.perf_counter()
    work = dict(_T_PARAMS_PRISTINE)
    stats = {"merged": 0, "skipped": 0, "details": []}
    t_merge = 0.0
    if specs:
        # The merger mutates the dict it is handed, so it gets a shallow copy of
        # the pristine tree and never the tree itself -- otherwise repeated
        # swaps would stack deltas and drift.
        os.environ["ANIMA_MERGE_DEVICE"] = "device"
        tm = time.perf_counter()
        work, stats = load_and_merge_anima_loras(work, specs, dtype=jnp.bfloat16)
        t_merge = time.perf_counter() - tm
    # Split the two costs: the merge is compute (one-time kernel compiles on the
    # first call), the device_put is 3.9 GB of host->accelerator traffic and is
    # paid on every single swap, LoRA or not.
    tp = time.perf_counter()
    t_params = jax.device_put(work, jax.devices()[0])
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), t_params)
    t_put = time.perf_counter() - tp
    del work
    secs = time.perf_counter() - t0
    _LORA_SPECS = list(specs)
    log(f"LoRA swap: merged={stats['merged']} skipped={stats['skipped']} "
        f"in {secs:.2f}s (merge {t_merge:.2f}s + device_put {t_put:.2f}s) "
        f"| {[os.path.basename(p) + '@' + str(s) for p, s in specs] or 'base'}"
        f" | {hbm_stats()}")
    for d in (stats.get("details") or [])[:5]:
        log(f"  swap: {d}")
    return stats, secs, t_merge, t_put


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

# What is merged into the weights right now. swap_loras maintains this; seeding
# it from the boot spec matters because the diff below compares against it -- an
# empty seed would make the first prompt re-merge a LoRA that is already loaded,
# and would make a tag-free prompt look like it needed nothing when the boot
# LoRA was still resident.
_LORA_SPECS = list(lora_specs)

# Directories scanned for <lora:name:scale> resolution. Both are accepted: the
# UI's own folder and the one the benchmark tooling downloads into.
LORA_DIRS = [d for d in (os.environ.get("ANIMA_LORA_DIR", "").strip(),
                         os.path.join(CROOT, "anima_lora"), os.path.join(CROOT, "loras")) if d]


def installed_lora_files():
    """Every .safetensors under LORA_DIRS, cheapest-first ordered."""
    out = []
    for d in LORA_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".safetensors"):
                    out.append(os.path.join(root, f))
    return sorted(set(out))

t0 = time.perf_counter()
# Keep the as-booted tree alive on the host: it is what every later hot-swap
# rebuilds from. jax.device_put does not consume the host arrays, so this holds
# a reference rather than making a copy, and HBM is unchanged.
_T_PARAMS_PRISTINE = t_params
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

# --- SnapRef (reference adapter) -------------------------------------------------------
# Every callable below is a *separate* jit with a different signature, so the plain path
# above keeps its existing executable and behaviour exactly as before. Nothing here runs
# unless a request carries "snapref".
# A second module instance whose scan emits every layer's self-attention (k, v): the
# reference pass needs those, the target pass does not, so they cannot share one instance.
_SNAPREF_RECORDER = FlaxAnimaCosmosTransformer(layers=28, scan_blocks=True, record_kv=True)

@jax.jit
def snapref_target_forward(tp, lat, t_vec, ctx, pad, lora, refkv, refbias):
    return transformer.apply({"params": tp}, lat, t_vec, ctx, None, pad,
                             lora=lora, ref_kv=refkv, ref_bias=refbias)

@jax.jit
def snapref_reference_forward(tp, lat, ctx, pad, cos, sin, mode_vec, lora):
    # record_kv is a Module field, not a call argument: a separate instance is what makes the
    # scan emit every layer's self-attention (k, v) instead of the residual stream alone.
    return _SNAPREF_RECORDER.apply({"params": tp}, lat,
                                   jnp.zeros((lat.shape[0],), jnp.float32), ctx, None, pad,
                                   rope_override=(cos, sin), t_embed_add=mode_vec, lora=lora)

def snapref_encode_latent(z):
    """Reference pixels (1,3,1,H,W) -> sampler-space latent, jitted on one active shape."""
    global _SNAPREF_ENC_KEY
    key = (int(z.shape[-2]), int(z.shape[-1]))
    if key != _SNAPREF_ENC_KEY:
        _SNAPREF_ENC.clear()
        _SNAPREF_ENC_CACHE.clear()
        _SNAPREF_ENC_KEY = key
        _, merged = anima_snapref.encode_reference_latent(vae, z, vae.latents_mean, vae.latents_std)
        _SNAPREF_ENC_CACHE[key] = (merged,)
        _SNAPREF_ENC[key] = jax.jit(lambda q, _m=merged: anima_snapref.encode_reference_latent(
            _m, q, vae.latents_mean, vae.latents_std)[0])
    return _SNAPREF_ENC[key](z)

_SNAPREF_ENC = {}
_SNAPREF_ENC_CACHE = {}
_SNAPREF_ENC_KEY = None

def cfg_prediction(latents, timestep, context, neg_context, pad, guidance):
    t_vec = jnp.broadcast_to(timestep / jnp.float32(1000.0), (latents.shape[0],)).astype(jnp.float32)
    nc = transformer_forward(t_params, latents, t_vec, context, pad)
    nu = transformer_forward(t_params, latents, t_vec, neg_context, pad)
    nc32 = nc.astype(jnp.float32)
    nu32 = nu.astype(jnp.float32)
    return nu32 + jnp.float32(guidance) * (nc32 - nu32)

# --- SnapRef session: reference prep + reference-aware CFG -------------------------------
SNAPREF_DIRS = [d for d in (os.environ.get("ANIMA_SNAPREF_DIR", "").strip(),
                            os.path.join(CROOT, "anima_snapref"),
                            os.path.join(os.path.dirname(LORA_DIRS[0]), "snapref"))
                if d]
_SNAPREF_ADAPTERS = {}          # current adapter path -> SnapRefAdapter (lazy)
_SNAPREF_ACTIVE_ADAPTER = None
_SNAPREF_STATE = {"kv": None}   # cached reference K/V of the current run

def clear_snapref_runtime():
    """Release request-scoped reference K/V without evicting reusable XLA executables."""
    _SNAPREF_STATE["kv"] = None
    gc.collect()


def installed_snapref_files():
    out = []
    for d in SNAPREF_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".safetensors") and ("snapref" in f.lower() or "anima_ref_kv" in f.lower()):
                    out.append(os.path.join(root, f))
    return sorted(set(out))

def resolve_snapref_adapter(name):
    """Accept an absolute path, a bare file name, or '' (the only installed file)."""
    files = installed_snapref_files()
    name = (name or "").strip()
    if not name:
        if not files:
            raise anima_snapref.SnapRefError("no SnapRef adapter installed; upload one to "
                                             + ", ".join(SNAPREF_DIRS))
        return files[0]
    if os.path.exists(name):
        return name
    for f in files:
        if os.path.basename(f) == name or os.path.basename(f).rsplit(".", 1)[0] == name:
            return f
    raise anima_snapref.SnapRefError(f"SnapRef adapter not found: {name} (have {[os.path.basename(f) for f in files]})")

def get_snapref_adapter(name):
    global _SNAPREF_ACTIVE_ADAPTER
    path = resolve_snapref_adapter(name)
    if _SNAPREF_ACTIVE_ADAPTER != path:
        _SNAPREF_ADAPTERS.clear()
        gc.collect()
        _SNAPREF_ACTIVE_ADAPTER = path
    if path not in _SNAPREF_ADAPTERS:
        t0 = time.perf_counter()
        _SNAPREF_ADAPTERS[path] = anima_snapref.load_snapref(path, num_layers=28, dtype=jnp.bfloat16)
        ad = _SNAPREF_ADAPTERS[path]
        log(f"SnapRef adapter loaded in {time.perf_counter()-t0:.2f}s: {os.path.basename(path)} "
            f"(release {ad.meta.get('version', 'unknown')}, step {ad.meta.get('step', '?')}, "
            f"streams {list(ad.stream_names())}, ref_t {ad.ref_t}, modes {list(ad.modes)}, "
            f"mode_embed {None if ad.mode_embed is None else ad.mode_embed.shape})")
    return _SNAPREF_ADAPTERS[path]

def _open_reference_image(path):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img.convert("RGB")


def _load_reference_pixels(path, megapixels, canvas_size=None):
    """Reference image file -> (1,3,1,h,w) pixels in [-1,1]."""
    from PIL import Image
    img = _open_reference_image(path)
    if canvas_size is None:
        w, h = anima_snapref.reference_latent_size(img.width, img.height, megapixels)
        img = img.resize((w, h), Image.LANCZOS)
    else:
        w, h = (int(canvas_size[0]), int(canvas_size[1]))
        img.thumbnail((w, h), Image.LANCZOS)
        canvas = Image.new("RGB", (w, h), "white")
        canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
        img = canvas
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, :, None]          # (1,3,1,h,w)
    return jnp.asarray(arr * 2.0 - 1.0), (w, h)


def _load_stitched_reference_pixels(paths, megapixels):
    """Stitch references side-by-side, preserving each image's aspect ratio."""
    from PIL import Image
    imgs = [_open_reference_image(p) for p in paths]
    if not imgs:
        raise anima_snapref.SnapRefError("stitch layout has no positive references")
    target_h = max(im.height for im in imgs)
    resized = []
    for im in imgs:
        w = max(1, round(im.width * target_h / im.height))
        resized.append(im.resize((w, target_h), Image.LANCZOS))
    canvas = Image.new("RGB", (sum(im.width for im in resized), target_h), "white")
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width
    w, h = anima_snapref.reference_latent_size(canvas.width, canvas.height, megapixels)
    canvas = canvas.resize((w, h), Image.LANCZOS)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, :, None]
    return jnp.asarray(arr * 2.0 - 1.0), (w, h)

def _snapref_cond_tensor(text):
    """Encode the optional appearance caption the same way the positive prompt is encoded."""
    pairs = [(text or "", np.ones(max(1, len(text or "")), np.float32))]
    qh, qmask, t5ids, t5mask = _encode_text_list(pairs)[0]
    return cond_forward(cond_params, jnp.asarray(qh, dtype=jnp.float32), jnp.asarray(qmask),
                        jnp.asarray(t5ids), jnp.asarray(t5mask))

def prepare_snapref(cfg, target_hw):
    """Encode the reference(s) once per run: one reference pass, cached per-block K/V."""
    t0 = time.perf_counter()
    ad = get_snapref_adapter(cfg.get("adapter"))
    refs = cfg.get("images") or []
    if not refs:
        raise anima_snapref.SnapRefError("snapref.images is empty")
    weights = [float(w) for w in (cfg.get("weights") or [1.0] * len(refs))]
    if len(weights) != len(refs):
        raise anima_snapref.SnapRefError("snapref.weights must match snapref.images")
    if sum(1 for w in weights if w > 0) == 0:
        raise anima_snapref.SnapRefError("every reference has weight 0")
    mode = str(cfg.get("mode") or "cs")
    requested_megapixels = float(cfg.get("ref_megapixels") or 1.0)
    layout = str(cfg.get("multi_layout") or "separate").lower()
    if layout not in ("separate", "stitch"):
        raise anima_snapref.SnapRefError("snapref.multi_layout must be 'separate' or 'stitch'")
    active = [(p, w) for p, w in zip(refs, weights) if w > 0]
    if not active:
        raise anima_snapref.SnapRefError("every reference has weight 0")
    megapixels = anima_snapref.reference_megapixels_for_layout(
        requested_megapixels, len(active), layout,
        float(os.environ.get("ANIMA_SNAPREF_REF_MP_BUDGET", "1.0")))
    canvas_size = (anima_snapref.reference_canvas_size(megapixels)
                   if layout == "separate" else None)
    if megapixels != requested_megapixels:
        log(f"SnapRef separate layout: {len(active)} references share "
            f"{requested_megapixels:.3g} MP -> {megapixels:.3g} MP each")
    if canvas_size is not None:
        log(f"SnapRef separate layout: canonical reference canvas {canvas_size[0]}x{canvas_size[1]}")
    caption = str(cfg.get("ref_caption") or "")
    ref_ctx = _snapref_cond_tensor(caption)
    mode_vec = ad.mode_vector(mode)
    if layout == "stitch":
        paths = [p for p, _w in active]
        px, (pw, ph) = _load_stitched_reference_pixels(paths, megapixels)
        lat = snapref_encode_latent(px)
        rh, rw = max(1, ph // 16), max(1, pw // 16)
        th, tw = max(1, target_hw[0] // 16), max(1, target_hw[1] // 16)
        cos, sin = anima_snapref.reference_rope((th, tw), (rh, rw), ad.ref_t)
        _out, kv = snapref_reference_forward(t_params, lat, ref_ctx, jnp.zeros(
            (1, 1, int(target_hw[0]), int(target_hw[1])), jnp.float32), cos, sin, mode_vec,
            ad.lora_tree("ref"))
        kv[0].block_until_ready()
        records, sizes = [kv], [int(kv[0].shape[2])]
        effective_weights = [max(weights)]
        log(f"SnapRef stitched {len(paths)} references -> {pw}x{ph}px, {sizes[0]} tokens")
    else:
        records, sizes, effective_weights = [], [], []
        for path, w in active:
            px, (pw, ph) = _load_reference_pixels(path, megapixels, canvas_size)
            lat = snapref_encode_latent(px)
            rh, rw = max(1, ph // 16), max(1, pw // 16)
            th, tw = max(1, target_hw[0] // 16), max(1, target_hw[1] // 16)
            cos, sin = anima_snapref.reference_rope((th, tw), (rh, rw), ad.ref_t)
            _out, kv = snapref_reference_forward(t_params, lat, ref_ctx, jnp.zeros(
                (1, 1, int(target_hw[0]), int(target_hw[1])), jnp.float32), cos, sin, mode_vec,
                ad.lora_tree("ref"))
            kv[0].block_until_ready()
            records.append(kv); sizes.append(int(kv[0].shape[2])); effective_weights.append(w)
            log(f"SnapRef reference {os.path.basename(path)} -> {pw}x{ph}px, {sizes[-1]} tokens, weight {w}")
    cache = anima_snapref.concat_reference_kv(records)
    size_list = list(sizes)
    bias = anima_snapref.key_biases(size_list,
                                    anima_snapref.reference_biases(cfg.get("strength", 0.0),
                                                                   effective_weights),
                                    (target_hw[0] // 16) * (target_hw[1] // 16))
    _SNAPREF_STATE["kv"] = cache
    secs = time.perf_counter() - t0
    log(f"SnapRef reference cached: {len(records)} image(s), {sum(size_list)} tokens, "
        f"{secs:.2f}s, mode {mode}, strength {cfg.get('strength', 0.0)}")
    return {"adapter": os.path.basename(ad.path), "mode": mode, "sizes": size_list,
            "weights": [w for w in weights if w > 0], "biases": list(bias.shape),
            "ref_seconds": round(secs, 3), "tokens": int(sum(size_list)),
            "ref_pixels": [
                (canvas_size if canvas_size is not None else
                 _load_reference_pixels(p, megapixels)[1])
                for p in refs
            ][:len(records)]}

def snapref_cfg_prediction(latents, timestep, context, neg_context, pad, guidance, cfg):
    """CFG along the SnapRef branches, then the documented post-CFG steps.

    Off (the trained default): only the positive branch reads the reference, so cfg pushes
    toward it — exactly the initial CFG per-step sample.
    ref_in_uncond: both branches read it.
    fidelity < 1: every row reads it, the negative rows with an extra log(1-f) key bias.
    ref_guidance_on: both read it, plus one reference-free negative call per step, and
      guided = base + rg*(uncond_ref - base) + cfg*(cond_ref - uncond_ref), then APG.
    """
    cache = _SNAPREF_STATE["kv"]
    ad = get_snapref_adapter(cfg.get("adapter"))
    lora = ad.lora_tree("tgt")
    sizes = list(cfg["_sizes"])
    biases = [anima_snapref.reference_biases(cfg.get("strength", 0.0), cfg["_weights"])[i]
              for i in range(len(cfg["_weights"]))]
    # The transformer patchifies the latent 2x2, so the token count is the latent grid
    # divided by the patch area -- not the latent pixel count. Using the raw grid made the
    # reference bias 4x too long and the attention add failed to broadcast.
    _pa = 2
    tgt_tokens = (latents.shape[-2] // _pa) * (latents.shape[-1] // _pa)
    t_vec = jnp.broadcast_to(timestep / jnp.float32(1000.0), (latents.shape[0],)).astype(jnp.float32)

    rg_on = bool(cfg.get("ref_guidance_on"))
    in_uncond = bool(cfg.get("ref_in_uncond")) or rg_on
    fid = cfg.get("fidelity", 1.0)
    ubias = anima_snapref.uncond_bias_for_fidelity(fid, in_uncond, rg_on)

    def run(ctx, extra_bias=0.0, with_ref=True):
        if not with_ref:
            return snapref_target_forward(t_params, latents, t_vec, ctx, pad, lora, None, None)
        b = jnp.asarray(anima_snapref.key_biases(sizes, biases, tgt_tokens, extra_bias))
        return snapref_target_forward(t_params, latents, t_vec, ctx, pad, lora, cache, b)

    cond_ref = run(context)
    if not in_uncond and ubias is None:
        uncond_ref = run(neg_context, with_ref=False)     # trained default
        base = uncond_ref
    elif ubias is None:
        uncond_ref = run(neg_context)
        base = uncond_ref
    else:
        uncond_ref = run(neg_context, extra_bias=ubias)
        base = None

    cond_ref = cond_ref.astype(jnp.float32)
    uncond_ref = uncond_ref.astype(jnp.float32)
    if not rg_on:
        out = uncond_ref + jnp.float32(guidance) * (cond_ref - uncond_ref)
    else:
        if not anima_snapref.guidance_active(float(np.asarray(timestep)), float(cfg.get("ref_end", 1.0))):
            out = uncond_ref + jnp.float32(guidance) * (cond_ref - uncond_ref)
        else:
            base = run(neg_context, with_ref=False).astype(jnp.float32)
            out = anima_snapref.ref_guidance_combine(cond_ref, uncond_ref, base,
                                                     guidance, cfg.get("ref_guidance", 4.5))
        out = anima_snapref.snapref_correct(out, cond_ref, cfg.get("correction") or "apg")
    return out

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

def save_u8(u8, path):
    """Save a numpy RGB (H, W, 3) array as a lossless PNG.

    No overlay: images used to carry a burned-in "step X/Y, N ms/step, CFG Z"
    progress stamp so consecutive previews did not look frozen. The UI now has
    a real progress bar and status line, and the stamp ended up permanently in
    archived/final images, so it was removed — output pixels == input pixels.
    """
    from PIL import Image
    img = Image.fromarray(u8)
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
    # Warmup shape. Colab/TPU keeps 1024; ANIMA_WARMUP_PX lets the same script be exercised
    # at a smaller production shape (a CPU box cannot compile 1024px in reasonable time).
    # A request at a different size simply pays its own compile, exactly as before.
    _WPX = int(os.environ.get("ANIMA_WARMUP_PX", "1024"))
    _sched = FlaxFlowMatchScheduler()
    _st8 = FlowMatchSchedulerState.create()
    _sr = np.linspace(1.0, 1.0 / 4, 4).astype(np.float32)
    _st8 = _sched.set_timesteps(_st8, sigmas=jnp.asarray(_sr))
    _ts = np.asarray(_st8.timesteps); _sg = np.asarray(_st8.sigmas[:-1])
    # Warm the transformer at the real 1024px latent/pad shapes so the first user
    # generation does not pay the XLA compile.
    _lat = jnp.asarray(np.random.default_rng(0).standard_normal(
        (1, 16, 1, _WPX // 8, _WPX // 8)).astype(np.float32))
    _pad = jnp.zeros((1, 1, _WPX, _WPX), dtype=jnp.float32)
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
    _tf_compiled.add((1, _WPX, _WPX))
    del _lat, _pad, _ctx, _pred
    gc.collect()
    log(f"warmup: transformer forward compiled+ran at {_WPX}px in {time.perf_counter()-t0:.1f}s; {hbm_stats()}")
    t0 = time.perf_counter()
    # fp32, not bf16: the real latents entering decode_latents are fp32 (they come
    # from an fp32 noise tensor), and decode_latents derives latents_mean/std from
    # the latent dtype, so a bf16 warmup compiles an executable the server never
    # calls -- the first real preview then recompiled the VAE (~50s, matching this
    # warmup's own compile time).
    _wz = jnp.zeros((1, 16, 1, _WPX // 8, _WPX // 8), dtype=jnp.float32)
    _wv = decode_latents(_wz)
    log(f"warmup: VAE decode compiled+ran at {_WPX}px in {time.perf_counter()-t0:.1f}s; out {_wv[0].shape}; {hbm_stats()}")
    _vae_compiled.add((_WPX, _WPX))
    del _wz, _wv
    gc.collect()
    log(f"warmup done; executables compiled; host RSS {rss_gb():.2f} GB")
except Exception:
    log("warmup raised:\n" + traceback.format_exc())
    raise

log(f"SERVER READY total {(time.perf_counter()-t_all)/60:.1f} min; watching {REQ_PATH}")
if XLA_CACHE:
    try:
        _n = len(os.listdir(XLA_CACHE))
    except Exception:
        _n = -1
    log(f"xla cache: {_n} entries at {XLA_CACHE}")
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

    # --- hot LoRA swap: same process, no restart, no recompile -----------------
    # transformer_forward takes the params as a jitted argument, so replacing the
    # weight arrays reuses the compiled executable (asserted in
    # anima_tpu_project/ops/xla_cache_bench.py, weight_swap -> reused_executable).
    if d.get("lora_swap") is not None:
        specs = [(str(it.get("path", "")), float(it.get("scale", 1.0)))
                 for it in (d.get("lora_swap") or [])
                 if str(it.get("path", "")).strip()]
        try:
            stats, secs, t_merge, t_put = swap_loras(specs)
            write_prog(stage="lora-swapped", step=0, gen_id=gen_id,
                       swap_seconds=round(secs, 3),
                       merge_seconds=round(t_merge, 3),
                       put_seconds=round(t_put, 3),
                       merged=stats.get("merged"), skipped=stats.get("skipped"),
                       loras=[{"path": p, "scale": s} for p, s in specs])
            json.dump({**d, "go": False, "status": "swapped",
                       "swap_seconds": round(secs, 3)}, open(REQ_PATH, "w"))
        except Exception:
            log("LoRA swap failed:\n" + traceback.format_exc())
            write_prog(stage="error", step=0, gen_id=gen_id,
                       error="lora swap failed")
            json.dump({**d, "go": False, "status": "error"}, open(REQ_PATH, "w"))
        last_mtime = os.path.getmtime(REQ_PATH)
        continue
    # Reset before the FIRST record of this request, so a client polling "starting" can
    # never see the previous request's preview_gaveup / preview_error.
    prev_ok = 0
    prev_fail = 0
    prev_fails_total = 0
    prev_err = ""
    prev_giveup = False
    _pv_expected = 0
    _vae_failed = set()
    # Clean up any interrupt left over from a previous, already-aborted run so it
    # cannot kill the next one.
    try:
        if os.path.exists(INTERRUPT_PATH):
            os.remove(INTERRUPT_PATH)
    except OSError:
        pass

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

        # --- in-prompt LoRA control: <lora:name:0.8> -------------------------
        # Semantics:
        #   * only the POSITIVE prompt is authoritative for which LoRAs are
        #     merged; a tag in the negative prompt is stripped and otherwise
        #     ignored, so it can never wipe the stack by accident;
        #   * a positive prompt with no tags leaves the weights alone, so the
        #     Gradio panel's applied LoRAs survive an ordinary prompt;
        #   * `<lora:none>` is the explicit way back to the bare base;
        #   * if tags resolve to nothing at all (every name is a typo), the
        #     current merge is KEPT and the names are reported -- clearing the
        #     stack because of a typo would be destructive and silent.
        # Tags are always stripped from both prompts: leaving them in would
        # feed the encoder literal "<lora:...>" tokens.
        _lora_note = {}
        _tags_pos = "<lora:" in (_p or "").lower()
        _tags_neg = "<lora:" in (_n or "").lower()
        if _tags_neg:
            _n = prompt_control.strip_lora_tags(_n)[0]
        if _tags_pos:
            _parsed = prompt_control.parse_lora_tags(_p, installed_lora_files())
            _p = _parsed["clean"]
            _want = _parsed["specs"]
            _have = [(str(a), float(b)) for a, b in (globals().get("_LORA_SPECS") or [])]
            _lora_note = {"lora_requested": [{"path": a, "scale": b} for a, b in _want],
                          "lora_unresolved": _parsed["unresolved"],
                          "lora_ignored_clip": _parsed["ignored_clip"],
                          "lora_clamped": _parsed["clamped"],
                          "lora_cleared": bool(_parsed.get("clear")),
                          "lora_swapped": False}
            if _parsed["unresolved"]:
                log("WARNING: prompt names LoRA(s) that are not installed: "
                    + ", ".join(_parsed["unresolved"])
                    + f" (searched {', '.join(LORA_DIRS)})")
            _skip = (not _want and not _parsed.get("clear") and _parsed["unresolved"])
            if _skip:
                log("prompt named only unknown LoRAs -> keeping the current merge")
                _lora_note["lora_skipped_reason"] = "no named LoRA is installed"
            elif prompt_control.lora_specs_key(_want) != prompt_control.lora_specs_key(_have):
                log(f"prompt requests {len(_want)} LoRA(s), merged set differs -> hot swap")
                try:
                    _sw_stats, _sw_secs, _sw_merge, _sw_put = swap_loras(_want)
                    _lora_note.update(lora_swapped=True,
                                      auto_swap_seconds=round(_sw_secs, 3),
                                      auto_swap_merged=_sw_stats.get("merged"),
                                      auto_swap_skipped=_sw_stats.get("skipped"))
                    log(f"in-prompt LoRA swap: {_sw_secs:.2f}s "
                        f"(merge {_sw_merge:.2f}s + put {_sw_put:.2f}s) "
                        f"merged={_sw_stats.get('merged')}")
                    if not _sw_stats.get("merged") and _want:
                        log("WARNING: swap reported 0 merged targets; the prompt's "
                            "LoRAs had no effect")
                except Exception:
                    log("in-prompt LoRA swap failed:\n" + traceback.format_exc())
                    _lora_note["auto_swap_error"] = "swap failed; using previous weights"
            else:
                log(f"prompt requests {len(_want)} LoRA(s), already merged -> no swap")
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
        prompt_states = max(len(pos_ctxs), len(neg_ctxs))
        prompt_switches = [round(b, 3) for b in
                           sorted(set(pos_plan["breakpoints"]) |
                                  set(neg_plan["breakpoints"])) if 0 < b < 1]
        write_prog(stage="text-encoded", step=0, steps=STEPS, seed=int(SEED), seeds=seeds,
                   gen_id=gen_id, batch_total=BATCH,
                   prompt_states=prompt_states, prompt_switches=prompt_switches)

        sched = FlaxFlowMatchScheduler()
        st = FlowMatchSchedulerState.create()
        sigmas_raw = np.linspace(1.0, 1.0 / STEPS, STEPS).astype(np.float32)
        st = sched.set_timesteps(st, sigmas=jnp.asarray(sigmas_raw))
        timesteps = np.asarray(st.timesteps)
        sigmas = np.asarray(st.sigmas[:-1])

        # SnapRef: opt-in reference adapter. Absent/off keeps the plain path untouched.
        _snap = d.get("snapref") or {}
        snap_info = {}
        if _snap and _snap.get("enabled", True) and (_snap.get("images") or []):
            write_prog(stage="snapref", step=0, steps=STEPS, seed=int(SEED), seeds=seeds,
                       gen_id=gen_id, batch_total=BATCH, note="encoding reference")
            snap_info = prepare_snapref(_snap, (H, W))
            _snap["_sizes"] = snap_info["sizes"]
            _snap["_weights"] = snap_info["weights"]
            snap_info.update(adapter=snap_info.get("adapter"), mode=snap_info.get("mode"),
                             strength=float(_snap.get("strength", 0.0)),
                             fidelity=float(_snap.get("fidelity", 1.0)),
                             ref_in_uncond=bool(_snap.get("ref_in_uncond", False)),
                             ref_guidance_on=bool(_snap.get("ref_guidance_on", False)),
                             ref_guidance=float(_snap.get("ref_guidance", 4.5)))
        else:
            _SNAPREF_STATE["kv"] = None

        # All BATCH images share every transformer call: latents (B,16,1,h,w), context
        # (B,512,1024), pad (B,1,H,W). The padding mask is concatenated channel-wise inside the
        # transformer, so its batch dim must equal the latents' batch dim.
        latents = jnp.asarray(np.concatenate(
            [np.random.default_rng(s).standard_normal((1, 16, 1, H // 8, W // 8)).astype(np.float32)
             for s in seeds], axis=0))
        pad = jnp.zeros((BATCH, 1, H, W), dtype=jnp.float32)
        # prompt_states / prompt_switches belong in `common` so every later record
        # (denoise, done, error) carries them. Writing them only on the intermediate
        # "text-encoded" record left the FINAL record with None, which is exactly the
        # record the UI and the archive sidecar read.
        common = dict(steps=STEPS, seed=int(SEED), seeds=seeds, gen_id=gen_id,
                      batch_total=BATCH, prompt_states=prompt_states,
                      prompt_switches=prompt_switches, **_lora_note)
        if snap_info:
            common["snapref"] = snap_info
        if (BATCH, H, W) not in _tf_compiled:
            log(f"new transformer shape batch={BATCH} {H}x{W}: XLA compile on first step")
            write_prog(stage="compiling", step=0, note=f"transformer {BATCH}x{W}x{H}",
                       **common, **_pv_state())
        # An interrupt is a file the client creates; checked every step so a stop
        # lands within ~1 step (~0.5 s) instead of finishing the whole run.
        try:
            if os.path.exists(INTERRUPT_PATH):
                os.remove(INTERRUPT_PATH)
        except OSError:
            pass

        d0 = time.perf_counter()
        prev_total = 0.0
        first_s = 0.0
        _interrupted = False
        for i in range(STEPS):
            if os.path.exists(INTERRUPT_PATH):
                try:
                    os.remove(INTERRUPT_PATH)
                except OSError:
                    pass
                _interrupted = True
                log(f"INTERRUPT at step {i}/{STEPS}; aborting before decode")
                write_prog(stage="interrupted", step=i, steps=STEPS,
                           **common, **_pv_state())
                break
            context = pos_ctxs[pos_map[i]]
            neg_context = neg_ctxs[neg_map[i]]
            if snap_info:
                pred = snapref_cfg_prediction(latents, jnp.asarray(np.float32(timesteps[i])),
                                              context, neg_context, pad, GUIDANCE, _snap)
            else:
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
                    save_u8(make_grid_u8(_imgs), SNAP_PATH)
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
        if _interrupted:
            # Skip the VAE decode entirely: the run was cancelled, so producing
            # (and writing) an image would be misleading.
            clear_snapref_runtime()
            json.dump({**d, "go": False, "status": "interrupted"}, open(REQ_PATH, "w"))
            last_mtime = os.path.getmtime(REQ_PATH)
            continue
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
        clear_snapref_runtime()
    except Exception as _err:
        clear_snapref_runtime()
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
if os.name == "nt" and len(code) > 24000:
    # Windows caps a command line at 32767 characters, and this program is larger than that.
    # Write it to a file and run the file: Colab (POSIX) keeps the proven -c path.
    _main = os.path.join(os.environ.get("ANIMA_CONTENT_ROOT", "/content"), "anima_server_main.py")
    with open(_main, "w", encoding="utf-8") as _fh:
        _fh.write(code)
    r = subprocess.Popen([py, "-u", _main], cwd=repo, env=env)
else:
    r = subprocess.Popen([py, "-u", "-c", code], cwd=repo, env=env)
print("server pid", r.pid, flush=True)
