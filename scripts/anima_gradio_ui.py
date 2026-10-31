"""Gradio UI for the Anima-Aesthetic TPU server.

Run on the Colab CPU host (NOT inside the TPU venv): the TPU server owns the
device, and this UI only writes /content/anima_request.json and polls
/content/anima_progress.json + /content/anima_snap.png.

Features
  * True batching: N images share every denoise step on the TPU (image k uses seed+k).
  * Prompt weighting: (tag:1.3). Parsed and applied on the server, where the tokenizer is.
  * Resolution from an aspect ratio (W:H, editable, with a live shape preview), locked to
    0.5 / 1.0 / 1.5 MP, or "Custom size" for typing pixel dimensions directly.
  * The live preview is a real VAE decode of the in-progress latents (a grid when batching).
"""
import io
import json
import math
import os
import shutil
import time
import uuid
import warnings
from collections import OrderedDict

# Gradio wraps every queued call in MatplotlibBackendMananger(), which imports
# matplotlib. Under Colab the kernel exports
# MPLBACKEND=module://matplotlib_inline.backend_inline, which matplotlib rejects
# with "not a valid value for backend" -- so EVERY queued call, Generate
# included, dies with `event: error` before our code runs. Force a real backend
# here, before anything can import matplotlib, so every launch path is safe.
os.environ["MPLBACKEND"] = "agg"

# gradio's routes.py still references starlette's deprecated
# HTTP_422_UNPROCESSABLE_ENTITY status constant, so every queue_join (every
# poll tick AND every API call) prints a StarletteDeprecationWarning to stderr.
# It is gradio's bug, not ours — silence exactly that message, nothing else.
# No category filter on purpose: starlette changed the base of
# StarletteDeprecationWarning from DeprecationWarning (older) to FutureWarning
# (newer), so a category-restricted filter silently stops matching after a
# starlette upgrade. The message alone is unambiguous.
warnings.filterwarnings("ignore",
                        message=r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.*")

from PIL import Image

# Handshake files, shared with the server (see anima_aesthetic_server.py).
REQ_PATH = "/content/anima_request.json"
PROG_PATH = "/content/anima_progress.json"
# LoRA support reuses the CLI boot path rather than a parallel one.
LORA_DIR = "/content/anima_lora"
JOB_PATH = "/content/anima_job.json"
LAUNCH_LOG = "/content/server_launch.log"
VENV_PY = "/content/maxdiffusion/.venv/bin/python"
BOOT_SCRIPT = "/content/anima_job.py"
# Where a running boot announces itself, so a UI restart (watchdog) does not
# forget that the sampler is mid-reboot.
REBOOT_PATH = "/content/anima_reboot.json"
# Live state of a link-triggered LoRA download (written by scripts/add_lora.py).
DL_STATE_PATH = "/content/anima_download.json"
# Repo root, so the detached downloader runs from the pulled checkout.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADD_LORA_SCRIPT = os.path.join(REPO_ROOT, "scripts", "add_lora.py")
SNAP_PATH = "/content/anima_snap.png"
INTERRUPT_PATH = "/content/anima_interrupt"   # create to cancel the running generation

# True while this process has a generation in flight. The 1 Hz progress poll
# stays silent otherwise, so a freshly loaded page does not resurrect the last
# run's progress (the server's progress file survives restarts) and an idle page
# stops emitting an update chip every second.
_RUN_ACTIVE = {"flag": False, "until": 0.0}
_STALE_AFTER_S = 300.0
FINAL_PATH = "/content/anima_perstep.png"

# Where finished images are archived (Drive, usually). Empty disables archiving.
SAVE_ENV = "ANIMA_SAVE_DIR"
# The server resolves a negative seed into a real one and reports it back.
RANDOM_SEED = -1
MAX_BATCH = 8  # keep in sync with the server's clamp

FERN_PROMPT = "masterpiece, best quality, 1girl, fern (sousou no frieren), sousou no frieren, @izei1337, purple hair, black robe, lips, sidelocks, very long hair, puffy sleeves, white dress, butterfly on hand, eyelashes, simple background, closed mouth, mage staff, arm at side, straight hair, blush, solo, purple eyes, chromatic aberration, purple pupils, looking at viewer, hand up, standing, bug, robe, black background, signature, bright pupils, black coat, coat, long sleeves, blue butterfly, upturned eyes, wide sleeves, blunt bangs, from above, dress, blunt ends, long hair, purple butterfly, butterfly, tsurime, half updo"
FERN_NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"

# ---------------------------------------------------------------------------
# Resolution: aspect ratio + megapixel lock
# ---------------------------------------------------------------------------
MIB = 1024 * 1024  # "1 MP" = 1024x1024 px, the model's native size (and the warmed shape)
MP_TARGETS = OrderedDict([
    ("0.5 MP", 0.5 * MIB),
    ("1.0 MP", 1.0 * MIB),
    ("1.5 MP", 1.5 * MIB),
])
CUSTOM = "Custom size"
MP_CHOICES = list(MP_TARGETS) + [CUSTOM]
MULT = 16                      # VAE stride 8 x patchify stride 2
MIN_PIX, MAX_PIX = 256, 2048
ASPECT_PRESETS = OrderedDict([
    ("1:1", (1, 1)), ("4:3", (4, 3)), ("3:2", (3, 2)), ("16:9", (16, 9)), ("21:9", (21, 9)),
    ("3:4", (3, 4)), ("2:3", (2, 3)), ("9:16", (9, 16)),
])


def _pos(x, default=1.0):
    """A positive float from whatever the UI handed us (None/NaN/0/negative -> default)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if v > 0 and math.isfinite(v) else float(default)


def _snap(v):
    """Round to a multiple of MULT and clamp to [MIN_PIX, MAX_PIX]."""
    return int(min(max(int(round(v / MULT)) * MULT, MIN_PIX), MAX_PIX))


def size_for_aspect(aspect_w, aspect_h, target_px):
    """(W, H), both multiples of 16, whose ratio is ~aspect_w:aspect_h and area ~target_px.

    Exact area and exact ratio are not both reachable on a 16-px grid, so this searches the
    neighbouring grid points and keeps the one with the smallest combined log-error.
    """
    ar = _pos(aspect_w) / _pos(aspect_h)
    base = round(math.sqrt(target_px * ar) / MULT)
    best = None
    for dw in range(-3, 4):
        w = _snap((base + dw) * MULT)
        hb = round(target_px / w / MULT)
        for dh in (-1, 0, 1):
            h = _snap((hb + dh) * MULT)
            err = abs(math.log(w * h / target_px)) + abs(math.log((w / h) / ar))
            if best is None or err < best[0]:
                best = (err, w, h)
    return best[1], best[2]


def resolve_size(aspect_w, aspect_h, mp_label, width, height):
    """Final (W, H) for a request, from the current control values."""
    if mp_label in MP_TARGETS:
        return size_for_aspect(aspect_w, aspect_h, MP_TARGETS[mp_label])
    return _snap(_pos(width, 1024)), _snap(_pos(height, 1024))


def _fmt(v):
    v = float(v)
    return str(int(v)) if v == int(v) else f"{v:g}"


def preview_html(aspect_w, aspect_h, mp_label, width, height):
    """A to-scale box of the chosen shape plus the exact pixel size and megapixels."""
    w, h = resolve_size(aspect_w, aspect_h, mp_label, width, height)
    if mp_label in MP_TARGETS:
        ratio = f"{_fmt(_pos(aspect_w))}:{_fmt(_pos(aspect_h))}"
    else:
        g = math.gcd(w, h)
        ratio = f"{w // g}:{h // g}"
    scale = min(170 / w, 110 / h)
    bw, bh = max(int(w * scale), 8), max(int(h * scale), 8)
    lock = f"locked to {mp_label}" if mp_label in MP_TARGETS else "custom size"
    return (
        '<div style="display:flex;align-items:center;gap:18px;min-height:130px">'
        '<div style="width:180px;height:120px;display:flex;align-items:center;justify-content:center;flex:none">'
        f'<div style="width:{bw}px;height:{bh}px;box-sizing:border-box;'
        'border:2px solid var(--color-accent, #7c3aed);border-radius:4px;'
        'background:color-mix(in srgb, var(--color-accent, #7c3aed) 14%, transparent);'
        'display:flex;align-items:center;justify-content:center;'
        f'font:600 13px sans-serif;color:var(--body-text-color, inherit)">{ratio}</div></div>'
        '<div style="font:14px/1.5 sans-serif;color:var(--body-text-color, inherit)">'
        f'<div style="font-size:20px;font-weight:600">{w} &times; {h}</div>'
        f'<div>{w * h / MIB:.2f} MP &middot; {lock}</div>'
        f'<div style="opacity:.7">width &times; height, multiples of {MULT}</div></div></div>'
    )


# ---------------------------------------------------------------------------
# Server handshake helpers
# ---------------------------------------------------------------------------
def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _read_prog():
    return _read_json(PROG_PATH, {"stage": "waiting"})


def _read_img(path, newer_than=None):
    """Load an image; with newer_than, ignore files older than that time (stale leftovers)."""
    try:
        if newer_than is not None and os.path.getmtime(path) < newer_than:
            return None
        with open(path, "rb") as f:
            data = f.read()
        if not data:
            return None
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:
        return None


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def save_dir():
    """Archive directory for finished images, or "" when archiving is disabled."""
    return os.environ.get(SAVE_ENV, "").strip()


def _unique(path):
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def save_outputs(prog, req):
    """Archive every image of a finished run (plus the grid) with a JSON sidecar each.

    Each individual image's sidecar has its own seed and batch_count 1, so it can be
    regenerated on its own. Returns a status string, or None when archiving is disabled.
    """
    d = save_dir()
    if not d:
        return None
    try:
        os.makedirs(d, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        images = prog.get("images") or [FINAL_PATH]
        seeds = prog.get("seeds") or [prog.get("seed", req.get("seed"))]
        meta = {k: v for k, v in req.items() if k not in ("go", "out", "gen_id")}
        tag = f"steps{req['steps']}_{req['width']}x{req['height']}"
        n = 0
        for k, src in enumerate(images):
            seed = seeds[k] if k < len(seeds) else seeds[0]
            dst = _unique(os.path.join(d, f"anima_{ts}_seed{seed}_{tag}.png"))
            shutil.copy2(src, dst)
            with open(os.path.splitext(dst)[0] + ".json", "w") as f:
                json.dump({**meta, "seed": seed, "batch_count": 1}, f, indent=2, sort_keys=True)
            n += 1
        if len(images) > 1 and prog.get("grid"):
            dst = _unique(os.path.join(d, f"anima_{ts}_grid{len(images)}_seed{seeds[0]}_{tag}.png"))
            shutil.copy2(prog["grid"], dst)
            with open(os.path.splitext(dst)[0] + ".json", "w") as f:
                json.dump({**meta, "seeds": seeds}, f, indent=2, sort_keys=True)
            n += 1
        return f"saved {n} file(s) to {d}"
    except Exception as e:  # noqa: BLE001
        return f"<archive failed: {type(e).__name__}: {e}>"


def _preview_fragment(prog, preview_every):
    """The live-preview state as one status fragment, or None when there is nothing to say.

    A failed preview used to be invisible to the client: the server logged a single line and
    carried on, so the UI showed an empty preview behind a healthy status line and a normal
    "done". The server now publishes preview_ok / preview_fails / preview_error /
    preview_gaveup, and this is what makes them readable.
    """
    err = prog.get("preview_error")
    if err and prog.get("preview_gaveup"):
        return (f"previews disabled after failures: {err} "
                f"(traceback in /content/anima_server.log)")
    if err:
        return f"previews failing: {err} (see /content/anima_server.log)"
    if prog.get("preview_gaveup"):
        return "previews disabled (see /content/anima_server.log)"
    if preview_every is None:
        return None            # the caller did not report a preview setting: say nothing
    if not preview_every:
        return "previews off"
    ok = prog.get("preview_ok")
    if ok:
        total = prog.get("preview_total")
        return f"previews {ok}/{total}" if total else f"previews {ok}"
    return None


PROMPT_STYLES = ("comfy", "a1111", "safe")

SYNTAX_HELP = """\
**Weights** — `(tag:1.3)` strengthens, `(tag:0.7)` weakens, groups nest and multiply. \
`(tag)` alone is `(tag:1.1)`. Range clamps to 0–3.

**Schedules** — switch mid-denoise on a fraction of the steps:
`[before:after:0.5]` · `[cat:0.5]` (empty before) · `[before:during:after:0.3,0.7]` \
range · `[a|b:0.25]` alternating · `[SEQ:a:0.2:b:0.5:c]` chain · `[a:b:TAG]` with the \
Tags box. All nest, e.g. `a [red:[blue::0.7]:0.5] cat`.

**Combinators** — `AND` and `CAT` join prompts; `AVG(0.3)` averages them. \
`BREAK` is a no-op here: prompt-control ignores it for T5 encoders, which is what we use.

**Literals** — `#` starts a comment; escape with `\\#`, `\\:`, `\\\\`. \
Text inside `"double quotes"` never triggers operators.

**Styles** — `comfy`: bare `(x)` is emphasis (ComfyUI behaviour), so a danbooru tag \
like `fern (sousou no frieren)` must be escaped as `\\(...\\)`. `a1111`: additionally \
`[x]` weakens and `{x}` strengthens. `safe`: only `(x:weight)` is syntax, so plain \
parentheses stay literal — pick this to keep old prompts byte-identical.
"""


def _status_line(prog, fallback_steps, preview_every=None):
    stage = prog.get("stage", "?")
    step = prog.get("step", 0)
    steps = prog.get("steps") or fallback_steps
    batch = prog.get("batch_total") or 1
    bits = [f"{stage} {step}/{steps}"]
    if batch > 1:
        bits.append(f"batch of {batch}")
    seeds = prog.get("seeds") or []
    if len(seeds) > 1:
        bits.append(f"seeds {seeds[0]}..{seeds[-1]}")
    elif prog.get("seed") is not None:
        bits.append(f"seed {prog['seed']}")
    if prog.get("ms_per_step"):
        bits.append(f"{prog['ms_per_step']} ms/step")
    if prog.get("rss_gb"):
        bits.append(f"RSS {prog['rss_gb']} GB")
    # Prompt scheduling: surface how many states the prompt resolves to and where it
    # switches, so a schedule that silently never fires is visible rather than assumed.
    states = prog.get("prompt_states")
    if states and states > 1:
        sw = prog.get("prompt_switches") or []
        bits.append(f"prompt: {states} states"
                    + (f", switches at {', '.join(str(x) for x in sw)}" if sw else ""))
    if stage == "compiling":
        bits = [f"compiling {prog.get('note', 'new shape')} (one-time per size/batch, can take 1-2 min)"]
    if stage == "done":
        bits = [f"done in {prog.get('elapsed_s', '?')}s",
                f"denoise {prog.get('denoise_s', '?')}s",
                f"decode {prog.get('decode_s', '?')}s"]
        if batch > 1:
            bits.insert(0, f"{batch} images")
        hbm = prog.get("hbm") or {}
        if hbm:
            bits.append(f"HBM peak {hbm.get('peak_gb', '?')}/{hbm.get('limit_gb', '?')} GB")
    if stage == "error":
        bits = [f"error: {prog.get('error') or 'see /content/anima_server.log'}"]
    line = " | ".join(str(b) for b in bits)
    # Appended after the stage overrides so the preview state survives the "done" and "error"
    # rewrites -- that is where a silent preview failure was previously invisible.
    frag = _preview_fragment(prog, preview_every)
    return f"{line} | {frag}" if frag else line


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _upd(**kw):
    """gr.update without importing gradio at module scope (keeps start_gen drivable)."""
    import gradio as gr
    return gr.update(**kw)


def btn_reset():
    """Generate's fully-enabled resting state, for after a reboot completes."""
    return _upd(visible=True, interactive=True)


def interrupt_gen():
    """Ask the server to cancel the running generation."""
    try:
        with open(INTERRUPT_PATH, "w") as f:
            f.write("1")
        return "interrupt requested — stopping after the current step"
    except Exception as e:
        return f"could not request interrupt: {type(e).__name__}: {e}"


# ------------------------------------------------------------------------- LoRA
# LoRAs are merged at BOOT on the CPU, before the params go to the TPU (see the
# ANIMA_LORAS block in anima_aesthetic_server.py). There is no runtime swap, so
# "Apply" rewrites /content/anima_job.json and re-runs anima_job.py — the exact
# boot path the CLI uses. Cost: ~4-5 minutes during which the sampler is DOWN.

_REBOOT = {"active": False, "t0": 0.0, "proc": None, "names": [], "scale": 1.0}
# Last finished download already advertised in the dropdown, so the choices are
# not rewritten on every timer tick.
_DL_HANDLED = {"key": None}
REBOOT_TIMEOUT_S = 1500.0
REBOOT_HINT = "about 4-5 minutes"


def _installed_loras():
    """Files available to merge. Anything else in the dropdown is rejected."""
    try:
        return sorted(f for f in os.listdir(LORA_DIR) if f.endswith(".safetensors"))
    except Exception:
        return []


def _loaded_loras():
    """What the RUNNING server merged, read back from its own boot log.

    None  -> no boot log to judge from.
    []    -> the server booted with no LoRA.
    """
    try:
        with open(LAUNCH_LOG, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return None
    for line in reversed(lines):
        if "no LoRA configured" in line:
            return []
        # Split on the marker, not the first ":" — every line starts with a
        # "[HH:MM:SS]" stamp, so a naive split returns "09:02] merging ...".
        if "merging " in line and "LoRA(s):" in line:
            tail = line.split("LoRA(s):", 1)[1]
            return [t.strip() for t in tail.split(",") if t.strip()]
    return None


def _lora_state_html(note="", pct=None):
    """State panel for the LoRA accordion.

    pct is the LIVE BOOT progress (0-100) and only exists while a reboot is
    running: once the boot finishes, the panel drops back to the plain
    "In the running server: ..." line and the bar disappears.
    """
    import html as _html

    loaded = _loaded_loras()
    if loaded is None:
        body = "unknown (no boot log yet)"
    elif not loaded:
        body = "none — base model"
    else:
        body = ", ".join(f"<code>{_html.escape(x)}</code>" for x in loaded)
    out = f"<div class='lora-state'>In the running server: {body}</div>"

    # A link download in flight (or just finished) belongs in this panel: it is
    # the thing the user is waiting on, and it needs no server restart.
    dl = _read_json(DL_STATE_PATH) or {}
    dst = dl.get("status")
    if dst in ("resolving", "downloading", "validating"):
        name = _html.escape(str(dl.get("name") or "…"))
        if dst == "downloading":
            pct_dl = int(dl.get("pct") or 0)
            mb = (dl.get("done") or 0) / 1e6
            tot = (dl.get("total") or 0) / 1e6
            label = (f"downloading {name} — {pct_dl}% "
                     f"({mb:.0f}/{tot:.0f} MB)" if tot
                     else f"downloading {name} — {mb:.0f} MB")
        elif dst == "validating":
            label = f"validating {name}…"
        else:
            label = f"resolving {name}…"
        out += (f"<div class='lora-dl'>↓ {label}</div>"
                f"<div class='lora-bar'><div class='lora-bar-fill' "
                f"style='width:{int(dl.get('pct') or 0)}%'></div></div>")
    elif dst == "error":
        out += (f"<div class='lora-dl lora-err'>✗ "
                f"{_html.escape(str(dl.get('message') or 'download failed'))}</div>")
    elif dst == "ok":
        # A short confirmation, not a permanent one: the file is in the list now.
        if time.time() - float(dl.get("t") or 0) < 120:
            n = dl.get("tensors")
            extra = f" ({n} tensors)" if n else ""
            out += (f"<div class='lora-dl'>✓ "
                    f"{_html.escape(str(dl.get('name') or 'file'))}{extra} ready — "
                    f"select it above, then Apply</div>")

    if pct is not None:
        pct = max(0, min(100, int(pct)))
        out += (f"<div class='lora-bar'>"
                f"<div class='lora-bar-fill' style='width:{pct}%'></div></div>")
    if note:
        out += f"<div class='lora-note'>{_html.escape(note)}</div>"
    return out


def _boot_progress():
    """Turn the sampler's boot log into (pct, label, last line).

    do_boot() truncates the log before every boot, so the file holds exactly
    one boot's worth of lines. A missing/empty file just means the boot has
    not stamped anything yet — not an error.
    """
    try:
        with open(LAUNCH_LOG, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return 0, "starting…", ""
    import re as _re

    pct, label, detail = 0, "starting…", ""
    for line in lines:
        l = line.strip()
        if not l:
            continue
        detail = l[:140]
        if "Qwen3 TPU model ready" in l:
            pct, label = 8, "text encoder ready"
        elif "adapter block" in l and "/6" in l:
            m = _re.search(r"block (\d+)/6", l)
            if m:
                cur = int(m.group(1))
                pct, label = 10 + int(14 * cur / 6), f"conditioner {cur}/6"
        elif "transformer block" in l and "/28" in l:
            m = _re.search(r"block (\d+)/28", l)
            if m:
                cur = int(m.group(1))
                pct, label = 24 + int(36 * cur / 28), f"transformer {cur}/28"
        elif "merging " in l and "LoRA(s):" in l:
            pct, label = 60, "merging LoRA(s)…"
        elif "LoRA merged:" in l:
            m = _re.search(r"in ([\d.]+)s", l)
            pct, label = 75, (f"LoRA merged in {m.group(1)}s" if m
                              else "LoRA merged")
        elif "vae converted+merged" in l:
            pct, label = 80, "VAE ready"
        elif "warmup: transformer forward" in l:
            pct, label = 86, "compiling transformer exec (1024px)"
        elif "warmup: VAE decode" in l:
            pct, label = 92, "compiling VAE decode"
        elif "SERVER READY" in l:
            pct, label = 100, "ready"
    return pct, label, detail


def _resume_reboot():
    """Adopt a boot that outlived the UI process.

    The watchdog restarts the UI when the tunnel dies, and that wipes the
    in-memory reboot state. If a reboot was mid-flight, pick it back up from
    the intent file so the tracker (and the Generate lock) come back.
    """
    try:
        d = json.load(open(REBOOT_PATH))
        age = time.time() - float(d["t0"])
        if 0 < age < REBOOT_TIMEOUT_S:
            _REBOOT.update(active=True, t0=float(d["t0"]), proc=None,
                           names=list(d.get("names") or []),
                           scale=float(d.get("scale", 1.0)))
    except Exception:
        pass


def apply_loras(names, scale):
    """Write the boot job and restart the sampler with the chosen LoRAs.

    Detached on purpose: the boot takes minutes and the tunnel buffers long
    streaming responses, so the button returns at once and poll_progress follows
    the reboot. Returns (state html, generate-button update).
    """
    import subprocess

    import gradio as gr

    if _RUN_ACTIVE["flag"]:
        return (_lora_state_html("a generation is running — interrupt it first"),
                gr.update())
    if _REBOOT["active"]:
        return _lora_state_html("a reboot is already in progress"), gr.update()

    allowed = set(_installed_loras())
    chosen = [n for n in (names or []) if n in allowed]
    rejected = [n for n in (names or []) if n not in allowed]
    if rejected:
        return (_lora_state_html(f"ignored unknown LoRA(s): {', '.join(rejected)}"),
                gr.update())
    if not chosen and names:
        return _lora_state_html("none of those files exist in " + LORA_DIR), gr.update()

    scale = float(scale if scale is not None else 1.0)
    payload = {"step": "boot"}
    if chosen:
        payload["loras"] = [{"name": n, "scale": scale} for n in chosen]
    else:
        # No "loras" and no "name": anima_job boots the base model.
        payload["label"] = "base"

    _atomic_json(JOB_PATH, payload)
    env = {**os.environ, "MPLBACKEND": "agg", "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.Popen(
            [VENV_PY, "-u", BOOT_SCRIPT], cwd="/content", env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,   # outlive this click
        )
    except Exception as e:
        return (_lora_state_html(f"could not start the reboot: {type(e).__name__}: {e}"),
                gr.update())

    _REBOOT.update(active=True, t0=time.time(), proc=proc,
                   names=chosen, scale=scale)
    _atomic_json(REBOOT_PATH, {"t0": _REBOOT["t0"], "names": chosen,
                               "scale": scale})
    label = ", ".join(f"{n}@{scale:g}" for n in chosen) if chosen else "no LoRA"
    return (_lora_state_html(f"rebooting with {label}. The server is down for "
                             f"{REBOOT_HINT}; leave this page open."),
            gr.update(interactive=False))


def _reboot_state(stage=None):
    """Follow the detached boot. Returns (status, text)."""
    if not _REBOOT["active"]:
        return "idle", ""
    elapsed = time.time() - _REBOOT["t0"]
    proc = _REBOOT["proc"]
    rc = proc.poll() if proc is not None else None

    try:
        mtime = os.path.getmtime(PROG_PATH)
    except OSError:
        mtime = 0.0
    if stage is None:
        stage = _read_prog().get("stage")

    # do_boot() deletes the progress record before it boots, so a "ready" with a
    # fresh mtime can only come from THIS boot.
    if stage == "ready" and mtime >= _REBOOT["t0"] - 1:
        _REBOOT["active"] = False
        _clear_reboot_intent()
        label = ", ".join(_REBOOT["names"]) or "no LoRA"
        return "done", f"server ready with {label} after {elapsed:.0f}s"
    if rc not in (None, 0):
        _REBOOT["active"] = False
        _clear_reboot_intent()
        return "error", f"boot exited with code {rc} — see {LAUNCH_LOG}"
    try:
        with open(LAUNCH_LOG, encoding="utf-8", errors="replace") as f:
            tail = f.readlines()[-400:]
    except Exception:
        tail = []
    if any("Traceback (most recent call last)" in l for l in tail):
        _REBOOT["active"] = False
        _clear_reboot_intent()
        return "error", f"boot raised an exception — see {LAUNCH_LOG}"
    if elapsed > REBOOT_TIMEOUT_S:
        _REBOOT["active"] = False
        _clear_reboot_intent()
        return "error", f"boot timed out after {elapsed:.0f}s"
    return "rebooting", (f"rebooting… {elapsed:.0f}s "
                         f"(sampler at {stage or 'loading'})")


def _clear_reboot_intent():
    try:
        os.remove(REBOOT_PATH)
    except OSError:
        pass


def add_lora_from_link(url):
    """Download + validate a LoRA from a HuggingFace/CivitAI link.

    Deliberately does NOT touch the sampler: fetching a file is safe while the
    TPU is busy generating, and merging it is a separate, explicit Apply.
    The downloader is a detached process because a 600 MB fetch through the
    tunnel would look dead as a blocking click; poll_progress follows
    /content/anima_download.json instead.
    """
    import subprocess

    import gradio as gr

    u = (url or "").strip()
    if not u:
        return _lora_state_html("paste a HuggingFace or CivitAI link first")

    cur = _read_json(DL_STATE_PATH) or {}
    if cur.get("status") in ("resolving", "downloading", "validating"):
        return _lora_state_html(f"already fetching {cur.get('name') or 'a file'} "
                                f"— let it finish")
    if not os.path.exists(ADD_LORA_SCRIPT):
        return _lora_state_html(f"downloader missing on this checkout: {ADD_LORA_SCRIPT}")
    if not os.path.exists(VENV_PY):
        return _lora_state_html(f"venv python missing: {VENV_PY}")

    # Drop the previous result first, so a stale "ok" can never be read as the
    # outcome of this attempt.
    try:
        os.remove(DL_STATE_PATH)
    except OSError:
        pass
    _atomic_json(DL_STATE_PATH, {"status": "resolving", "url": u, "t": time.time()})

    try:
        subprocess.Popen([VENV_PY, "-u", ADD_LORA_SCRIPT, u], cwd=REPO_ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except Exception as e:
        _atomic_json(DL_STATE_PATH, {"status": "error", "url": u,
                                     "message": f"{type(e).__name__}: {e}",
                                     "t": time.time()})
        return _lora_state_html(f"could not start the download: "
                                f"{type(e).__name__}: {e}")
    return _lora_state_html("fetching — progress appears here")


def poll_progress():
    """Surface generation progress over ordinary requests instead of SSE.

    The browser's Server-Sent-Events stream gets buffered by the tunnel in front
    of us: measured from the page itself, zero events arrived during a 21.5 s
    run and all 32 landed at once when it finished, so the live preview only
    ever appeared at the end. Polling sidesteps that completely, because each
    tick is a plain request/response that no proxy has a reason to hold back.

    Returns: result, status, progress, generate button, interrupt button,
        LoRA state panel.
    """
    import gradio as gr

    prog = _read_prog()
    stage = prog.get("stage")

    # A finished link download should appear in the picker without the user
    # having to press Refresh. Track what we already advertised so the choices
    # are not rewritten on every tick.
    dl = _read_json(DL_STATE_PATH) or {}
    dl_key = (dl.get("name"), dl.get("t")) if dl.get("status") == "ok" else None
    if dl_key and dl_key != _DL_HANDLED["key"]:
        _DL_HANDLED["key"] = dl_key
        dl_update = gr.update(choices=_installed_loras())
    else:
        dl_update = gr.update()

    # A reboot owns the sampler: report it and hold Generate inert, so a click
    # cannot land in the gap where nothing is reading the request file yet.
    rb_status, rb_text = _reboot_state(stage)
    if rb_status == "rebooting":
        pct, label, _detail = _boot_progress()
        note = f"{pct}% — {label}"
        return (gr.update(), f"{rb_text} | {note}",
                progress_bar_html(pct, f"rebooting {pct}%"),
                gr.update(visible=True, interactive=False), gr.update(visible=False),
                _lora_state_html(f"{label}  ({pct}%)", pct=pct), dl_update)
    if rb_status == "done":
        # 100% reached: the tracker and its bar clear themselves, leaving only
        # the plain "In the running server: ..." line.
        return (gr.update(), "reboot complete — server ready",
                progress_bar_html(100, "ready"), btn_reset(),
                gr.update(visible=False), _lora_state_html(), dl_update)

    # Generate comes back only once the sampler has confirmed it is up.
    btn_idle = gr.update(visible=True, interactive=True)

    # Safety net: a client that vanished mid-run leaves the flag set, so also
    # give up on a record the server has not touched in a long time.
    if _RUN_ACTIVE["flag"] and time.time() - float(prog.get("t") or 0) > _STALE_AFTER_S:
        _RUN_ACTIVE["flag"] = False
    if not stage or not _RUN_ACTIVE["flag"]:
        return (gr.update(), gr.update(), gr.update(), btn_idle, gr.update(),
                _lora_state_html(), dl_update)

    steps = int(prog.get("steps") or 30)
    step = int(prog.get("step") or 0)
    busy = stage in ("starting", "text_encode", "denoise", "decode", "preview")
    # The progress record does not carry preview_every, so read the requested
    # value from the request file: otherwise the status line claims "previews
    # off" on runs that are plainly producing previews.
    requested = _read_json(REQ_PATH) or {}
    status = _status_line(prog, steps, int(requested.get("preview_every") or 0))
    pct = 100 if stage == "done" else int(round(100.0 * step / max(1, steps)))

    # Only touch the image while a run is in flight: once it ends, the final
    # grid from start_gen must stay on screen rather than being replaced by the
    # last intermediate preview.
    snap = gr.update()
    if busy:
        loaded = _read_img(SNAP_PATH)
        if loaded is not None:
            snap = loaded

    return (snap, status, progress_bar_html(pct, f"step {step}/{steps}"),
            gr.update(visible=not busy, interactive=not busy), gr.update(visible=busy),
            _lora_state_html(), dl_update)


def start_gen(prompt, neg, aspect_w, aspect_h, mp_label, width, height, steps, seed,
              guidance, preview_every, batch_count, prompt_style="comfy", tags=""):
    W, H = resolve_size(aspect_w, aspect_h, mp_label, width, height)
    batch = max(1, min(MAX_BATCH, int(batch_count or 1)))
    req = {
        # Weighting and scheduling syntax ((tag:1.3), [a:b:0.5], AND/CAT/AVG) is sent
        # through untouched; the server parses it with the prompt-control grammar.
        "prompt": prompt,
        "negative_prompt": neg,
        "height": H, "width": W,
        "steps": int(steps), "guidance": float(guidance),
        "seed": int(seed) if seed is not None else RANDOM_SEED,
        "preview_every": int(preview_every or 0),
        "batch_count": batch,
        "prompt_style": (prompt_style or "comfy"),
        "out": FINAL_PATH, "go": True,
        "gen_id": uuid.uuid4().hex,
    }
    _tags = [t.strip() for t in (tags or "").replace(",", " ").split() if t.strip()]
    if _tags:
        req["tags"] = _tags
    t_submit = time.time()
    _RUN_ACTIVE["flag"] = True
    _atomic_json(REQ_PATH, req)
    # Generous: the first run at a new size/batch pays an XLA compile (1-2 min).
    deadline = t_submit + 900 + 1800 * batch
    seen = None
    # outputs: result image, status, progress bar, generate btn, interrupt btn, gallery
    BUSY = (_upd(visible=False), _upd(visible=True))
    IDLE = (_upd(visible=True), _upd(visible=False))
    yield (_upd(value=None, visible=True), "queued...",
           progress_bar_html(0, "queued"), *BUSY,
           _upd(visible=False, value=[]))
    while time.time() < deadline:
        prog = _read_prog()
        mine = prog.get("gen_id") == req["gen_id"]
        if not mine:
            # Another run's leftover record: never show its status or its images.
            status = "waiting for the server to pick up the request"
            stage = None
        else:
            stage = prog.get("stage")
            status = _status_line(prog, req["steps"], req["preview_every"])
        snap = _read_img(SNAP_PATH, newer_than=t_submit - 1) if mine else None
        done_steps = int(prog.get("step") or 0) if mine else 0
        pct = int(round(100.0 * done_steps / max(1, int(req["steps"]))))
        if stage == "done":
            images = [im for im in (_read_img(p) for p in prog.get("images", [])) if im is not None]
            seeds = prog.get("seeds") or []
            gallery_items = [(im, f"seed {seeds[k]}" if k < len(seeds)
                              else f"image {k + 1}")
                             for k, im in enumerate(images)]
            grid = _read_img(prog.get("grid") or FINAL_PATH)
            saved = save_outputs(prog, req)
            if saved:
                status += f" | {saved}"
            _RUN_ACTIVE["flag"] = False
            if int(req.get("batch_count") or 1) > 1:
                # The two occupy the same panel: a batch shows the individual
                # images with their seeds instead of the grid.
                yield (_upd(visible=False), status,
                       progress_bar_html(100, "done"), *IDLE,
                       _upd(visible=True, value=gallery_items))
            else:
                yield (grid or snap, status, progress_bar_html(100, "done"),
                       *IDLE, _upd(visible=False, value=[]))
            return
        if stage == "interrupted":
            _RUN_ACTIVE["flag"] = False
            yield snap, status, progress_bar_html(pct, f"step {done_steps}/{int(req['steps'])}"), *IDLE, []
            return
        if stage == "error":
            _RUN_ACTIVE["flag"] = False
            yield None, status, progress_bar_html(0, "error"), *IDLE, []
            return
        marker = (prog.get("t"), prog.get("step"), stage, prog.get("preview_step"))
        if marker != seen:
            seen = marker
            yield snap, status, progress_bar_html(pct, f"step {done_steps}/{int(req['steps'])}"), *BUSY, []
        time.sleep(0.5)
    # Render the last progress record instead of dumping a raw dict: a timeout line full of
    # Python dict syntax was unreadable, and hid the one field that explains a stall (stage).
    _to = _read_prog()
    _to_status = _status_line(_to, req["steps"], req["preview_every"])
    _RUN_ACTIVE["flag"] = False
    yield (None, f"TIMEOUT after {int(time.time() - t_submit)}s: {_to_status}",
           progress_bar_html(0, "timeout"), *IDLE, [])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def disable_compression_for_sse():
    """Stop Gradio's Brotli middleware from buffering the queue event stream.

    Gradio installs BrotliMiddleware, which is built on Starlette's
    GZipResponder. Compression middleware must buffer chunks before it can
    compress them, so the /queue/data SSE stream never flushes -- the browser
    gets every progress event in one burst when the run ends. That is exactly
    what makes the live preview look frozen: the transfer is fine, the events
    are simply held back. Behind a Cloudflare tunnel it is worse, because the
    tunnel buffers the already-buffered response again.

    Fix: for the SSE path only, drop `accept-encoding` so the compressor never
    engages. Everything else stays compressed.
    """
    import gradio.brotli_middleware as _bm
    import gradio.routes as _routes

    base = getattr(_bm, "BrotliMiddleware", None)
    if base is None or getattr(base, "_sse_safe", False):
        return False

    class SSESafeBrotli(base):
        _sse_safe = True

        async def __call__(self, scope, receive, send):
            if scope.get("type") == "http" and "/queue/data" in (scope.get("path") or ""):
                scope = dict(scope)
                headers = [(k, v) for (k, v) in scope.get("headers", [])
                           if k.lower() != b"accept-encoding"]
                headers.append((b"accept-encoding", b"identity"))
                scope["headers"] = headers
            await super().__call__(scope, receive, send)

    _bm.BrotliMiddleware = SSESafeBrotli
    # routes.py imported the name at import time, so patch its reference too.
    if getattr(_routes, "BrotliMiddleware", None) is base:
        _routes.BrotliMiddleware = SSESafeBrotli
    return True


RESPONSIVE_CSS = """
/* Custom progress bar: the Slider's reset knob and numeric range read as a
   control the user can drag, which it is not. */
.anima-progress {
  position: relative;
  width: 100%;
  height: 24px;
  border-radius: 6px;
  background: var(--border-color-primary, #3a3a3a);
  overflow: hidden;
}
.anima-progress-fill {
  height: 100%;
  width: 0%;
  background: var(--color-accent, #f97316);
  transition: width 0.3s ease;
}
.anima-progress-text {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  color: var(--body-text-color, #eee);
  text-shadow: 0 0 3px rgba(0, 0, 0, 0.6);
}

/* Boot tracker: shown only while a LoRA reboot is running, then it clears. */
.lora-state { font-size: 12px; margin-bottom: 4px; }
.lora-bar {
  position: relative;
  width: 100%;
  height: 10px;
  border-radius: 5px;
  background: var(--border-color-primary, #3a3a3a);
  overflow: hidden;
  margin: 4px 0 2px 0;
}
.lora-bar-fill {
  height: 100%;
  width: 0%;
  background: var(--color-accent, #f97316);
  transition: width 0.4s ease;
}
.lora-note {
  font-size: 12px;
  color: var(--body-text-color-subdued, #bbb);
  margin-top: 2px;
  font-variant-numeric: tabular-nums;
}
.lora-dl {
  font-size: 12px;
  margin-top: 4px;
  color: var(--body-text-color-subdued, #bbb);
  font-variant-numeric: tabular-nums;
}
.lora-dl.lora-err { color: var(--error-text-color, #f87171); }

#anima-result img, #anima-result video {
  max-height: 72vh !important;
  object-fit: contain;
}
#anima-actions { flex-wrap: wrap; }
#anima-controls .gr-row, #anima-controls .flex { flex-wrap: wrap !important; }

/* Narrow: collapse to a single column, preserving DOM order
   (result, then prompt, then the button). */
@media (max-width: 900px) {
  #anima-main { flex-direction: column !important; }
  #anima-main > * { min-width: 100% !important; }
  #anima-actions { flex-direction: column !important; }
  #anima-actions > * { width: 100% !important; }
  #anima-result img, #anima-result video { max-height: 42vh !important; }
  #anima-controls .gr-row, #anima-controls .flex { flex-direction: column !important; }
  #anima-advanced .gr-row > *, #anima-advanced .flex > * { min-width: 100% !important; }
  .gradio-container { padding: 8px !important; }
  #anima-status textarea { font-size: 0.85rem; }
}
"""


def progress_bar_html(pct, text=""):
    """Render the progress element. gr.HTML rather than a Slider: a slider
    implies a draggable control and renders a reset knob we do not want."""
    pct = max(0, min(100, int(pct)))
    safe = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<div class="anima-progress">'
            f'<div class="anima-progress-fill" style="width:{pct}%"></div>'
            f'<span class="anima-progress-text">{safe}</span></div>')


def build():
    import gradio as gr

    # A boot may have outlived a UI restart (watchdog): adopt it so the tracker
    # and the Generate lock survive the process swap.
    _resume_reboot()

    w0, h0 = size_for_aspect(1, 1, MP_TARGETS["1.0 MP"])

    with gr.Blocks(title="Anima Aesthetic TPU", fill_height=True) as demo:
        # Wide screens get two columns (result | controls). Below 900px the CSS
        # stacks them, and because the result comes first in the DOM that
        # collapse reads as result -> prompt -> button on a phone.
        with gr.Row(elem_id="anima-main"):
            with gr.Column(scale=3):
                result = gr.Image(label="Result", height=560, elem_id="anima-result")
                # Individual images only mean something for a batch, so this stays
                # hidden until a batch actually produces them. It shares this column
                # with the result, so showing one hides the other.
                gallery = gr.Gallery(label="Individual images (with seeds)", columns=3,
                                     height=560, object_fit="contain", visible=False,
                                     elem_id="anima-gallery")
                # Below BOTH panels: the bar describes whichever one is showing, and
                # stays in place when the gallery takes the result's spot.
                progress = gr.HTML(progress_bar_html(0, "idle"),
                                   elem_id="anima-progress")
            with gr.Column(scale=2, elem_id="anima-controls"):
                # Prompt and negative belong together: one group, no accordion.
                prompt = gr.Textbox(value=FERN_PROMPT, lines=5, label="Prompt",
                                    placeholder="prompt — supports (weights:1.3) "
                                                "and [before:after:0.5] schedules")
                neg = gr.Textbox(value=FERN_NEG, lines=2,
                                 label="Negative prompt (weights work here too)")
                with gr.Row(elem_id="anima-actions"):
                    btn = gr.Button("Generate", variant="primary", size="lg", scale=4)
                    stop = gr.Button("Interrupt", variant="stop", size="lg", scale=4,
                                     visible=False)
                status = gr.Textbox(label="Status", interactive=False, show_label=False,
                                    placeholder="idle", elem_id="anima-status")

                with gr.Row():
                    steps = gr.Slider(4, 50, value=30, step=1, label="Steps")
                    batch_count = gr.Slider(1, MAX_BATCH, value=1, step=1,
                                            label="Batch count",
                                            info="Images generated together, step by "
                                                 "step. Image k uses seed + k.")

                gr.Markdown("### Resolution")
                with gr.Row():
                    aspect_w = gr.Number(value=1, label="Aspect W", minimum=0.1)
                    aspect_h = gr.Number(value=1, label="Aspect H", minimum=0.1)
                    preset = gr.Dropdown(list(ASPECT_PRESETS), value=None, label="Preset",
                                         info="Quick-fill the ratio")
                    swap = gr.Button("Swap W/H", size="sm")
                mp_target = gr.Radio(MP_CHOICES, value="1.0 MP", label="Megapixels",
                                     info="1 MP = 1024 x 1024 px. Custom size lets you "
                                          "type the pixels.")
                with gr.Row():
                    width = gr.Number(value=w0, label="Width (px)", precision=0,
                                      interactive=False)
                    height = gr.Number(value=h0, label="Height (px)", precision=0,
                                       interactive=False)
                shape_preview = gr.HTML(preview_html(1, 1, "1.0 MP", w0, h0))

                with gr.Row():
                    guidance = gr.Slider(1.0, 8.0, value=4.0, step=0.5,
                                         label="CFG guidance")
                    seed = gr.Number(value=RANDOM_SEED, label="Seed (-1 = random)",
                                     precision=0)
                    preview_every = gr.Slider(0, 10, value=5, step=1,
                                              label="VAE preview every N steps (0 = off)",
                                              info="Each preview decodes every image "
                                                   "in the batch.")
                with gr.Row():
                    prompt_style = gr.Dropdown(list(PROMPT_STYLES), value="comfy",
                                               label="Prompt style",
                                               info="comfy = ComfyUI core rules. a1111 "
                                                    "adds [x] / {x}. safe = only "
                                                    "(x:weight), plain parens stay literal.")
                    tags = gr.Textbox(value="", label="Tags (for [a:b:TAG] selection)",
                                      info="Space or comma separated, e.g. SECOND_PASS")
                with gr.Accordion("LoRA (merged at boot — Apply restarts the server)",
                                  open=False):
                    lora_pick = gr.Dropdown(choices=_installed_loras(), value=[],
                                            multiselect=True, label="LoRAs to merge",
                                            info="Everything in /content/anima_lora. "
                                                 "All selected files merge in one pass.")
                    with gr.Row():
                        lora_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05,
                                               label="Scale (applies to all selected)")
                        lora_refresh = gr.Button("Refresh list", size="sm")
                    # Fetching a file is safe while the TPU generates: only the
                    # merge needs a restart, so this is a separate button.
                    lora_url = gr.Textbox(
                        label="Add from link (HuggingFace or CivitAI)",
                        placeholder="https://huggingface.co/user/repo/blob/main/x.safetensors "
                                    "or https://civitai.com/models/12345",
                        info="Downloads and validates into /content/anima_lora. "
                             "A CivitAI model page becomes its API download route; "
                             "a bare HuggingFace repo picks a .safetensors file.")
                    lora_add = gr.Button("Download & validate", variant="secondary")
                    lora_apply = gr.Button("Apply & restart server", variant="primary")
                    lora_state = gr.HTML(_lora_state_html())
                    gr.Markdown("A restart takes ~4-5 minutes: the sampler reloads the "
                                "base model and merges the LoRAs on the CPU before it "
                                "can generate again. Generate stays disabled until it "
                                "reports ready.")
                with gr.Accordion("Prompt syntax reference", open=False):
                    gr.Markdown(SYNTAX_HELP)

        gr.Markdown(f"Seed `-1` picks a random seed and reports the value used. "
                    f"Finished images are archived to `{save_dir() or '(disabled)'}`.")

        # ---- resolution wiring -------------------------------------------------------
        def apply_size(aw, ah, mp, w, h):
            """Recompute pixels + preview. Locked: W/H are derived and read-only.
            Custom: W/H are typed by the user, so their values are never rewritten here."""
            html = preview_html(aw, ah, mp, w, h)
            if mp in MP_TARGETS:
                W, H = resolve_size(aw, ah, mp, w, h)
                return (gr.update(value=W, interactive=False),
                        gr.update(value=H, interactive=False), html)
            return gr.update(interactive=True), gr.update(interactive=True), html

        size_inputs = [aspect_w, aspect_h, mp_target, width, height]
        size_outputs = [width, height, shape_preview]
        gr.on(triggers=[aspect_w.change, aspect_h.change, mp_target.change,
                        width.change, height.change],
              fn=apply_size, inputs=size_inputs, outputs=size_outputs, show_progress="hidden")

        def pick_preset(name):
            if name in ASPECT_PRESETS:
                a, b = ASPECT_PRESETS[name]
                return a, b
            return gr.update(), gr.update()

        preset.change(pick_preset, inputs=preset, outputs=[aspect_w, aspect_h],
                      show_progress="hidden")
        swap.click(lambda a, b: (b, a), inputs=[aspect_w, aspect_h],
                   outputs=[aspect_w, aspect_h], show_progress="hidden")

        # ---- persisted settings (browser localStorage) --------------------------
        # Gr.BrowserState keeps a JSON blob in localStorage, so the prompt and the
        # rest of the form survive a reload or a fresh browser session.
        settings = gr.BrowserState(default_value=None, storage_key="anima_settings_v1")
        SETTING_KEYS = ["prompt", "neg", "prompt_style", "tags", "steps", "guidance",
                        "seed", "preview_every", "batch_count", "aspect_w", "aspect_h",
                        "mp_target", "width", "height"]
        setting_inputs = [prompt, neg, prompt_style, tags, steps, guidance, seed,
                          preview_every, batch_count, aspect_w, aspect_h, mp_target,
                          width, height]

        def save_settings(*vals):
            return {k: v for k, v in zip(SETTING_KEYS, vals)}

        def restore_settings(saved):
            if not isinstance(saved, dict):
                return [gr.update() for _ in setting_inputs]
            return [gr.update() if saved.get(name) is None else gr.update(value=saved[name])
                    for name in SETTING_KEYS]

        gr.on(triggers=[c.change for c in setting_inputs],
              fn=save_settings, inputs=setting_inputs, outputs=[settings],
              show_progress="hidden")
        demo.load(restore_settings, inputs=[settings], outputs=setting_inputs,
                  show_progress="hidden")

        btn.click(start_gen,
                  inputs=[prompt, neg, aspect_w, aspect_h, mp_target, width, height, steps,
                          seed, guidance, preview_every, batch_count, prompt_style, tags],
                  outputs=[result, status, progress, btn, stop, gallery],
                  concurrency_limit=1)  # one TPU: never interleave two requests
        # Progress is POLLED, not streamed. The tunnel in front of this app
        # buffers the SSE stream, so streamed updates all arrive at once when the
        # run is already over (measured: 0 events during a 21.5 s run, 32 at the
        # end). A timer tick is a plain request/response, which nothing buffers.
        timer = gr.Timer(1.0, active=True)
        timer.tick(poll_progress, inputs=[],
                   outputs=[result, status, progress, btn, stop, lora_state, lora_pick],
                   queue=False, show_progress="hidden")
        # A plain queued call, so it can run while start_gen is still streaming.
        stop.click(interrupt_gen, inputs=[], outputs=[status], queue=False)

        # ---- LoRA wiring ------------------------------------------------------
        lora_refresh.click(lambda: gr.update(choices=_installed_loras()),
                           inputs=[], outputs=[lora_pick], show_progress="hidden")
        # Fetching is independent of the sampler, so it is not gated on a run;
        # the handler refuses only if another download is already in flight.
        lora_add.click(add_lora_from_link, inputs=[lora_url], outputs=[lora_state],
                       concurrency_limit=1)
        # concurrency_limit=1 keeps a double click from launching two boots.
        lora_apply.click(apply_loras, inputs=[lora_pick, lora_scale],
                         outputs=[lora_state, btn], concurrency_limit=1)
        demo.load(lambda: gr.update(choices=_installed_loras()), inputs=[],
                  outputs=[lora_pick], show_progress="hidden")
    demo.queue()
    return demo


if __name__ == "__main__":
    _port = int(os.environ.get("ANIMA_UI_PORT", "7860"))
    print("sse-safe compression patch:", disable_compression_for_sse(), flush=True)
    demo = build()
    demo.launch(
        server_name=os.environ.get("ANIMA_UI_HOST", "0.0.0.0"),
        server_port=_port,
        share=False,
        prevent_thread_lock=True,
        show_error=True,
        css=RESPONSIVE_CSS,   # Gradio 6 moved css/head here from the Blocks ctor
    )
    print(f"ANIMA_UI_UP on port {_port}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
