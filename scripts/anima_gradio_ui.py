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
from collections import OrderedDict

from PIL import Image

# Handshake files, shared with the server (see anima_aesthetic_server.py).
REQ_PATH = "/content/anima_request.json"
PROG_PATH = "/content/anima_progress.json"
SNAP_PATH = "/content/anima_snap.png"
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
def _read_prog():
    try:
        with open(PROG_PATH) as f:
            return json.load(f)
    except Exception:
        return {"stage": "waiting"}


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
def start_gen(prompt, neg, aspect_w, aspect_h, mp_label, width, height, steps, seed,
              guidance, preview_every, batch_count):
    W, H = resolve_size(aspect_w, aspect_h, mp_label, width, height)
    batch = max(1, min(MAX_BATCH, int(batch_count or 1)))
    req = {
        # Weight syntax ((tag:1.3)) is sent through untouched; the server parses it.
        "prompt": prompt,
        "negative_prompt": neg,
        "height": H, "width": W,
        "steps": int(steps), "guidance": float(guidance),
        "seed": int(seed) if seed is not None else RANDOM_SEED,
        "preview_every": int(preview_every or 0),
        "batch_count": batch,
        "out": FINAL_PATH, "go": True,
        "gen_id": uuid.uuid4().hex,
    }
    t_submit = time.time()
    _atomic_json(REQ_PATH, req)
    # Generous: the first run at a new size/batch pays an XLA compile (1-2 min).
    deadline = t_submit + 900 + 1800 * batch
    seen = None
    # outputs: final grid, gallery of individual images, live preview, status
    yield None, [], None, "queued..."
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
        if stage == "done":
            images = [im for im in (_read_img(p) for p in prog.get("images", [])) if im is not None]
            seeds = prog.get("seeds") or []
            gallery = [(im, f"seed {seeds[k]}" if k < len(seeds) else f"image {k + 1}")
                       for k, im in enumerate(images)]
            grid = _read_img(prog.get("grid") or FINAL_PATH)
            saved = save_outputs(prog, req)
            if saved:
                status += f" | {saved}"
            yield grid, gallery, snap, status
            return
        if stage == "error":
            yield None, [], snap, status
            return
        marker = (prog.get("t"), prog.get("step"), stage, prog.get("preview_step"))
        if marker != seen:
            seen = marker
            yield None, [], snap, status
        time.sleep(0.5)
    # Render the last progress record instead of dumping a raw dict: a timeout line full of
    # Python dict syntax was unreadable, and hid the one field that explains a stall (stage).
    _to = _read_prog()
    _to_status = _status_line(_to, req["steps"], req["preview_every"])
    yield (None, [], _read_img(SNAP_PATH, newer_than=t_submit - 1),
           f"TIMEOUT after {int(time.time() - t_submit)}s: {_to_status}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build():
    import gradio as gr

    w0, h0 = size_for_aspect(1, 1, MP_TARGETS["1.0 MP"])

    with gr.Blocks(title="Anima Aesthetic TPU") as demo:
        gr.Markdown("# Anima Aesthetic v1.1 (TPU v5e)\n"
                    "Transformer + conditioner: Anima Aesthetic v1.1 | "
                    "Qwen3 text encoder, tokenizers, VAE: Anima-Base-v1.0-Diffusers | "
                    "BF16_BF16_F32\n\n"
                    f"Seed `-1` picks a random seed and reports the value used. "
                    f"Finished images are archived to `{save_dir() or '(disabled)'}`.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(value=FERN_PROMPT, lines=8, label="Prompt",
                                    info="Weights: (tag:1.3) strengthens, (tag:0.7) weakens; 1.0 is "
                                         "neutral, range 0-3. Groups nest. Plain parentheses such as "
                                         "fern (sousou no frieren) are left alone; inside a weighted "
                                         "group write \\( and \\) for literal ones.")
                neg = gr.Textbox(value=FERN_NEG, lines=2, label="Negative prompt (weights work here too)")

                gr.Markdown("### Resolution")
                with gr.Row():
                    aspect_w = gr.Number(value=1, label="Aspect W", minimum=0.1)
                    aspect_h = gr.Number(value=1, label="Aspect H", minimum=0.1)
                    preset = gr.Dropdown(list(ASPECT_PRESETS), value=None, label="Preset",
                                         info="Quick-fill the ratio")
                    swap = gr.Button("Swap W/H", size="sm")
                mp_target = gr.Radio(MP_CHOICES, value="1.0 MP", label="Megapixels",
                                     info="1 MP = 1024 x 1024 px. Custom size lets you type the pixels.")
                with gr.Row():
                    width = gr.Number(value=w0, label="Width (px)", precision=0, interactive=False)
                    height = gr.Number(value=h0, label="Height (px)", precision=0, interactive=False)
                shape_preview = gr.HTML(preview_html(1, 1, "1.0 MP", w0, h0))

                with gr.Row():
                    steps = gr.Slider(4, 50, value=30, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="CFG guidance")
                with gr.Row():
                    seed = gr.Number(value=RANDOM_SEED, label="Seed (-1 = random)", precision=0)
                    preview_every = gr.Slider(0, 10, value=5, step=1,
                                              label="VAE preview every N steps (0 = off)",
                                              info="Each preview decodes every image in the batch.")
                batch_count = gr.Slider(1, MAX_BATCH, value=1, step=1, label="Batch count",
                                        info="Images generated together, step by step. "
                                             "Image k uses seed + k.")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False)
                final_img = gr.Image(label="Final image / grid", height=520)
                gallery = gr.Gallery(label="Individual images (with seeds)", columns=4,
                                     height="auto", object_fit="contain")
                snap_img = gr.Image(label="Live preview (VAE-decoded latent)", height=520)

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

        btn.click(start_gen,
                  inputs=[prompt, neg, aspect_w, aspect_h, mp_target, width, height, steps,
                          seed, guidance, preview_every, batch_count],
                  outputs=[final_img, gallery, snap_img, status],
                  concurrency_limit=1)  # one TPU: never interleave two requests
    demo.queue()
    return demo


if __name__ == "__main__":
    _port = int(os.environ.get("ANIMA_UI_PORT", "7860"))
    demo = build()
    demo.launch(
        server_name=os.environ.get("ANIMA_UI_HOST", "0.0.0.0"),
        server_port=_port,
        share=False,
        prevent_thread_lock=True,
        show_error=True,
    )
    print(f"ANIMA_UI_UP on port {_port}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
