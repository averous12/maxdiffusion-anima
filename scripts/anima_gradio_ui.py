"""Gradio UI for the Anima-Aesthetic TPU server.

Run on the Colab CPU host (NOT inside the TPU venv): the TPU server owns the
device, and this UI only writes /content/anima_request.json and polls
/content/anima_progress.json + /content/anima_snap.png.

The live preview is a real VAE decode of the in-progress latent (not a
channel-mean snapshot), so it costs one warm decode (~0.5 s) per preview.
"""
import io
import json
import os
import shutil
import time
import uuid

from PIL import Image

# Handshake files, shared with the server (see anima_aesthetic_server.py).
#
# The request carries a `gen_id`; the server stamps that id into EVERY progress record it
# writes for that request. The UI only accepts a `done` record whose gen_id matches the
# request it submitted. Without that handshake the UI cannot tell its own result from the
# previous run's: it polled the progress file, found the leftover stage="done" record, and
# returned the previous image instantly -- the run looked like a no-op.
REQ_PATH = "/content/anima_request.json"
PROG_PATH = "/content/anima_progress.json"
SNAP_PATH = "/content/anima_snap.png"
FINAL_PATH = "/content/anima_perstep.png"

# Where finished images are archived (Drive, usually). Empty disables archiving.
SAVE_ENV = "ANIMA_SAVE_DIR"
# The server resolves a negative seed into a real one and reports it back.
RANDOM_SEED = -1

FERN_PROMPT = "masterpiece, best quality, 1girl, fern (sousou no frieren), sousou no frieren, @izei1337, purple hair, black robe, lips, sidelocks, feet out of frame, very long hair, puffy sleeves, white dress, butterfly on hand, eyelashes, simple background, closed mouth, mage staff, arm at side, straight hair, blush, solo, purple eyes, chromatic aberration, purple pupils, looking at viewer, hand up, standing, bug, robe, black background, signature, bright pupils, black coat, coat, long sleeves, blue butterfly, upturned eyes, wide sleeves, blunt bangs, from above, dress, blunt ends, long hair, purple butterfly, butterfly, tsurime, half updo"
FERN_NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"


def _read_prog():
    try:
        with open(PROG_PATH) as f:
            return json.load(f)
    except Exception:
        return {"stage": "waiting"}


def _read_img(path):
    """Return the image at `path` as a PIL image, or None if it is not there yet.

    Gradio's Image postprocess accepts np.ndarray | PIL.Image.Image | str | Path |
    None -- NOT raw bytes. Returning f.read() raised
        ValueError: Cannot process this value as an Image, it is of type: <class 'bytes'>
    on every yield, so the UI displayed nothing even though the server was generating
    correctly. A PIL image is used rather than the path itself so the live preview
    cannot be served from a stale path-keyed cache: the snapshot is rewritten in place
    at the same path on every preview, and each poll must show the current contents.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
        if not data:
            return None
        img = Image.open(io.BytesIO(data))
        img.load()  # decode now; nothing should depend on the buffer afterwards
        return img
    except Exception:
        return None


def _atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def save_dir():
    """Archive directory for finished images, or "" when archiving is disabled.

    Read at call time (not import time) so a caller can set the env var after import.
    """
    return os.environ.get(SAVE_ENV, "").strip()


def save_outputs(src_path, req):
    """Copy a finished image into the archive dir under a collision-free name.

    The name carries the timestamp, the resolved seed, the step count and the size, so
    a directory of runs stays self-describing and no two runs overwrite each other. A
    .json sidecar records the full request so any image can be reproduced exactly.
    Returns the destination path, or None when archiving is disabled.
    """
    d = save_dir()
    if not d:
        return None
    try:
        os.makedirs(d, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = (f"anima_{ts}_seed{req.get('seed')}_steps{req.get('steps')}"
                f"_{req.get('width')}x{req.get('height')}")
        dst = os.path.join(d, base + ".png")
        n = 1
        while os.path.exists(dst):          # same second, same params -> suffix
            dst = os.path.join(d, f"{base}_{n}.png")
            n += 1
        shutil.copy2(src_path, dst)
        with open(os.path.splitext(dst)[0] + ".json", "w") as f:
            json.dump(req, f, indent=2, sort_keys=True)
        return dst
    except Exception as e:  # noqa: BLE001 - never let archiving break a generation
        return f"<archive failed: {type(e).__name__}: {e}>"


def _status_line(prog, fallback_steps):
    stage = prog.get("stage", "?")
    step = prog.get("step", 0)
    steps = prog.get("steps") or fallback_steps
    bits = [f"{stage} {step}/{steps}"]
    if prog.get("seed") is not None:
        bits.append(f"seed {prog['seed']}")
    if prog.get("ms_per_step"):
        bits.append(f"{prog['ms_per_step']} ms/step")
    if prog.get("rss_gb"):
        bits.append(f"RSS {prog['rss_gb']} GB")
    if stage == "done":
        bits = [f"done in {prog.get('elapsed_s', '?')}s",
                f"denoise {prog.get('denoise_s', '?')}s",
                f"decode {prog.get('decode_s', '?')}s"]
        hbm = prog.get("hbm") or {}
        if hbm:
            bits.append(f"HBM peak {hbm.get('peak_gb', '?')}/{hbm.get('limit_gb', '?')} GB")
    if stage == "error":
        bits = ["error - see /content/anima_server.log"]
    return " | ".join(str(b) for b in bits)


def start_gen(prompt, neg, height, width, steps, seed, guidance, preview_every):
    # A cleared gr.Number yields None; fall back to a random seed rather than raising.
    req = {"prompt": prompt, "negative_prompt": neg,
           "height": int(height), "width": int(width),
           "steps": int(steps), "guidance": float(guidance),
           "seed": int(seed) if seed is not None else RANDOM_SEED,
           "preview_every": int(preview_every or 0),
           "out": FINAL_PATH, "go": True,
           # A unique id so the UI can tell its own progress records from a previous run's.
           # The server echoes this id in every record it writes for this request.
           "gen_id": uuid.uuid4().hex}
    _atomic_json(REQ_PATH, req)
    deadline = time.time() + 1800
    seen = None
    yield _read_img(FINAL_PATH), _read_img(SNAP_PATH), "queued..."
    while time.time() < deadline:
        prog = _read_prog()
        snap = _read_img(SNAP_PATH)
        final = _read_img(FINAL_PATH)
        status = _status_line(prog, req["steps"])
        stage = prog.get("stage")
        # A record from an earlier run (or from boot) must not end this generation.
        # The image is only read once the matching `done` record is visible, and the
        # server writes the PNG before that record, so the file is complete.
        if prog.get("gen_id") != req["gen_id"]:
            stage = None
            status = "waiting for the server to pick up the request | " + status
        if stage == "done":
            # The server resolves a negative seed into a real one; archive under that
            # value so the filename identifies the run that actually happened.
            saved = save_outputs(FINAL_PATH, {**req, "seed": prog.get("seed", req["seed"])})
            if saved:
                status += f" | saved {saved}"
            yield final, snap, status
            return
        if stage == "error":
            yield final, snap, status
            return
        marker = (prog.get('t'), prog.get('step'), stage, prog.get('preview_step'))
        if marker != seen:
            seen = marker
            yield final, snap, status
        time.sleep(0.5)
    prog = _read_prog()
    yield _read_img(FINAL_PATH), _read_img(SNAP_PATH), f"timeout at {prog}"


def build():
    import gradio as gr
    with gr.Blocks(title="Anima Aesthetic TPU") as demo:
        gr.Markdown("# Anima Aesthetic v1.1 (TPU v5e)\n"
                    "Transformer + conditioner: Anima Aesthetic v1.1 | "
                    "Qwen3 text encoder, tokenizers, VAE: Anima-Base-v1.0-Diffusers | "
                    "BF16_BF16_F32\n\n"
                    f"Seed `-1` picks a random seed and reports the value used. "
                    f"Finished images are archived to `{save_dir() or '(disabled)'}`.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(value=FERN_PROMPT, lines=8, label="Prompt")
                neg = gr.Textbox(value=FERN_NEG, lines=2, label="Negative prompt")
                with gr.Row():
                    height = gr.Number(value=1024, label="Height")
                    width = gr.Number(value=1024, label="Width")
                with gr.Row():
                    steps = gr.Slider(4, 50, value=30, step=1, label="Steps")
                    guidance = gr.Slider(1.0, 8.0, value=4.0, step=0.5, label="CFG guidance")
                with gr.Row():
                    seed = gr.Number(value=RANDOM_SEED, label="Seed (-1 = random)")
                    preview_every = gr.Slider(0, 10, value=5, step=1,
                                             label="VAE preview every N steps (0 = off)")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False)
                final_img = gr.Image(label="Final image", height=520)
                snap_img = gr.Image(label="Live preview (VAE-decoded latent)", height=520)
        btn.click(start_gen,
                  inputs=[prompt, neg, height, width, steps, seed, guidance, preview_every],
                  outputs=[final_img, snap_img, status])
    demo.queue()  # streaming intermediate yields requires the Gradio queue
    return demo


if __name__ == "__main__":
    # No Gradio tunnel: `anima_cloudflared.py` publishes this port instead.
    # Runs as a long-lived subprocess, so block after launch() returns.
    import os
    import time
    _port = int(os.environ.get("ANIMA_UI_PORT", "7860"))
    demo = build()
    demo.queue()  # streaming intermediate yields requires the Gradio queue
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
