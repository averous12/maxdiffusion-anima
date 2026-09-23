"""Gradio UI for the Anima-Aesthetic TPU server.

Run on the Colab CPU host (NOT inside the TPU venv): the TPU server owns the
device, and this UI only writes /content/anima_request.json and polls
/content/anima_progress.json + /content/anima_snap.png.

The live preview is a real VAE decode of the in-progress latent (not a
channel-mean snapshot), so it costs one warm decode (~0.5 s) per preview.
"""
import json
import os
import time

REQ_PATH = "/content/anima_request.json"
PROG_PATH = "/content/anima_progress.json"
SNAP_PATH = "/content/anima_snap.png"
FINAL_PATH = "/content/anima_perstep.png"

FERN_PROMPT = "masterpiece, best quality, 1girl, fern (sousou no frieren), sousou no frieren, @izei1337, purple hair, black robe, lips, sidelocks, feet out of frame, very long hair, puffy sleeves, white dress, butterfly on hand, eyelashes, simple background, closed mouth, mage staff, arm at side, straight hair, blush, solo, purple eyes, chromatic aberration, purple pupils, looking at viewer, hand up, standing, bug, robe, black background, signature, bright pupils, black coat, coat, long sleeves, blue butterfly, upturned eyes, wide sleeves, blunt bangs, from above, dress, blunt ends, long hair, purple butterfly, butterfly, tsurime, half updo"
FERN_NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"


def _read_prog():
    try:
        with open(PROG_PATH) as f:
            return json.load(f)
    except Exception:
        return {"stage": "waiting"}


def _read_img(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _status_line(prog, fallback_steps):
    stage = prog.get("stage", "?")
    step = prog.get("step", 0)
    steps = prog.get("steps") or fallback_steps
    bits = [f"{stage} {step}/{steps}"]
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
    req = {"prompt": prompt, "negative_prompt": neg,
           "height": int(height), "width": int(width),
           "steps": int(steps), "guidance": float(guidance), "seed": int(seed),
           "preview_every": int(preview_every),
           "out": FINAL_PATH, "go": True}
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
        if stage == "done":
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
                    "BF16_BF16_F32")
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
                    seed = gr.Number(value=0, label="Seed")
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
    return demo


if __name__ == "__main__":
    build().launch(share=True)
