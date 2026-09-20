"""Gradio UI for the Anima-Aesthetic TPU server.

Run on the Colab CPU host (NOT inside the TPU venv): the TPU server owns the
device, and this UI only writes /content/anima_request.json and polls
/content/anima_progress.json + /content/anima_snap.png.
"""
import json
import os
import time

REQ_PATH = "/content/anima_request.json"
PROG_PATH = "/content/anima_progress.json"
SNAP_PATH = "/content/anima_snap.png"

FERN_PROMPT = "masterpiece, best quality, 1girl, fern (sousou no frieren), sousou no frieren, @izei1337, purple hair, black robe, lips, sidelocks, feet out of frame, very long hair, puffy sleeves, white dress, butterfly on hand, eyelashes, simple background, closed mouth, mage staff, arm at side, straight hair, blush, solo, purple eyes, chromatic aberration, purple pupils, looking at viewer, hand up, standing, bug, robe, black background, signature, bright pupils, black coat, coat, long sleeves, blue butterfly, upturned eyes, wide sleeves, blunt bangs, from above, dress, blunt ends, long hair, purple butterfly, butterfly, tsurime, half updo"
FERN_NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"


def _read_prog():
    try:
        return json.load(open(PROG_PATH))
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


def start_gen(prompt, neg, height, width, steps, seed, preview_every):
    req = {"prompt": prompt, "negative_prompt": neg,
           "height": int(height), "width": int(width),
           "steps": int(steps), "guidance": 4.0, "seed": int(seed),
           "preview_every": int(preview_every),
           "out": "/content/anima_perstep.png", "go": True}
    _atomic_json(REQ_PATH, req)
    deadline = time.time() + 900
    seen = None
    while time.time() < deadline:
        prog = _read_prog()
        snap = _read_img(SNAP_PATH)
        final = _read_img("/content/anima_perstep.png")
        status = f"{prog.get('stage', '?')} {prog.get('step', 0)}/{prog.get('steps', req['steps'])}"
        if prog.get("stage") == "done":
            yield final, snap, f"done in {prog.get('elapsed_s', '?')}s"
            return
        if prog.get("stage") == "error":
            yield final, snap, "error — see anima_server.log"
            return
        marker = (prog.get('t'), prog.get('step'), prog.get('stage'))
        if marker != seen:
            seen = marker
            yield final, snap, status
        time.sleep(0.5)
    prog = _read_prog()
    yield _read_img("/content/anima_perstep.png"), _read_img(SNAP_PATH), f"timeout at {prog}"


def build():
    import gradio as gr
    with gr.Blocks(title="Anima Aesthetic TPU") as demo:
        gr.Markdown("# Anima Aesthetic v1.1 (TPU v5e)")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(value=FERN_PROMPT, lines=6, label="Prompt")
                neg = gr.Textbox(value=FERN_NEG, lines=2, label="Negative prompt")
                with gr.Row():
                    height = gr.Number(value=1024, label="Height")
                    width = gr.Number(value=1024, label="Width")
                with gr.Row():
                    steps = gr.Slider(4, 50, value=30, step=1, label="Steps")
                    seed = gr.Number(value=0, label="Seed")
                    preview_every = gr.Slider(1, 10, value=5, step=1,
                                             label="Latent snapshot every N steps")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False)
                final_img = gr.Image(label="Final image")
                snap_img = gr.Image(label="Live latent (channel-mean, normalized)")
        btn.click(start_gen,
                  inputs=[prompt, neg, height, width, steps, seed, preview_every],
                  outputs=[final_img, snap_img, status])
    return demo


if __name__ == "__main__":
    build().launch(share=True)
