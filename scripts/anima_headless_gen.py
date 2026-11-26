"""Headless one-shot generation through the Gradio UI's own start_gen.

This exists so a notebook can exercise the full UI path *without* importing the UI into the
kernel. `anima_gradio_ui` imports gradio, which lives in the repo venv, not in the Colab system
kernel -- and running long work in the kernel is what makes a runtime unresponsive for every
other cell. So the notebook calls this file with the venv python and tails its log.

Configuration is by environment, so no argument plumbing is needed by the caller:

  ANIMA_PROMPT, ANIMA_NEG          prompt / negative prompt
  ANIMA_W, ANIMA_H                 size (rounded to a multiple of 16 by the UI)
  ANIMA_STEPS, ANIMA_SEED, ANIMA_CFG, ANIMA_PREVIEW_EVERY
  ANIMA_OUT                        where to save the final image
  ANIMA_SAVE_DIR                   archive dir (the UI copies finished images there)

Exit code 0 only if an image was actually produced and looked at as a PIL image: a valid PNG on
disk is a transport check, not proof that the UI yielded something Gradio could display.
"""
import json
import os
import sys
import time
import traceback

SCRIPTS = os.environ.get("ANIMA_SCRIPTS", "/content/maxdiffusion/scripts")
sys.path.insert(0, SCRIPTS)

PROG = os.environ.get("ANIMA_PROG", "/content/anima_progress.json")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def main():
    try:
        import anima_gradio_ui as ui
    except Exception as e:
        log(f"cannot import the UI ({type(e).__name__}: {e})")
        log("this must run with the repo venv python, where gradio is installed")
        return 2

    prompt = os.environ.get("ANIMA_PROMPT", "masterpiece, best quality, 1girl, solo")
    neg = os.environ.get("ANIMA_NEG", "worst quality, low quality, blurry")
    w, h = env_int("ANIMA_W", 1024), env_int("ANIMA_H", 1024)
    steps = env_int("ANIMA_STEPS", 4)
    seed = env_int("ANIMA_SEED", 1234)
    cfg = env_float("ANIMA_CFG", 4.0)
    preview_every = env_int("ANIMA_PREVIEW_EVERY", 0)
    out = os.environ.get("ANIMA_OUT", "/content/ui_final.png")

    log(f"ui module imported; request {w}x{h} steps={steps} seed={seed} cfg={cfg} "
        f"preview_every={preview_every}")

    n_img = 0
    types = set()
    final_saved = False
    pv_last = None
    t0 = time.time()

    try:
        for yield_tuple in ui.start_gen(prompt, neg, 1, 1, "Custom size",
                                       w, h, steps, seed, cfg, preview_every, 1):
            image = yield_tuple[0]
            status = yield_tuple[1]
            try:
                prog = json.load(open(PROG))
            except Exception:
                prog = {}
            if status != pv_last:
                pv_last = status
                log(f"[{time.time() - t0:6.1f}s] status: {status}")
            if prog.get("stage"):
                log(f"             server stage={prog.get('stage')} "
                    f"step={prog.get('step')}/{prog.get('steps')}")

            # isinstance, not a class-name check: PIL decodes PNGs to PngImageFile, so a
            # name comparison against {'Image'} fails on a perfectly good image.
            try:
                from PIL import Image as _PILImage
                is_img = isinstance(image, _PILImage.Image)
            except Exception:
                is_img = image is not None

            if is_img and image is not None:
                n_img += 1
                types.add(type(image).__name__)
                if prog.get("stage") == "done":
                    image.save(out)
                    final_saved = True
                    log(f"[{time.time() - t0:6.1f}s] final {image.size} {image.mode} -> {out}")
    except Exception:
        log("start_gen raised:")
        traceback.print_exc()
        return 3

    log(f"previews/frames yielded={n_img} types={sorted(types)} saved={final_saved}")

    # bytes here is exactly the defect that made the browser UI show nothing.
    if n_img == 0:
        log("FAILED: the UI yielded no image (gradio needs PIL.Image/ndarray/path, not bytes)")
        return 4
    if not final_saved:
        log("FAILED: never reached stage=done, so no final image was saved")
        return 5

    try:
        from PIL import Image
        im = Image.open(out)
        log(f"PNG check: {im.size} {im.mode} {os.path.getsize(out)} bytes")
    except Exception as e:
        log(f"PNG check failed: {e}")
        return 6

    log("HEADLESS GEN OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
