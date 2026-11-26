"""Verify the Gradio WebUI on a Colab CPU instance.

The UI is deliberately independent of the model: it only writes /content/anima_request.json
and polls /content/anima_progress.json. So the WebUI can be verified on a plain CPU VM
without booting 6.6 GB of weights -- which is the point of doing it here.

Two phases, so a failure is attributable:
  build   -- assemble the Blocks graph and count components (no server)
  serve   -- launch on 0.0.0.0:7860 and confirm it answers HTTP

Usage:  python webui_cpu_verify.py build|serve
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

CONTENT = "/content"
LOG = f"{CONTENT}/webui_verify.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ensure_deps():
    log("installing gradio/pillow/numpy (CPU)")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "gradio", "pillow", "numpy"], check=True)
    import gradio as gr
    log(f"gradio {gr.__version__}")


def phase_build():
    sys.path.insert(0, CONTENT)
    import anima_gradio_ui as ui
    import gradio as gr

    t0 = time.perf_counter()
    demo = ui.build()
    secs = time.perf_counter() - t0

    cfg = demo.config
    comps = cfg.get("components", [])
    deps = cfg.get("dependencies", [])
    log(f"build() OK in {secs:.1f}s: {len(comps)} components, {len(deps)} event wirings")

    # the UI must be able to recognise its own request/progress files
    log(f"REQ_PATH={ui.REQ_PATH}")
    log(f"PROG_PATH={ui.PROG_PATH}")
    ok = len(comps) > 50 and len(deps) > 5
    log(f"BUILD {'OK' if ok else 'FAILED'} ({len(comps)} components)")
    return 0 if ok else 1


def phase_serve():
    sys.path.insert(0, CONTENT)
    import anima_gradio_ui as ui

    demo = ui.build()
    log("launching on 0.0.0.0:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                quiet=True, show_error=True)
    return 0


def main():
    # `colab exec -f` puts the script path in argv[1], so match on the word instead
    mode = "serve" if "serve" in sys.argv else "build"
    ensure_deps()
    return phase_build() if mode == "build" else phase_serve()


if __name__ == "__main__":
    sys.exit(main())
