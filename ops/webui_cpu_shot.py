"""Screenshot the running WebUI from inside the Colab VM.

The cloudflared quick tunnel returns 429 from this host, so there is no public URL to point a
browser at. Instead drive headless Chromium on the VM against http://127.0.0.1:7860 -- same
render, no tunnel, and it proves the page hydrates rather than merely returning 200.

Playwright's sync API refuses to start inside the Jupyter asyncio loop that `colab exec` runs
in, so the browser work happens in a child process.
"""
import os
import subprocess
import sys
import time

OUT = "/content/ui_shot.png"
CHILD_PATH = "/content/_shot_child.py"

CHILD = r'''
import json, os, time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:7860/"
OUT = "/content/ui_shot.png"

with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1440, "height": 1600})
    errs = []
    page.on("pageerror", lambda e: errs.append(str(e)))
    # gradio holds a websocket open, so "networkidle" never fires; wait for the mount instead
    page.goto(URL, wait_until="domcontentloaded", timeout=120000)
    try:
        page.wait_for_selector("gradio-app", timeout=60000)
    except Exception:
        pass
    time.sleep(8)
    counts = page.evaluate("""() => {
        const deep = (root, sel, acc) => {
            root.querySelectorAll(sel).forEach(e => acc.push(e));
            root.querySelectorAll('*').forEach(e => { if (e.shadowRoot) deep(e.shadowRoot, sel, acc); });
            return acc;
        };
        return {textareas: deep(document,'textarea',[]).length,
                inputs: deep(document,'input',[]).length,
                buttons: deep(document,'button',[]).length,
                tabs: deep(document,'[role=tab]',[]).length,
                bodyLen: document.body.innerText.length};
    }""")
    print("title=" + page.title())
    print("counts=" + json.dumps(counts))
    if errs:
        print("page errors: " + str(errs[:3]))
    page.screenshot(path=OUT, full_page=True)
    b.close()
print("SHOT", os.path.getsize(OUT) if os.path.exists(OUT) else 0)
'''


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ui_up():
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:7860/", timeout=4) as r:
            return r.status
    except Exception:
        return None


def main():
    st = ui_up()
    log(f"ui status before shot: {st}")
    if st is None:
        # the detached UI died with its exec; start it again and wait
        log("respawning ui")
        subprocess.Popen([sys.executable, "/content/webui_launch.py"], cwd="/content",
                         stdout=open("/content/ui.log", "a"), stderr=subprocess.STDOUT,
                         start_new_session=True)
        for _ in range(60):
            time.sleep(2)
            if ui_up():
                break
        log(f"ui status after respawn: {ui_up()}")

    log("installing playwright")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "playwright"], check=True)
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    open(CHILD_PATH, "w").write(CHILD)
    log("rendering in a child process")
    rc = subprocess.run([sys.executable, CHILD_PATH], capture_output=True, text=True)
    log(f"child rc={rc.returncode}")
    for line in (rc.stdout or "").splitlines():
        log("  " + line)
    if rc.returncode != 0:
        for line in (rc.stderr or "").splitlines()[-8:]:
            log("  ! " + line)

    size = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    log(f"SHOT {OUT} {size} B")
    ok = size > 20000
    log("WEBUI RENDER OK" if ok else "WEBUI RENDER FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
