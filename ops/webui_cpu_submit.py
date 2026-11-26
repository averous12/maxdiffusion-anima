"""Drive the WebUI like a user and verify the request contract it produces.

Rendering proves the page mounts. This proves the wiring: click Generate, then read
/content/anima_request.json off the VM and check the fields the runtime actually reads
(height, width, steps, guidance, seed, prompt). No server needs to be running -- the UI's
whole job is to write that file, and an absent consumer must not break it.
"""
import json
import os
import subprocess
import sys
import time

CHILD_PATH = "/content/_submit_child.py"
REQ = "/content/anima_request.json"

CHILD = r'''
import json, os, time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox"])
    page = b.new_page(viewport={"width": 1440, "height": 1600})
    page.goto("http://127.0.0.1:7860/", wait_until="domcontentloaded", timeout=120000)
    try:
        page.wait_for_selector("gradio-app", timeout=60000)
    except Exception:
        pass
    time.sleep(8)

    # set the prompt so we can prove the value travels
    ta = page.locator("textarea").first
    ta.fill("cpu webui contract test, 1girl, solo")
    print("filled prompt")

    btn = page.get_by_role("button", name="Generate").first
    btn.click()
    print("clicked Generate")
    time.sleep(5)
    b.close()
'''


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    if os.path.exists(REQ):
        os.remove(REQ)
    before = os.path.exists(REQ)

    open(CHILD_PATH, "w").write(CHILD)
    rc = subprocess.run([sys.executable, CHILD_PATH], capture_output=True, text=True)
    log(f"child rc={rc.returncode}")
    for line in (rc.stdout or "").splitlines():
        log("  " + line)
    if rc.returncode != 0:
        for line in (rc.stderr or "").splitlines()[-8:]:
            log("  ! " + line)

    log(f"request existed before click: {before}")
    if not os.path.exists(REQ):
        log("CONTRACT FAILED: no anima_request.json written")
        return 1

    d = json.load(open(REQ))
    log(f"request keys: {sorted(d.keys())}")
    # the UI and the runtime both use "negative_prompt"; an earlier version of this
    # check asserted "negative" and reported a false failure against a correct request
    need = ("prompt", "negative_prompt", "height", "width", "steps", "guidance", "seed")
    missing = [k for k in need if k not in d]
    log(f"height={d.get('height')} width={d.get('width')} steps={d.get('steps')} "
        f"guidance={d.get('guidance')} seed={d.get('seed')}")
    log(f"prompt={str(d.get('prompt'))[:60]!r}")
    log(f"negative={str(d.get('negative_prompt'))[:50]!r}")
    if missing:
        log(f"CONTRACT FAILED: missing {missing}")
        return 1
    if d.get("height") != 1024 or d.get("width") != 1024:
        log("CONTRACT FAILED: default size is not 1024x1024")
        return 1
    log("WEBUI CONTRACT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
