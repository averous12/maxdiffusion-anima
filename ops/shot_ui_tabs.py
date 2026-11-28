"""Screenshot every WebUI tab and dump the visible control labels.

Static config checks prove the widgets exist; only a render proves they are laid out in a way a
person can read. Uses the local Helium/Chromium binary rather than downloading one.
"""
import json
import os
import sys

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:%s/" % (sys.argv[1] if len(sys.argv) > 1 else "7899")
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:/Users/avero/AppData/Local/hermes/cache/scratch/ui_shots"
CANDIDATES = [
    r"C:/Program Files/imput/Helium/Application/chrome.exe",
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
]
TABS = ["Settings", "SnapRef", "Control", "Regional", "LoRA", "Help"]

os.makedirs(OUT, exist_ok=True)
exe = next((p for p in CANDIDATES if os.path.exists(p)), None)
print("browser:", exe or "playwright default")


def main():
    errors = []
    with sync_playwright() as pw:
        browser = (pw.chromium.launch(executable_path=exe, headless=True) if exe
                   else pw.chromium.launch(headless=True))
        page = browser.new_page(viewport={"width": 1500, "height": 1150})
        page.on("pageerror", lambda e: errors.append(str(e)))
        # Gradio keeps an event stream open, so "networkidle" never fires; wait for the
        # tab bar itself instead of guessing at a timeout.
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("button[role='tab']", timeout=60000)
        page.wait_for_timeout(2500)

        found = [t.inner_text().strip() for t in page.query_selector_all("button[role='tab']")]
        print("tabs rendered:", found)
        order_ok = [t for t in found if t in TABS] == TABS
        print(f"{'PASS' if order_ok else 'FAIL'}  tab order is {TABS}")
        if not order_ok:
            print("      got:", found)

        for name in TABS:
            try:
                page.get_by_role("tab", name=name, exact=True).click(timeout=8000)
                page.wait_for_timeout(700)
                path = os.path.join(OUT, "%s.png" % name.lower())
                page.screenshot(path=path, full_page=True)
                # labels + info text currently visible, so the copy can be reviewed as text too
                labels = page.eval_on_selector_all(
                    "label span, .label-wrap span",
                    "els => els.map(e => e.innerText.trim()).filter(t => t && t.length < 90)")
                info = page.eval_on_selector_all(
                    ".info, span.info, .gr-info",
                    "els => els.map(e => e.innerText.trim()).filter(Boolean)")
                print(json.dumps({"tab": name, "png": path, "labels": labels[:14],
                                  "info": info[:8]}))
            except Exception as e:
                print(f"FAIL  tab {name}: {type(e).__name__}: {e}")
                errors.append(f"{name}: {e}")

        print("page errors:", len(errors))
        for e in errors[:5]:
            print("   ", e[:160])
        browser.close()
    return 0 if order_ok and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
