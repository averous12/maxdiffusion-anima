"""Real browser test for the Gradio UI using the Chromium already on this machine.

Launches its OWN browser instance with a throwaway profile. It never touches an
already-running Chrome and never kills one: the only process it terminates is the
one it spawned.

    python ops/browser_test.py --url http://127.0.0.1:7871 --out C:/tmp/shots
"""
import argparse
import os
import re
import sys

CHROME = r"C:/Program Files/imput/Helium/Application/chrome.exe"

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7871")
    ap.add_argument("--out", default=os.path.join(os.environ.get("TEMP", "/tmp"), "shots"))
    ap.add_argument("--chrome", default=CHROME)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=args.chrome,
            headless=True,
            args=["--no-first-run", "--no-default-browser-check",
                  "--disable-extensions", "--disable-background-networking"],
        )
        page = browser.new_page(viewport={"width": 1300, "height": 1000})
        # Gradio keeps an SSE stream open, so "networkidle" never fires.
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("text=Regional", timeout=60000)
        page.wait_for_timeout(2500)

        # ---- tabs exist -------------------------------------------------------
        for name in ("Settings", "LoRAs", "Decode", "SnapRef", "Regional", "Help"):
            ok = page.get_by_role("tab", name=name, exact=True).count() > 0
            check(f"tab {name} exists", ok)

        # ---- regional tab -----------------------------------------------------
        page.get_by_role("tab", name="Regional", exact=True).click()
        page.wait_for_timeout(1200)
        page.screenshot(path=os.path.join(args.out, "01_regional_tab.png"), full_page=True)

        common = page.get_by_label("Common prompt — prepended to every checked box", exact=False)
        check("common prompt visible", common.count() > 0)
        check("common prompt prefilled",
              common.count() > 0 and "masterpiece" in (common.first.input_value() or ""),
              common.first.input_value()[:40] if common.count() else "missing")

        neg = page.get_by_label("Common negative — prepended to every checked box", exact=False)
        check("common negative visible", neg.count() > 0)

        # Box 1 is open by default; Box 2 is a collapsed accordion.
        box1_prompt = page.get_by_label("Box prompt", exact=False).first
        check("box 1 prompt visible", box1_prompt.count() > 0)
        check("box 1 include checked",
              page.get_by_label("Include", exact=False).first.is_checked())

        # ---- interact ---------------------------------------------------------
        box1_prompt.fill("1girl, solo, red raincoat, night city background")
        # Sliders expose "number input for <label>"; a loose name match for "End"
        # also matches "prep(END)ed" in the common-prompt label, which silently
        # typed into the wrong field. Target the aria-label exactly.
        end_input = page.locator('input[aria-label="number input for End"]').first
        end_input.fill("0.6")
        end_input.blur()
        page.get_by_role("button", name="Refresh preview").click()
        page.wait_for_timeout(1800)
        preview = page.locator("text=model calls/step").first
        check("preview updates after refresh", preview.count() > 0)
        html = page.content()
        check("preview reports coverage", ("uncovered" in html) or ("Coverage complete" in html))
        page.screenshot(path=os.path.join(args.out, "02_preview.png"), full_page=True)
        # The common block must survive every interaction untouched.
        check("common prompt intact after editing",
              "masterpiece" in (common.first.input_value() or ""),
              common.first.input_value()[:40])
        check("end slider took the value", end_input.input_value().startswith("0.6"),
              end_input.input_value())

        # Untick the only box: preview must say plain generation.
        page.get_by_label("Include", exact=False).first.uncheck()
        page.get_by_role("button", name="Refresh preview").click()
        page.wait_for_timeout(1800)
        check("unticked box yields plain-run verdict",
              "plain generation" in page.content())
        page.get_by_label("Include", exact=False).first.check()
        page.get_by_role("button", name="Refresh preview").click()
        page.wait_for_timeout(1500)

        # ---- preview follows the selected aspect ------------------------------
        def canvas_size():
            # Measure the real rendered box of the regional canvas. Matching on raw
            # HTML picks up the main panel's resolution preview instead.
            box = page.locator("#regional-canvas").first.bounding_box()
            return (round(box["width"]), round(box["height"])) if box else (0, 0)

        def set_aspect(aw, ah):
            page.get_by_role("tab", name="Settings", exact=True).click()
            page.wait_for_timeout(700)
            for label, val in (("Aspect W", aw), ("Aspect H", ah)):
                box = page.get_by_label(label, exact=True).first
                box.fill(str(val))
                box.blur()
            page.wait_for_timeout(2500)   # change handlers redraw the preview
            page.get_by_role("tab", name="Regional", exact=True).click()
            page.wait_for_timeout(900)

        w0, h0 = canvas_size()
        check("default preview is square", w0 == h0 and w0 > 0, f"{w0}x{h0}")

        set_aspect(16, 9)
        w1, h1 = canvas_size()
        check("16:9 makes the canvas wide", w1 > h1 > 0, f"{w1}x{h1}")
        check("16:9 keeps the long side at 200", w1 == 200, f"{w1}")
        check("16:9 ratio is close to 1.78",
              abs((w1 / max(1, h1)) - 16 / 9) < 0.06, f"{w1 / max(1, h1):.2f}")
        page.screenshot(path=os.path.join(args.out, "02b_landscape.png"), full_page=True)

        set_aspect(9, 16)
        w2, h2 = canvas_size()
        check("9:16 makes the canvas tall", h2 > w2 > 0, f"{w2}x{h2}")
        check("9:16 keeps the long side at 200", h2 == 200, f"{h2}")
        page.screenshot(path=os.path.join(args.out, "02c_portrait.png"), full_page=True)

        set_aspect(1, 1)

        page.get_by_role("tab", name="Help", exact=True).click()
        page.wait_for_timeout(1200)
        body = page.content()
        for section in ("What each tab does", "Prompt syntax", "Regional prompting",
                        "Performance and cost", "API and CLI", "Troubleshooting"):
            check(f"help section {section!r}", section in body)
        check("help documents the blend", "max(Σm, 1)" in body)
        check("help no longer denies regional", "Not supported:** regional prompting" not in body)

        # Open every accordion and confirm content renders.
        for label in ("Regional prompting", "Performance and cost"):
            page.get_by_text(label, exact=True).first.click()
            page.wait_for_timeout(700)
        page.screenshot(path=os.path.join(args.out, "03_help.png"), full_page=True)
        check("cost table rendered", "2 + 2N" in page.content())

        # ---- snapref / decode tabs still intact --------------------------------
        page.get_by_role("tab", name="SnapRef", exact=True).click()
        page.wait_for_timeout(800)
        check("snapref controls present", page.get_by_label("Use SnapRef", exact=False).count() > 0)
        page.screenshot(path=os.path.join(args.out, "04_snapref.png"), full_page=True)

        # Only our own browser is closed; nothing else is touched.
        browser.close()


if __name__ == "__main__":
    main()
    print(f"BROWSER {PASS}/{PASS + FAIL}")
    sys.exit(1 if FAIL else 0)
