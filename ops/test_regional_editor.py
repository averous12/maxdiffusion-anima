"""Contract tests for the Gradio regional editor: widgets -> request JSON.

Runs without a TPU: it exercises the assembly and the preview together, which is the
part that used to be a hand-written JSON textbox.
"""
import base64
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]

import anima_regional_ui as rui  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


def png(path):
    from PIL import Image
    Image.new("RGB", (8, 8), (120, 30, 30)).save(path)


def slot(enabled=True, prompt="1girl, red coat", negative="", x=0.0, y=0.0, w=0.5, h=1.0,
         start=0.0, end=1.0, mode="char", ref1=None, ref2=None, ref3=None, weights=""):
    return [enabled, prompt, negative, x, y, w, h, start, end, mode,
            ref1, ref2, ref3, weights]


def test_unchecked_boxes_are_dropped():
    values = slot(True, "1girl") + slot(False, "1boy")
    regions = rui.assemble(values)
    check("unchecked dropped", len(regions) == 1 and regions[0]["prompt"] == "1girl",
          str(regions))


def test_box_is_canonical_and_clamped():
    regions = rui.assemble(slot(x=0.5, y=0.0, w=0.9, h=1.0))
    x, y, w, h = regions[0]["box"]
    check("box clamped to canvas", abs(x + w - 1.0) < 1e-6, f"{x} {w}")
    regions = rui.assemble(slot(x=-0.2, y=-0.1, w=0.1, h=0.1))
    x, y, w, h = regions[0]["box"]
    check("negative coords clamped", x == 0.0 and y == 0.0 and w >= 0.05 and h >= 0.05,
          f"{regions[0]['box']}")


def test_per_box_references_and_weights():
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "a.png"), os.path.join(td, "b.png")
        png(a), png(b)
        regions = rui.assemble(slot(ref1=a, ref2=b, weights="1.0, 0.6"))
        snap = regions[0]["snapref"]
        check("two refs encoded", len(regions[0]["ref_images_b64"]) == 2)
        check("weights parsed", snap["weights"] == [1.0, 0.6], str(snap["weights"]))
        check("budget scales with ref count", abs(snap["ref_mp_budget"] - 0.5) < 1e-6,
              str(snap["ref_mp_budget"]))
        raw = base64.b64decode(regions[0]["ref_images_b64"][0])
        check("ref payload is png", raw[:4] == b"\x89PNG")
        # Blank weights default to 1.0 each.
        regions = rui.assemble(slot(ref1=a, ref2=b, weights=""))
        check("default weights", regions[0]["snapref"]["weights"] == [1.0, 1.0])


def test_no_reference_means_no_snapref():
    regions = rui.assemble(slot())
    check("no snapref key", "snapref" not in regions[0], str(regions[0].keys()))


def test_negative_only_when_present():
    check("no empty negative key",
          "negative_prompt" not in rui.assemble(slot(negative="  "))[0])
    regions = rui.assemble(slot(negative="bad hands"))
    check("negative carried", regions[0]["negative_prompt"] == "bad hands")


def test_wire_shape_matches_server():
    """The dict must be accepted by the server's own normalizer."""
    from maxdiffusion.models import anima_regional as ar
    values = slot(True, "1girl") + slot(True, "1boy", x=0.5)
    regions = rui.assemble(values)
    normalized = ar.normalize_schedule(regions, maximum=16,
                                       common={"prompt": "masterpiece, best quality"})
    check("server accepts editor output", len(normalized) == 2, str(len(normalized)))
    check("common prepended by server",
          normalized[0]["prompt"] == "masterpiece, best quality, 1girl",
          normalized[0]["prompt"])


def test_preview_flags_gaps_and_overlap():
    gap = rui.preview_html(slot(True, "1girl", w=0.4))
    check("gap reported", "uncovered" in gap, gap[:80])
    full = rui.preview_html(slot(True, "1girl", w=1.0))
    check("complete coverage reported", "Coverage complete" in full, full[:80])
    over = rui.preview_html(slot(True, "1girl", w=1.0) + slot(True, "1boy", w=1.0))
    check("overlap reported", "covered twice" in over, over[:80])
    empty = rui.preview_html(slot(False, "x"))
    check("empty state", "plain generation" in empty, empty[:80])


def test_preview_warns_on_missing_common():
    html = rui.preview_html(slot(True, "1girl", w=1.0), "", "")
    check("warns when common empty", "Common prompt is empty" in html)


def test_preview_from_widgets_flattens():
    out = rui.preview_from_widgets(*(slot(True, "1girl", w=1.0) + ["mp, bq", "low"]))
    check("gradio entry point", "Coverage complete" in out, out[:80])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(f"REGIONAL_UI {PASS}/{PASS + FAIL}")
    sys.exit(1 if FAIL else 0)