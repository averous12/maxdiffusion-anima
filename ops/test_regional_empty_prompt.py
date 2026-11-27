"""SnapRef + a ticked-but-empty regional box used to fail on the server, not the UI.

The runtime's regional canonicaliser joins the common prompt with the box prompt and rejects an
empty result:

    RegionalPromptError: region 0 prompt is empty

The editor happily assembled the box (Include ticked, prompt blank), wrote the request, and the
error came back mid-submit naming "region 0" -- an index the UI never shows, and nothing to do
with SnapRef, which is why it looked like a keys/size problem in the snapref payload.

Now that is caught before the request is written and the message names the box.

Run from the workspace root.
"""
import sys

sys.path.insert(0, "maxdiffusion/scripts")

FIRST12 = ("masterpiece, 1girl", "worst quality", 1, 1, "Custom size", 512, 512, 4, 1234, 4.0, 0, 1)
REF = "samples/pose4_control.png"


class _Captured(Exception):
    pass


def capture(ui, mid, boxes, common_pos=""):
    """start_gen takes the regional editor values positionally, after the common prompt."""
    seen = {}
    mid = list(mid)
    mid[18] = common_pos      # regional_common_pos sits at 18 in the positional contract

    def fake_atomic(path, payload):
        seen["req"] = payload
        raise _Captured()

    real = ui._atomic_json
    ui._atomic_json = fake_atomic
    try:
        gen = ui.start_gen(*FIRST12, *mid, *boxes)
        try:
            next(gen)
        except _Captured:
            pass
        except StopIteration as stop:
            seen["returns"] = stop.value
    finally:
        ui._atomic_json = real
    return seen


def box(enabled, prompt=""):
    # enabled, prompt, negative, x, y, w, h, start, end, mode, ref1..3, weights
    return [enabled, prompt, "", 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, "char", None, None, None, ""]


def main():
    import os

    import anima_gradio_ui as ui

    if not os.path.exists(REF):
        print(f"SKIP: no reference image at {REF}")
        return 2

    fails = []
    # prompt_style, tags, snapref_on, adapter, images, mode, layout, weights, strength, ...
    MID = ["comfy", "", True, "", [REF], "cs", "separate", None, 0.9, 1.0, False, False,
           4.5, "", "pixel", "", None, "", "", "", 0.35,
           False, None, "", 1.0, "letterbox", 0.0, 1.0]
    boxes = box(True, "") + box(False) + box(False) + box(False)

    # --- A: the reported case. Must be refused before a request is written.
    got = capture(ui, MID, boxes)
    msg = ""
    if "returns" in got and got["returns"]:
        msg = str(got["returns"][0])
    refused = ("Box 1" in msg) and ("no prompt" in msg)
    wrote = "req" in got
    print(f"{'PASS' if refused and not wrote else 'FAIL'}  snapref + ticked empty box is refused, "
          f"naming the box (wrote_request={wrote})")
    print(f"      message: {msg[:120]}")
    if not (refused and not wrote):
        fails.append("empty box refusal")

    # --- B: with a prompt, snapref + regional must go through together.
    got = capture(ui, MID, box(True, "1girl, red coat") + box(False) * 1 +
                  box(False) + box(False))
    req = got.get("req") or {}
    both = "regional" in req and "snapref" in req
    slots = [r.get("slot") for r in req.get("regional", [])]
    print(f"{'PASS' if both else 'FAIL'}  snapref + regional both present "
          f"(keys: {sorted(req) if req else 'no request'})")
    print(f"      regional slots: {slots}")
    if not both:
        fails.append("regional+snapref")

    # --- C: an empty box is legitimate when the common prompt carries the text.
    got = capture(ui, MID, boxes, common_pos="cinematic scene")
    req = got.get("req") or {}
    ok_common = "regional" in req and "snapref" in req
    print(f"{'PASS' if ok_common else 'FAIL'}  empty box allowed when a common prompt is set")
    if not ok_common:
        fails.append("common prompt")

    # --- D: the server-side backstop is still there for API clients.
    sys.path.insert(0, "maxdiffusion/src")
    from maxdiffusion.models import anima_regional as R
    try:
        R.normalize_schedule([{"prompt": "", "box": [0, 0, 0.5, 1]}], maximum=16)
        print("FAIL  server backstop missing (empty prompt accepted)")
        fails.append("server backstop")
    except Exception as e:
        print(f"PASS  server still rejects it directly: {type(e).__name__}: {e}")

    print()
    print("RESULT: REGIONAL EMPTY PROMPT OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
