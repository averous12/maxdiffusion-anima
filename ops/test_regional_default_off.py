"""An untouched UI must produce a plain request.

The regional editor shipped with box 1 already ticked ("Include"), covering the left half
(x=0, w=0.5) with a blank prompt. Regional is a different conditioning path, so every generation
from a fresh page silently took it, and the user had not asked for anything of the kind.

Two checks:
  1. every "Include" checkbox defaults to False;
  2. start_gen with the real default values emits a request with no regional, no lllite and no
     snapref keys -- i.e. the default path is plain.

The second one is the one that matters: a False checkbox is only correct if the request path
also drops the stage.
"""
import sys

sys.path.insert(0, "maxdiffusion/scripts")

FIRST12 = ("masterpiece, 1girl", "worst quality", 1, 1, "Custom size", 512, 512, 4, 1234, 4.0, 0, 1)


class _Captured(Exception):
    pass


def capture(ui, args=(), **kw):
    seen = {}

    def fake_atomic(path, payload):
        seen["req"] = payload
        raise _Captured()

    real = ui._atomic_json
    ui._atomic_json = fake_atomic
    try:
        gen = ui.start_gen(*FIRST12, *args, **kw)
        try:
            next(gen)
        except _Captured:
            pass
        except Exception as e:
            print(f"FAIL  start_gen raised {type(e).__name__}: {e}")
            return None
    finally:
        ui._atomic_json = real
    return seen.get("req")


def main():
    import anima_gradio_ui as ui

    fails = []
    cfg = ui.build().config["components"]

    # --- every checkbox default, so any "on by default" surprise is visible here
    boxes = []
    for c in cfg:
        if c.get("type") == "checkbox":
            p = c.get("props") or {}
            boxes.append((p.get("label"), p.get("value")))
    print("checkbox defaults:")
    for label, value in boxes:
        mark = "  <-- ON BY DEFAULT" if value else ""
        print(f"   {str(label)[:52]:54} {value}{mark}")
    includes = [v for l, v in boxes if l == "Include"]
    inc_ok = bool(includes) and not any(includes)
    print(f"\n{'PASS' if inc_ok else 'FAIL'}  all {len(includes)} regional Include boxes default off")
    if not inc_ok:
        fails.append("Include default")

    # --- the default request must be plain
    n_boxes = len(includes)
    flat = []
    for _ in range(n_boxes):
        # enabled, prompt, negative, x, y, w, h, start, end, mode, ref1, ref2, ref3, weights
        flat += [False, "", "", 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, "char", None, None, None, ""]
    # Everything between batch_count and the regional expansion, in signature order, so the
    # box values land on *regional_values rather than on the snapref parameters.
    MID = ("comfy", "", False, "", None, "cs", "separate", None, 0.0, 1.0, False, False,
           4.5, "", "pixel", "", None, "", "", "", 0.35,
           False, None, "", 1.0, "letterbox", 0.0, 1.0)
    req = capture(ui, MID + tuple(flat))
    if req is None:
        fails.append("default request")
    else:
        plain = [k for k in ("regional", "lllite", "snapref", "lora_swap", "loras") if k in req]
        ok = not plain
        print(f"{'PASS' if ok else 'FAIL'}  default request is plain "
              f"(extra keys: {plain or 'none'})")
        if not ok:
            fails.append("default not plain")
        print(f"      keys: {sorted(req)}")

    print()
    print("RESULT: DEFAULT IS PLAIN" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
