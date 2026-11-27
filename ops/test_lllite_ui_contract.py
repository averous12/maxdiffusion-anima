"""The Control tab must actually put lllite into the request, and must not break SnapRef.

Two things are checked, because both were real risks:

1. `req["lllite"]` is emitted with the control image as a *path*, the model name, strength,
   aspect and the step window. The server takes a path and coerces it; pixels there produce the
   `(1, 1) '<Un21'` PIL error that made every control request fail.
2. Turning control OFF must leave SnapRef intact. While adding the block, the `else` clause of
   `if snap_cfg is not None:` was briefly orphaned by the insert, which would have silently
   stripped snapref from every request with control disabled.

start_gen is a generator that writes the request and then polls, so `_atomic_json` is replaced
with a catcher and the generator is advanced once.
"""
import os
import sys

sys.path.insert(0, "maxdiffusion/scripts")

CONTROL = "samples/pose4_control.png"
FIRST12 = ("masterpiece, 1girl", "worst quality", 1, 1, "Custom size", 512, 512, 4, 1234, 4.0, 0, 1)


class _Captured(Exception):
    """Raised to unwind the generator without tripping PEP 479 (StopIteration -> RuntimeError)."""


def capture(ui, **kw):
    seen = {}

    def fake_atomic(path, payload):
        seen["path"] = path
        seen["req"] = payload
        raise _Captured("captured")

    real = ui._atomic_json
    ui._atomic_json = fake_atomic
    try:
        gen = ui.start_gen(*FIRST12, **kw)
        try:
            next(gen)
        except _Captured:
            pass
        except Exception as e:
            # anything else is a genuine failure in request building
            print(f"FAIL  start_gen raised {type(e).__name__}: {e}")
            return None
    finally:
        ui._atomic_json = real
    return seen.get("req")


def main():
    import anima_gradio_ui as ui

    if not os.path.exists(CONTROL):
        print(f"SKIP: no control image at {CONTROL}")
        return 2

    fails = []

    # --- 1. control on
    req = capture(ui, lllite_on=True, lllite_image=CONTROL,
                  lllite_model="anima_3dpose_6200.safetensors", lllite_strength=0.75,
                  lllite_aspect="letterbox", lllite_start=0.1, lllite_end=0.9)
    if req is None:
        fails.append("control-on request")
    else:
        ll = req.get("lllite")
        ok = bool(ll) and ll.get("image") == CONTROL and ll.get("model") == \
            "anima_3dpose_6200.safetensors" and abs(ll.get("strength") - 0.75) < 1e-6 \
            and ll.get("aspect") == "letterbox" and ll.get("start_percent") == 0.1 \
            and ll.get("end_percent") == 0.9
        print(f"{'PASS' if ok else 'FAIL'}  control on -> req['lllite'] = {ll}")
        if not ok:
            fails.append("lllite payload")

    # --- 2. control off, snapref on: snapref must survive
    # the reference must be a real file: start_gen validates references and returns early
    # (yielding an error update) when the path is missing, which is correct behaviour
    req2 = capture(ui, lllite_on=False, snapref_on=True, snapref_images=[CONTROL],
                   snapref_strength=0.9)
    if req2 is None:
        fails.append("control-off request")
    else:
        has_ll = "lllite" in req2
        has_snap = "snapref" in req2
        ok = (not has_ll) and has_snap
        print(f"{'PASS' if ok else 'FAIL'}  control off -> lllite absent ({not has_ll}), "
              f"snapref kept ({has_snap})")
        if not ok:
            fails.append("snapref survives control off")

    # --- 3. no control image but enabled: must not fabricate a payload
    req3 = capture(ui, lllite_on=True, lllite_image=None,
                   lllite_model="anima_3dpose_6200.safetensors")
    if req3 is not None:
        ok = "lllite" not in req3
        print(f"{'PASS' if ok else 'FAIL'}  enabled with no image -> no lllite block")
        if not ok:
            fails.append("empty image guard")

    print()
    print("RESULT: CONTROL TAB CONTRACT OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
