"""Batching is split in two: batch size is one pass, batch count is consecutive passes.

4 images in one pass need four copies of the activations, which is what OOMs at 1024 px once
SnapRef and control are both active. 4 runs of 1 image never need more than one image's worth.

The wrapper sits in front of start_gen so the runtime contract is untouched: it lifts the runs
value out of the positional inputs and calls start_gen once per run, merging the galleries so a
later run cannot erase the earlier images.

Also pins the new default: no previews.

Run from the workspace root.
"""
import sys

sys.path.insert(0, "maxdiffusion/scripts")


def main():
    import gradio as gr

    import anima_gradio_ui as ui

    fails = []
    demo = ui.build()
    comps = demo.config["components"]

    def slider_default(label_fragment, exact=False):
        for c in comps:
            if c.get("type") != "slider":
                continue
            p = c.get("props") or {}
            lab = str(p.get("label") or "")
            if (lab == label_fragment) if exact else (label_fragment in lab):
                return p.get("value"), lab
        return None, None

    # --- defaults
    v, lab = slider_default("VAE preview")
    ok = v == 0
    print(f"{'PASS' if ok else 'FAIL'}  preview default is off  ({lab!r} = {v})")
    if not ok:
        fails.append("preview default")

    v, lab = slider_default("Batch count (consecutive runs)", exact=True)
    ok = v == 1
    print(f"{'PASS' if ok else 'FAIL'}  runs default is 1  ({lab!r} = {v})")
    if not ok:
        fails.append("runs default")

    v, lab = slider_default("Batch size (per run)", exact=True)
    ok = v == 1
    print(f"{'PASS' if ok else 'FAIL'}  batch size default is 1  ({lab!r} = {v})")
    if not ok:
        fails.append("batch size default")

    # --- the runs slot must sit exactly where start_gen's *args begins
    import inspect
    params = list(inspect.signature(ui.start_gen).parameters)
    idx = ui._runs_index()
    n_named = len([p for p in params if params.index(p) < len(params) - 1])
    boundary_ok = params[idx] == "regional_values"
    print(f"{'PASS' if boundary_ok else 'FAIL'}  runs index {idx} is the var-positional "
          f"boundary (got {params[idx] if idx < len(params) else 'out of range'})")
    if not boundary_ok:
        fails.append("runs index")

    # --- runs == 1 must forward the arguments untouched
    calls = []

    def fake_gen(*args):
        calls.append(list(args))
        yield (None, "status", None, None, None,
               gr.update(visible=True, value=[(f"img{len(calls)}.png", "seed 1")]))

    real_gen, real_idx = ui.start_gen, ui._runs_index
    ui.start_gen = fake_gen
    ui._runs_index = lambda: idx
    try:
        payload = list(range(idx + 1 + 14))          # runs sits at idx, then regional values
        payload[idx] = 1
        out = list(ui.start_gen_runs(*payload))
        fwd_ok = len(calls) == 1 and calls[0] == payload[:idx] + payload[idx + 1:]
        print(f"{'PASS' if fwd_ok else 'FAIL'}  runs=1 forwards once, runs value stripped "
              f"({len(calls)} call(s), arg count {len(calls[0]) if calls else 0})")
        if not fwd_ok:
            fails.append("runs=1 forwarding")

        # --- runs == 3 must run three times and keep every image
        calls.clear()
        payload[idx] = 3
        out = list(ui.start_gen_runs(*payload))
        ran_ok = len(calls) == 3
        final_gallery = ui._gallery_items(out[-1][5])
        merged_ok = len(final_gallery) == 3
        status_ok = "run 3/3" in str(out[-1][1])
        print(f"{'PASS' if ran_ok else 'FAIL'}  runs=3 calls start_gen three times ({len(calls)})")
        print(f"{'PASS' if merged_ok else 'FAIL'}  gallery keeps all images "
              f"({len(final_gallery)} of 3)")
        print(f"{'PASS' if status_ok else 'FAIL'}  status reports the run count: {out[-1][1]!r}")
        if not (ran_ok and merged_ok and status_ok):
            fails.append("runs=3 behaviour")

        # --- batch size 1: the gallery is empty and hidden, the image is only in the result
        # slot. Consecutive runs must still accumulate every image into one visible gallery,
        # otherwise each run wipes the previous one and only the last survives.
        calls.clear()

        def fake_single(*args):
            calls.append(list(args))
            n = len(calls)
            yield (f"img{n}.png", f"status {n}", None, None, None,
                   gr.update(visible=False, value=[]))

        ui.start_gen = fake_single
        payload[idx] = 4
        out = list(ui.start_gen_runs(*payload))
        items = ui._gallery_items(out[-1][5])
        visible = out[-1][5].get("visible")
        ok_single = len(items) == 4 and visible is True and all(
            str(items[k][0]) == f"img{k + 1}.png" for k in range(4))
        print(f"{'PASS' if ok_single else 'FAIL'}  batch size 1 x 4 runs fills one visible gallery "
              f"({len(items)} items, visible={visible})")
        if not ok_single:
            fails.append("gallery at batch 1")

        ui.start_gen = fake_gen

        # --- unknown/blank runs value must not explode
        calls.clear()
        payload[idx] = None
        list(ui.start_gen_runs(*payload))
        none_ok = len(calls) == 1
        print(f"{'PASS' if none_ok else 'FAIL'}  empty runs value falls back to a single run")
        if not none_ok:
            fails.append("runs None")
    finally:
        ui.start_gen, ui._runs_index = real_gen, real_idx

    print()
    print("RESULT: BATCH SPLIT OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
