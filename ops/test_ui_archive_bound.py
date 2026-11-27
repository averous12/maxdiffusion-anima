"""Archiving must never be able to hold the queue.

Reported symptom: the generation finished (server said GEN DONE, stage=done) but the browser
never got the image and no further request could be sent. The event runs with
concurrency_limit=1, so one stuck call is a permanent wedge -- and the call that sits between a
finished generation and its picture is save_outputs(), which writes to the archive directory:
usually a Drive FUSE mount.

_save_bounded caps that wait. The write still completes in the background; the caller gets a
report either way. This pins both halves: a fast save returns its own message, a hung save
returns within the budget instead of blocking forever.

Run from the workspace root.
"""
import sys
import time

sys.path.insert(0, "maxdiffusion/scripts")


def main():
    import anima_gradio_ui as ui

    fails = []
    real = ui.save_outputs

    try:
        # --- fast archive: the real message comes back untouched
        ui.save_outputs = lambda prog, req: "saved 2 new file(s) to /tmp/x"
        got = ui._save_bounded({}, {}, timeout=5)
        fast_ok = got == "saved 2 new file(s) to /tmp/x"
        print(f"{'PASS' if fast_ok else 'FAIL'}  fast save returns its own message: {got!r}")
        if not fast_ok:
            fails.append("fast save")

        # --- hung archive: bounded, and says so
        def hang(prog, req):
            time.sleep(30)
            return "saved 9 new file(s)"

        ui.save_outputs = hang
        t0 = time.time()
        got = ui._save_bounded({}, {}, timeout=1.0)
        dur = time.time() - t0
        hung_ok = dur < 5 and "still running" in got
        print(f"{'PASS' if hung_ok else 'FAIL'}  hung save returns in {dur:.1f}s with a note: "
              f"{got[:60]!r}")
        if not hung_ok:
            fails.append("hung save")

        # --- a raising save is reported, not raised: it happens inside a daemon thread, and a
        # traceback there lands in the UI log instead of the status line the user is reading.
        def boom(prog, req):
            raise RuntimeError("boom")

        ui.save_outputs = boom
        got = ui._save_bounded({}, {}, timeout=5)
        boom_ok = "archive failed" in got and "boom" in got
        print(f"{'PASS' if boom_ok else 'FAIL'}  raising save is reported, not raised: {got!r}")
        if not boom_ok:
            fails.append("raising save")
    finally:
        ui.save_outputs = real

    print()
    print("RESULT: ARCHIVE BOUNDED OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
