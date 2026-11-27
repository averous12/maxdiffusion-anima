"""The LoRA button must talk to the runtime the way the runtime actually listens.

It used to POST to localhost:8000/lora_swap. The runtime opens no HTTP port at all — it watches
the request file — so that button could only ever fail with a connection error.

Now it writes a `lora_swap` control request and waits for stage=lora-swapped. This pins that:
the payload shape, the path it is written to, the success message, the failure message, and the
absence of any HTTP call left behind.

Run from the workspace root.
"""
import os
import sys

sys.path.insert(0, "maxdiffusion/scripts")
UI_SRC = "maxdiffusion/scripts/anima_gradio_ui.py"


def main():
    import anima_gradio_ui as ui

    fails = []

    # --- no HTTP remains
    src = open(UI_SRC, encoding="utf-8").read()
    # A docstring may mention the old endpoint when explaining why it is gone; what matters is
    # that no request is actually made.
    no_post = "requests.post(" not in src
    no_req_import = "import requests" not in src
    no_port = no_post and no_req_import
    print(f"{'PASS' if no_port else 'FAIL'}  no HTTP call left "
          f"(requests.post={not no_post}, import requests={not no_req_import})")
    if not no_port:
        fails.append("http call left")

    # --- success path
    captured = {}
    real_atomic, real_read, real_installed = ui._atomic_json, ui._read_json, ui._installed_loras
    ui._installed_loras = lambda: ["demo.safetensors"]

    def fake_atomic(path, payload):
        captured["path"] = path
        captured["req"] = payload

    def fake_read(path, default=None):
        r = captured.get("req") or {}
        if path == ui.PROG_PATH:
            return {"gen_id": r.get("gen_id"), "stage": "lora-swapped",
                    "swap_seconds": 1.25, "merged": 280, "skipped": 0}
        return default if default is not None else {}

    ui._atomic_json = fake_atomic
    ui._read_json = fake_read
    ui.SWAP_TIMEOUT_S = 5
    try:
        html, _ = ui.apply_loras(["demo.safetensors"], 0.8)
    finally:
        ui._atomic_json, ui._read_json, ui._installed_loras = real_atomic, real_read, real_installed

    req = captured.get("req") or {}
    wrote_file = captured.get("path") == ui.REQ_PATH
    has_swap = isinstance(req.get("lora_swap"), list) and len(req["lora_swap"]) == 1
    spec = (req.get("lora_swap") or [{}])[0]
    spec_ok = (spec.get("path", "").endswith("demo.safetensors")
               and abs(float(spec.get("scale", 0)) - 0.8) < 1e-9)
    print(f"{'PASS' if wrote_file else 'FAIL'}  writes {captured.get('path')}")
    print(f"{'PASS' if has_swap and spec_ok else 'FAIL'}  lora_swap spec = {spec}")
    print(f"{'PASS' if req.get('go') and req.get('gen_id') else 'FAIL'}  carries go + gen_id "
          f"({req.get('gen_id')})")
    said_ok = "Hot-swapped" in html and "1.25" in html
    print(f"{'PASS' if said_ok else 'FAIL'}  success message reflects the swap result")
    if not (wrote_file and has_swap and spec_ok and said_ok):
        fails.append("success path")

    # --- dead server: no progress written, must not claim success
    captured.clear()
    # the file must still be considered installed, otherwise the call is rejected before the
    # request is written and this would test the wrong branch
    ui._installed_loras = lambda: ["demo.safetensors"]
    ui._atomic_json, ui._read_json = fake_atomic, (lambda p, d=None: d if d is not None else {})
    ui.SWAP_TIMEOUT_S = 0.2
    try:
        html2, _ = ui.apply_loras(["demo.safetensors"], 1.0)
    finally:
        ui._atomic_json, ui._read_json, ui._installed_loras = real_atomic, real_read, real_installed
    dead_ok = ("no response" in html2) and ("Hot-swapped" not in html2)
    if not dead_ok:
        print("      html2 was:", html2[:160])
    print(f"{'PASS' if dead_ok else 'FAIL'}  absent server reports no response, not success")
    if not dead_ok:
        fails.append("timeout path")

    # --- unknown file is rejected before any request is written
    captured.clear()
    ui._installed_loras = lambda: ["demo.safetensors"]
    ui._atomic_json = fake_atomic
    try:
        html3, _ = ui.apply_loras(["nope.safetensors"], 1.0)
    finally:
        ui._installed_loras, ui._atomic_json = real_installed, real_atomic
    reject_ok = "ignored unknown" in html3 and not captured
    print(f"{'PASS' if reject_ok else 'FAIL'}  unknown LoRA rejected, nothing written")
    if not reject_ok:
        fails.append("validation path")

    print()
    print("RESULT: LORA REWIRE OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
