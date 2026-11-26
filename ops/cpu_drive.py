"""Drive the real CPU runtime through every control mode and record what each produced.

One request per mode, identical prompt/seed/size, so any difference between the images is
attributable to the mode rather than to sampling. 256x256 / 1 step keeps CPU cost sane.

The server returns the PREVIOUS image if it finds a stale stage=done record (see the comment at
pipeline.py:1500), so every request carries a unique gen_id and the progress file is cleared
first; completion is matched on gen_id, never on stage alone.
"""
import hashlib
import json
import os
import sys
import time

CONTENT = "/content"
REQ = f"{CONTENT}/anima_request.json"
PROG = f"{CONTENT}/anima_progress.json"
OUTDIR = f"{CONTENT}/mode_out"
CTL = f"{CONTENT}/ctl_pose.png"
REF = f"{CONTENT}/ref_girl.png"
LLLITE_MODEL = "anima_3dpose_6200.safetensors"

BASE = {
    "prompt": "masterpiece, best quality, 1girl, solo, standing, simple background",
    "negative_prompt": "worst quality, low quality, blurry, jpeg artifacts",
    "height": 256, "width": 256, "steps": 1, "guidance": 4.0, "seed": 1234,
    "preview_every": 0, "prompt_style": "comfy", "batch_count": 1, "go": True,
}

REGIONAL = [{"prompt": "1girl with long silver hair, upper body",
             "box": [0.0, 0.0, 1.0, 1.0], "start": 0.0, "end": 1.0, "enabled": True}]
SNAPREF = {"enabled": True, "images": [REF], "strength": 0.8, "mode": "cs", "adapter": ""}
LLLITE = {"enabled": True, "image": CTL, "model": LLLITE_MODEL,
          "strength": 1.0, "aspect": "letterbox", "start_percent": 0.0, "end_percent": 1.0}

MODES = [
    ("plain", {}),
    ("regional", {"regional": REGIONAL}),
    ("snapref", {"snapref": SNAPREF}),
    ("lllite", {"lllite": LLLITE}),
    ("regional+snapref", {"regional": REGIONAL, "snapref": SNAPREF}),
    ("lllite+snapref", {"lllite": LLLITE, "snapref": SNAPREF}),
    ("regional+lllite", {"regional": REGIONAL, "lllite": LLLITE}),   # must be refused
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_mode(name, extra, timeout=1500):
    gen_id = f"cpu-{name.replace('+', '-')}-{int(time.time())}"
    req = dict(BASE, gen_id=gen_id, **extra)

    if os.path.exists(PROG):
        os.remove(PROG)
    with open(REQ, "w") as f:
        json.dump(req, f)

    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        time.sleep(5)
        try:
            d = json.load(open(PROG))
        except Exception:
            continue
        if d.get("gen_id") and d["gen_id"] != gen_id:
            continue                       # stale record from the previous mode
        st = d.get("stage")
        if st != last and st not in ("denoise",):
            log(f"  {name}: {st} {d.get('note', '')}")
            last = st
        if st == "done":
            out = d.get("out")
            return {"mode": name, "ok": True, "sha": sha(out),
                    "out": out, "elapsed": d.get("elapsed_s"),
                    "denoise": d.get("denoise_s"), "decode": d.get("decode_s"),
                    "reported_mode": d.get("mode")}
        if st == "error":
            return {"mode": name, "ok": False, "error": str(d.get("error"))[:220]}
    return {"mode": name, "ok": False, "error": f"timeout after {timeout}s"}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for p in (CTL, REF):
        if not os.path.exists(p):
            log(f"MISSING asset {p}")
            return 2

    results = []
    for name, extra in MODES:
        log(f"--- {name}")
        r = run_mode(name, extra)
        results.append(r)
        log(f"  -> {json.dumps(r)[:240]}")

    log("")
    log("=== MODE MATRIX ===")
    for r in results:
        if r["ok"]:
            log(f"  {r['mode']:18s} OK   sha={r['sha']}  {r['elapsed']}s "
                f"(denoise {r['denoise']}s, decode {r['decode']}s)")
        else:
            log(f"  {r['mode']:18s} FAIL {r['error']}")

    with open(f"{CONTENT}/mode_matrix.json", "w") as f:
        json.dump(results, f, indent=2)

    shas = {r["mode"]: r.get("sha") for r in results if r.get("ok")}
    log("")
    log(f"distinct images among successful modes: {len(set(shas.values()))}")
    if len(shas) > 1 and len(set(shas.values())) == 1:
        log("WARNING: every mode produced an identical image -- controls are inert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
