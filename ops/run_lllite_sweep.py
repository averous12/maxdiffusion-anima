"""Sweep LLLite strengths through the repo's own TPU driver.

There is no LLLite-specific driver: `scripts/anima_aesthetic_tpu.py` carries the LLLite
path behind `LLLITE_CKPT`, so a sweep is just the same script invoked with different env.
That keeps one runtime and means a fix to the driver cannot go stale in a fork.

Strength 0.0 is included by default because it is the control condition -- without a pass
that is guaranteed inert (the hook is bit-exact at 0) there is nothing to attribute a
change to.

Prints a sha256 per pass; identical digests across strengths mean the control did nothing.
"""
import hashlib
import os
import subprocess
import sys
import time

REPO = "/content/maxdiffusion"
DRIVER = f"{REPO}/scripts/anima_aesthetic_tpu.py"
LOG = "/content/anima_run.log"
STRENGTHS = [s.strip() for s in os.environ.get("LLLITE_STRENGTHS", "0.0,2.0,4.0").split(",")]
CKPT = os.environ.get("LLLITE_CKPT", "/content/anima_3dpose_6200.safetensors")
CONTROL = os.environ.get("LLLITE_CONTROL", "/content/pose4_control.png")
ASPECT = os.environ.get("LLLITE_ASPECT", "letterbox")
PROMPT = os.environ.get(
  "ANIMA_PROMPT",
  "masterpiece, best quality, 4girls, multiple girls, bikini, swimsuit, beach, ocean, "
  "blue sky, clouds, summer, sunlight, day, outdoors, looking at viewer")


def log(msg):
  line = f"[{time.strftime('%H:%M:%S')}] {msg}"
  print(line, flush=True)
  with open(LOG, "a") as f:
    f.write(line + "\n")


def digest(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()[:16]


def main():
  if not os.path.exists(CKPT):
    log(f"MISSING checkpoint {CKPT}")
    return 2
  if not os.path.exists(CONTROL):
    log(f"MISSING control {CONTROL}")
    return 2
  log(f"sweep strengths {STRENGTHS} | control {CONTROL} | aspect {ASPECT}")

  results = []
  for s in STRENGTHS:
    out = "anima_lllite_s" + s.replace(".", "")
    log(f"=== strength {s} -> /content/{out}.png ===")
    env = {**os.environ, "PYTHONPATH": f"{REPO}/src", "HF_HOME": "/content/hf_cache",
           "PYTHONUNBUFFERED": "1", "ANIMA_STEPS": os.environ.get("ANIMA_STEPS", "20"),
           "ANIMA_PROMPT": PROMPT, "ANIMA_OUT": out,
           "LLLITE_CKPT": CKPT, "LLLITE_CONTROL": CONTROL, "LLLITE_ASPECT": ASPECT,
           "LLLITE_STRENGTH": s}
    t0 = time.perf_counter()
    p = subprocess.Popen([f"{REPO}/.venv/bin/python", DRIVER], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
      line = line.rstrip()
      if line:
        log(line)
    p.wait()
    path = f"/content/{out}.png"
    if not os.path.exists(path):
      log(f"IMAGE MISSING {path} (driver exit={p.returncode})")
      results.append((s, None, None))
      continue
    d = digest(path)
    log(f"strength {s}: {os.path.getsize(path)} B sha256:{d} in {time.perf_counter()-t0:.1f}s")
    results.append((s, d, os.path.getsize(path)))

  log("=== SUMMARY (identical sha256 == the control did nothing) ===")
  for s, d, n in results:
    log(f"  strength {s:>4}: {'MISSING' if d is None else f'{d}  {n} B'}")
  seen = [d for _, d, _ in results if d]
  log(f"distinct images across {len(results)} passes: {len(set(seen))}")
  log("SWEEP DONE")
  return 0


if __name__ == "__main__":
  sys.exit(main())
