"""Run the LLLite driver at several strengths in one session, distinct output per strength.

The first A/B compared 1.0 against 0.0 and got a bit-identical result, which is stronger
evidence than a weak effect but could in principle be a strength that never reaches the
maths. Exaggerating the strength removes that doubt: if 2.0 and 4.0 also reproduce the
strength-0 image exactly, the hook contributes nothing on TPU at any strength.

Writes /content/anima_lllite_s<NN>.png for each strength and a summary line per pass.
"""
import hashlib
import os
import subprocess
import sys
import time

REPO = "/content/maxdiffusion"
DRIVER = f"{REPO}/scripts/anima_lllite_gen_tpu.py"
LOG = "/content/anima_run.log"
STRENGTHS = [s.strip() for s in os.environ.get("LLLITE_STRENGTHS", "2.0,4.0").split(",")]


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
  results = []
  for s in STRENGTHS:
    tag = "s" + s.replace(".", "")
    out = f"anima_lllite_{tag}"
    log(f"=== strength {s} -> /content/{out}.png ===")
    env = {**os.environ, "PYTHONPATH": f"{REPO}/src", "HF_HOME": "/content/hf_cache",
           "PYTHONUNBUFFERED": "1", "LLLITE_STRENGTH": s, "LLLITE_OUT": out}
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
    log(f"strength {s}: {os.path.getsize(path)} B sha256:{d} "
        f"in {time.perf_counter() - t0:.1f}s")
    results.append((s, d, os.path.getsize(path)))

  log("=== SUMMARY (identical sha256 == the control did nothing) ===")
  for s, d, n in results:
    log(f"  strength {s:>4}: {'MISSING' if d is None else f'{d}  {n} B'}")
  distinct = {d for _, d, _ in results if d}
  if os.path.exists("/content/anima_lllite.png"):
    base = digest("/content/anima_lllite.png")
    log(f"  reference strength 0.0 (anima_lllite.png): {base}")
    distinct |= {base}
  log(f"distinct images across all strengths: {len(distinct)}")
  log("TWO-STRENGTH RUN DONE")
  return 0


if __name__ == "__main__":
  sys.exit(main())
