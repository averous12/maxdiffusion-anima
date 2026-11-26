"""A/B pass: rerun the LLLite driver at a different strength, same seed and prompt.

Exists to answer one question with evidence -- does the control actually steer the image,
or does it merely run? At strength 0 the hook is a proven bit-exact no-op, so any difference
between this image and the strength-1.0 image is the control's doing, and sameness means the
control is being ignored.

  LLLITE_STRENGTH=0.0 LLLITE_OUT=anima_lllite_s0 python run_lllite_strength_ab.py
"""
import os
import subprocess
import sys
import time

REPO = "/content/maxdiffusion"
DRIVER = f"{REPO}/scripts/anima_lllite_gen_tpu.py"
LOG = "/content/anima_run.log"
STRENGTH = os.environ.get("LLLITE_STRENGTH", "0.0")
OUT = os.environ.get("LLLITE_OUT", "anima_lllite_s0")


def log(msg):
  line = f"[{time.strftime('%H:%M:%S')}] {msg}"
  print(line, flush=True)
  with open(LOG, "a") as f:
    f.write(line + "\n")


def main():
  log(f"=== A/B: strength {STRENGTH} -> /content/{OUT}.png ===")
  env = {**os.environ,
         "PYTHONPATH": f"{REPO}/src",
         "HF_HOME": "/content/hf_cache",
         "PYTHONUNBUFFERED": "1",
         "LLLITE_STRENGTH": STRENGTH,
         "LLLITE_OUT": OUT}
  t0 = time.time()
  p = subprocess.Popen([f"{REPO}/.venv/bin/python", DRIVER], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  for line in p.stdout:
    line = line.rstrip()
    if line:
      log(line)
  p.wait()
  log(f"driver exit={p.returncode} in {time.time() - t0:.1f}s")
  img = f"/content/{OUT}.png"
  log(f"IMAGE OK {img} {os.path.getsize(img)} B" if os.path.exists(img)
      else f"IMAGE MISSING {img}")
  log("AB DONE")
  return p.returncode


if __name__ == "__main__":
  sys.exit(main())
