"""Bootstrap the TPU session and run a real aesthetic generation.

Clones the fork rather than uploading 23 MB, overlays the local (uncommitted) LLLite work,
builds the venv with the project's own setup script, then runs the aesthetic driver.
Every stage prints and is time-boxed so a failure names itself instead of hanging.
"""
import os
import shutil
import subprocess
import sys
import time

REPO = "/content/maxdiffusion"
OVERLAY = "/content"            # flat: `colab upload` cannot create nested dirs
VENV_PY = f"{REPO}/.venv/bin/python"


def log(msg):
  print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(argv, cwd=None, env=None, timeout=3000, tail=30):
  log("$ " + " ".join(argv))
  t0 = time.perf_counter()
  p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=timeout)
  out = (p.stdout or b"").decode("utf-8", "replace") + (p.stderr or b"").decode("utf-8", "replace")
  dt = time.perf_counter() - t0
  for line in out.strip().splitlines()[-tail:]:
    print("  " + line, flush=True)
  log(f"exit={p.returncode} {dt:.1f}s")
  return p.returncode, out


def main():
  # ---- 1. code
  if os.path.isdir(f"{REPO}/.git"):
    run(["git", "fetch", "--depth", "1", "origin"], cwd=REPO)
    run(["git", "reset", "--hard", "origin/HEAD"], cwd=REPO)
  else:
    run(["git", "clone", "--depth", "1",
         "https://github.com/averous12/maxdiffusion-anima.git", REPO], timeout=900)
  run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, tail=2)

  # ---- 2. overlay the local LLLite files (the driver is NOT overlaid: LLLite lives in
  # the repo's own anima_aesthetic_tpu.py now, so there is no fork to substitute)
  for name, sub in (("anima_lllite_flax.py", "src/maxdiffusion/models"),
                    ("anima_cosmos_flax.py", "src/maxdiffusion/models")):
    src = os.path.join(OVERLAY, name)
    dst = f"{REPO}/{sub}/{name}"
    if not os.path.exists(src):
      log(f"MISSING overlay {src}")
      return 2
    shutil.copyfile(src, dst)
    log(f"overlaid {sub}/{name} ({os.path.getsize(dst)} B)")
  rc, out = run(["grep", "-c", "_lllite", f"{REPO}/src/maxdiffusion/models/anima_cosmos_flax.py"],
                tail=2)
  if rc != 0:
    log("FATAL: the cosmos transformer has no lllite hook — overlay did not take")
    return 3

  # ---- 3. venv + deps
  env = {k: v for k, v in os.environ.items()
         if k not in ("TPU_LIBRARY_PATH", "ENABLE_PJRT_COMPATIBILITY", "UV_SYSTEM_PYTHON")}
  env["PATH"] = "/usr/local/bin:" + env.get("PATH", "")
  rc, _ = run([sys.executable, f"{REPO}/scripts/anima_colab_setup.py"], env=env, timeout=3600)
  if not os.path.exists(VENV_PY):
    log("FATAL: venv python missing after setup")
    return 4

  # ---- 4. verify the overlay imports inside the real venv
  check = (
      "import jax\n"
      "print('devices:', jax.devices())\n"
      "from maxdiffusion.models import anima_lllite_flax as LL\n"
      "from maxdiffusion.models import anima_snapref_flax as SR\n"
      "import inspect\n"
      "w = LL.WorkflowChain()\n"
      "print('lllite strength', w.lllite_strength, 'steps', w.steps)\n"
      "print('snapref layers', inspect.signature(SR.load_snapref)"
      ".parameters['num_layers'].default)\n"
  )
  rc, _ = run([VENV_PY, "-c", check],
              env={**env, "PYTHONPATH": f"{REPO}/src", "HF_HOME": "/content/hf_cache",
                   "PYTHONUNBUFFERED": "1"}, timeout=600)
  if rc != 0:
    log("WARN: overlay import check failed (continuing to generation)")

  # ---- 5. validation run: the repo's own driver, LLLite switched on by env
  # No LLLite-specific driver exists. The sweep runner drives anima_aesthetic_tpu.py with
  # LLLITE_CKPT set, so exactly one runtime is under test.
  script = f"{REPO}/ops/run_lllite_sweep.py"
  if os.path.exists(script):
    log("running the LLLite strength sweep (repo driver + env)")
  else:
    script = f"{REPO}/scripts/anima_aesthetic_tpu.py"
    log("sweep runner absent; running the plain aesthetic driver")
  rc, out = run([VENV_PY, script],
                env={**env, "PYTHONPATH": f"{REPO}/src", "HF_HOME": "/content/hf_cache",
                     "PYTHONUNBUFFERED": "1"}, timeout=5400, tail=45)
  found = False
  for img in ("/content/anima_lllite_s0.png", "/content/anima_lllite_s20.png",
              "/content/anima_lllite_s40.png", "/content/anima_perstep.png"):
    if os.path.exists(img):
      found = True
      log(f"IMAGE OK {img} {os.path.getsize(img)} B")
  if not found:
    log("IMAGE MISSING (no output produced)")
  log("BOOTSTRAP DONE")
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
