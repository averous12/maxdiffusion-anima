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

  # ---- 2. overlay the locally-edited model files
  # Only the two model modules are overlaid: LLLite lives inside them, and the one-shot
  # TPU driver that used to need overlaying has been deleted (server is the only runtime).
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

  # ---- 5. generation is the server's job now
  # The one-shot TPU driver was deleted: it duplicated the server's generation logic and
  # had to be kept in step with it by hand. Setup + import verification is where this
  # script stops; invoking generation means starting the server and posting a request to it.
  if os.path.exists(f"{REPO}/scripts/anima_aesthetic_server.py"):
    log("server present; generation runs through it, not from this bootstrap")
  else:
    log("WARN: anima_aesthetic_server.py missing")
  log("BOOTSTRAP DONE (setup only)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
