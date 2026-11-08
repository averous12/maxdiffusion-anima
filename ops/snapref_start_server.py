"""Start the REAL server for the UI proof (CPU, 512px warmup, 0.25 MP references).

Same server script the Colab VM runs; only the content root, the weight cache, the
checkpoint file and the warmup shape come from the environment.
"""
import os
import subprocess
import sys
import time

ROOT = os.environ.get("ANIMA_CONTENT_ROOT", "E:/Workspace/colab/snapref_local/content")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "ui_server.log")

env = {**os.environ, "ANIMA_REPO": REPO, "ANIMA_CONTENT_ROOT": ROOT,
       "ANIMA_HF_CACHE": os.environ.get("ANIMA_LOCAL_HF", "E:/Workspace/colab/hf_cache"),
       "ANIMA_AESTHETIC_FILE": "split_files/diffusion_models/anima-base-v1.0.safetensors",
       "ANIMA_SNAPREF_DIR": os.path.join(ROOT, "anima_snapref"),
       "ANIMA_WARMUP_PX": os.environ.get("ANIMA_WARMUP_PX", "512"),
       "MPLBACKEND": "agg", "PYTHONUNBUFFERED": "1",
       "PYTHONPATH": os.path.join(REPO, "src")}

os.makedirs(ROOT, exist_ok=True)
logfh = open(LOG, "w", buffering=1)
proc = subprocess.Popen([sys.executable, "-u", os.path.join(REPO, "scripts", "anima_aesthetic_server.py")],
                        cwd=REPO, env=env, stdout=logfh, stderr=subprocess.STDOUT)
print("launcher pid", proc.pid, "log", LOG, flush=True)
t0 = time.time()
while time.time() - t0 < 2400:
  time.sleep(5)
  try:
    text = open(LOG, encoding="utf-8", errors="replace").read()
  except OSError:
    continue
  if "SERVER READY" in text:
    print(f"SERVER_READY after {time.time()-t0:.0f}s", flush=True)
    sys.exit(0)
  if "Traceback (most recent call last)" in text[-3000:] and "SERVER READY" not in text:
    print("SERVER_RAISED", flush=True)
    sys.exit(2)
print("SERVER_TIMEOUT", flush=True)
sys.exit(3)
