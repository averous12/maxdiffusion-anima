"""One-shot VM setup: clone, install, fetch checkpoints, spawn the server. Then RETURN.

Everything that can take minutes runs here as ordinary subprocess work -- that is fine, because
it terminates. Nothing polls for the server: the previous version waited for `ready` inside the
same exec, which held the Jupyter kernel for 45 minutes and made every later status check queue
forever. Spawning detached and returning keeps the kernel free.
"""
import json
import os
import subprocess
import sys
import time

CONTENT = "/content"
REPO = f"{CONTENT}/maxdiffusion"
URL = "https://github.com/averous12/maxdiffusion-anima.git"
LOG = f"{CONTENT}/srv.log"
PIDFILE = f"{CONTENT}/srv.pid"

CKPTS = [
    ("0ppxnhximxr/snapref-v2a", "snapref_v2a.safetensors"),
    ("toyxyz/Anima_ControlNet-LLLite", "anima_3dpose_6200.safetensors"),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        log(f"  rc={r.returncode}: {(cmd[1:3] if len(cmd) > 3 else cmd)}")
        for line in (r.stderr or "").splitlines()[-6:]:
            log("  ! " + line)
    return r.returncode


def main():
    # 1. repo
    if not os.path.isdir(REPO):
        log("cloning")
        if run(["git", "clone", "--depth", "1", URL, REPO]) != 0:
            return 2
    else:
        log("repo present")

    # 2. packages
    log("pip install (known-good set)")
    rc = run([sys.executable, "-m", "pip", "install", "-q",
              "--extra-index-url", "https://download.pytorch.org/whl/cpu",
              "-r", f"{CONTENT}/cpu_reqs.txt"])
    if rc != 0:
        return 3
    log("packages installed")

    # 3. checkpoints straight from HF -- the VM has the bandwidth, the laptop does not
    from huggingface_hub import hf_hub_download
    for repo, fname in CKPTS:
        dest = f"{CONTENT}/{fname}"
        if os.path.exists(dest):
            log(f"{fname} present")
            continue
        log(f"fetching {repo}/{fname}")
        try:
            hf_hub_download(repo_id=repo, filename=fname, repo_type="model", local_dir=CONTENT)
            log(f"  {os.path.getsize(dest)} B")
        except Exception as e:
            log(f"  FAILED {type(e).__name__}: {str(e)[:160]}")
            return 4

    # 4. spawn, do not wait
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{REPO}/src",
        "PYTHONUNBUFFERED": "1",
        "ANIMA_WARMUP_PX": "128",
        "ANIMA_LLLITE_DIR": CONTENT,
        "ANIMA_SNAPREF_DIR": CONTENT,
    })
    with open(LOG, "a") as f:
        f.write(f"\n===== spawn {time.strftime('%H:%M:%S')} =====\n")
    proc = subprocess.Popen([sys.executable, "-m", "maxdiffusion.runtime.pipeline"],
                            cwd=REPO, env=env, stdout=open(LOG, "a"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    open(PIDFILE, "w").write(str(proc.pid))
    time.sleep(3)
    log(f"spawned pid={proc.pid} alive={proc.poll() is None}")
    log("SETUP DONE -- poll /content/anima_progress.json for stage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
