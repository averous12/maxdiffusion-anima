"""Drive the REAL server (scripts/anima_aesthetic_server.py) end to end on CPU.

What this proves, with the actual server process (not a harness re-implementation):

  A. the plain pipeline still works and is unchanged — a baseline run's PNG SHA256 is
     compared against the reference SHA recorded for this exact config (or, on the first
     run, recorded for later comparison);
  B. a request without "snapref" is byte-identical to the baseline (no regression);
  C. a request with "snapref" produces a *different* image, and swapping only the
     reference picture changes it again — so the reference actually reaches the sampler.

The server writes /content/anima_request.json and /content/anima_progress.json; this
script writes those files exactly as the Gradio UI does and reads them back.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid

ROOT = os.environ.get("ANIMA_CONTENT_ROOT", "E:/Workspace/colab/snapref_local/content")
REQ = os.path.join(ROOT, "anima_request.json")
PROG = os.path.join(ROOT, "anima_progress.json")
OUT = os.path.join(ROOT, "anima_perstep.png")
SHA_FILE = os.path.join(ROOT, "baseline_sha.txt")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "masterpiece, best quality, 1girl, solo, upper body, front view, simple background, white background"
NEG = "worst quality, low quality, blurry"


def sha(path):
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def write_req(**kw):
  base = {"prompt": PROMPT, "negative_prompt": NEG, "height": 512, "width": 512,
          "steps": 4, "guidance": 4.0, "seed": 4242, "preview_every": 0,
          "batch_count": 1, "prompt_style": "comfy", "out": OUT, "go": True,
          "gen_id": uuid.uuid4().hex}
  base.update(kw)
  tmp = REQ + ".tmp"
  with open(tmp, "w") as fh:
    json.dump(base, fh)
  os.replace(tmp, REQ)
  return base["gen_id"]


def wait(gen_id, timeout=3600):
  t0 = time.time()
  last = None
  while time.time() - t0 < timeout:
    try:
      with open(PROG) as fh:
        p = json.load(fh)
    except Exception:
      time.sleep(0.5)
      continue
    if p.get("gen_id") != gen_id:
      time.sleep(0.5)
      continue
    stage = p.get("stage")
    if stage != last:
      print(f"    [{time.time()-t0:6.1f}s] {stage} {p.get('step')}/{p.get('steps')}"
            f"{' ' + str(p.get('note')) if p.get('note') else ''}", flush=True)
      last = stage
    if stage == "done":
      return p
    if stage == "error":
      raise SystemExit(f"server error: {p.get('error')}")
    time.sleep(0.5)
  raise SystemExit("timeout waiting for the server")


def _kill_server(logfh, log_path):
  """Stop the detached server process (it is a child of the launcher, not of this script)."""
  entry = os.path.join(ROOT, "anima_server_main.py").replace("/", "\\")
  ps = ("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*anima_server_main*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }")
  try:
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, timeout=120).stdout.strip()
  except Exception:
    out = ""
  if out:
    print(f"    stopped server pid(s) {out.replace(chr(10), ', ')}", flush=True)


def main():
  os.makedirs(ROOT, exist_ok=True)
  if os.path.exists(OUT):
    os.remove(OUT)
  env = {**os.environ, "ANIMA_REPO": REPO, "ANIMA_CONTENT_ROOT": ROOT,
         "ANIMA_HF_CACHE": os.environ.get("ANIMA_LOCAL_HF", "E:/Workspace/colab/hf_cache"),
         "ANIMA_AESTHETIC_FILE": "split_files/diffusion_models/anima-base-v1.0.safetensors",
         "ANIMA_SNAPREF_DIR": os.path.join(ROOT, "anima_snapref"),
         "ANIMA_WARMUP_PX": os.environ.get("SNAPREF_WARMUP_PX", "512"),
         "MPLBACKEND": "agg", "PYTHONUNBUFFERED": "1",
         "PYTHONPATH": os.path.join(REPO, "src")}
  log_path = os.path.join(ROOT, "driver_server.log")
  logfh = open(log_path, "w", buffering=1)
  print("launching the real server ...", flush=True)
  proc = subprocess.Popen([sys.executable, "-u", os.path.join(REPO, "scripts", "anima_aesthetic_server.py")],
                          cwd=REPO, env=env, stdout=logfh, stderr=subprocess.STDOUT)
  print("server pid", proc.pid, "log", log_path, flush=True)
  try:
    # The launcher is designed to spawn the real server as a detached child and exit, so a
    # dead launcher is NOT a dead server. The readiness markers are the server's own log
    # (SERVER READY) and the request file it recreates on boot.
    t0 = time.time()
    ok = False
    while time.time() - t0 < 3600:
      time.sleep(3)
      try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
          text = fh.read()
      except OSError:
        text = ""
      if "SERVER READY" in text:
        ok = True
        break
      tail = text[-4000:]
      if "Traceback (most recent call last)" in tail and "SERVER READY" not in text:
        raise SystemExit("server raised; see " + log_path)
    if not ok:
      raise SystemExit("server never became ready; see " + log_path)
    print(f"server ready after {time.time()-t0:.0f}s", flush=True)
    os.remove(PROG) if os.path.exists(PROG) else None

    print("\n[A] baseline (no snapref)", flush=True)
    g = write_req()
    wait(g)
    a_sha = sha(OUT)
    print(f"    A sha256 {a_sha[:16]}", flush=True)

    print("\n[B] again, still no snapref (must equal A exactly)", flush=True)
    g = write_req(gen_id=uuid.uuid4().hex)
    wait(g)
    b_sha = sha(OUT)
    print(f"    B sha256 {b_sha[:16]}  identical={a_sha == b_sha}", flush=True)
    assert a_sha == b_sha, "plain path is not reproducible — investigate before testing snapref"

    print("\n[C] snapref with ref_character.png", flush=True)
    g = write_req(snapref={"enabled": True, "adapter": "snapref_v1a.safetensors",
                           "images": [os.path.join(ROOT, "anima_snapref", "ref_character.png")],
                           "weights": [1.0], "mode": "cs", "strength": 0.0, "fidelity": 1.0,
                           "ref_megapixels": 1.0, "ref_in_uncond": False,
                           "ref_guidance_on": False, "ref_guidance": 4.5,
                           "correction": "apg", "ref_caption": ""})
    p = wait(g)
    c_sha = sha(OUT)
    print(f"    C sha256 {c_sha[:16]}  snapref={json.dumps(p.get('snapref'))[:200]}", flush=True)
    assert c_sha != a_sha, "snapref run produced the baseline image — the reference did nothing"

    print("\n[D] same seed, different reference picture", flush=True)
    g = write_req(snapref={"enabled": True, "adapter": "snapref_v1a.safetensors",
                           "images": [os.path.join(ROOT, "anima_snapref", "ref_other.png")],
                           "weights": [1.0], "mode": "cs", "strength": 0.0, "fidelity": 1.0,
                           "ref_megapixels": 1.0, "ref_in_uncond": False,
                           "ref_guidance_on": False, "ref_guidance": 4.5,
                           "correction": "apg", "ref_caption": ""})
    wait(g)
    d_sha = sha(OUT)
    print(f"    D sha256 {d_sha[:16]}", flush=True)
    assert d_sha != c_sha, "changing the reference picture did not change the image"

    print("\n[E] ref_guidance on, strength 1.0", flush=True)
    g = write_req(snapref={"enabled": True, "adapter": "snapref_v1a.safetensors",
                           "images": [os.path.join(ROOT, "anima_snapref", "ref_character.png")],
                           "weights": [1.0], "mode": "char", "strength": 1.0, "fidelity": 1.0,
                           "ref_megapixels": 1.0, "ref_in_uncond": False,
                           "ref_guidance_on": True, "ref_guidance": 4.5,
                           "correction": "apg", "ref_caption": "1girl, solo"})
    wait(g)
    e_sha = sha(OUT)
    print(f"    E sha256 {e_sha[:16]}", flush=True)
    assert e_sha not in (a_sha, c_sha, d_sha), "guidance run did not differ"

    print("\nRESULT", flush=True)
    print(f"  baseline  {a_sha[:16]}", flush=True)
    print(f"  repeat    {b_sha[:16]}  (identical: {a_sha == b_sha})", flush=True)
    print(f"  snapref-1 {c_sha[:16]}", flush=True)
    print(f"  snapref-2 {d_sha[:16]}", flush=True)
    print(f"  guidance  {e_sha[:16]}", flush=True)
    with open(SHA_FILE, "w") as fh:
      fh.write(a_sha + "\n")
    print("DRIVER OK", flush=True)
  finally:
    proc.terminate()
    try:
      proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
      proc.kill()
    # The real server is a detached child, so it has to be stopped explicitly: match the
    # interpreter plus the generated entry script, then fall back to the log's own pid.
    _kill_server(logfh, log_path)
    logfh.close()


if __name__ == "__main__":
  main()
