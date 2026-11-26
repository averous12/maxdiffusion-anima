"""Fetch the checkpoints on the VM from Hugging Face and verify their hashes.

Uploading 380 MB from a laptop when the same bytes sit on HF is a waste of the link -- the VM
should pull them. Each file is hash-checked against the value computed from the local copy, so
"downloaded" is not mistaken for "the right file".
"""
import hashlib
import os
import sys
import time

CONTENT = "/content"

WANT = [
    # (repo_id, filename, destination, expected sha256 prefix)
    ("0ppxnhximxr/snapref-v2a", "snapref_v2a.safetensors",
     f"{CONTENT}/snapref_v2a.safetensors", "8a22fc28d060447544eda283"),
    ("toyxyz/Anima_ControlNet-LLLite", "anima_3dpose_6200.safetensors",
     f"{CONTENT}/anima_3dpose_6200.safetensors", "97542346a658d23ca65368b5"),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from huggingface_hub import hf_hub_download

    ok = True
    for repo, fname, dest, expect in WANT:
        if os.path.exists(dest) and sha(dest).startswith(expect):
            log(f"{fname}: already present and correct")
            continue
        log(f"{repo}/{fname} -> {dest}")
        try:
            # local_dir makes HF place the blob at /content/<filename> directly, so there is
            # no copy step to get wrong
            hf_hub_download(repo_id=repo, filename=fname, repo_type="model",
                            local_dir=CONTENT)
        except Exception as e:
            log(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")
            ok = False
            continue
        got = sha(dest) if os.path.exists(dest) else ""
        match = got.startswith(expect)
        log(f"  size={os.path.getsize(dest)} sha={got[:24]} {'OK' if match else 'MISMATCH'}")
        ok = ok and match

    log("FETCH OK" if ok else "FETCH FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
