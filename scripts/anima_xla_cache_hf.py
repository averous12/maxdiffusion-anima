"""Keep the XLA compile cache in a HuggingFace repo instead of on Drive.

Why: the first compile at a given shape is the expensive part of this pipeline (the VAE decode
alone was 48.5 s at 1024 px, the transformer 16 s). XLA memoises those executables on disk, but
a Colab VM is wiped on every reconnect, so the cache only pays off within one session unless it
is stored somewhere durable.

Drive is the obvious place and the wrong one for this: every file read from the mounted FUSE
volume is slow, and XLA stats and mmaps its cache directory on every compile. A local directory
plus a copy in a HF repo keeps the hot path on local disk and the durability off-box.

Shape of the thing:
  * the cache directory is write-once -- an entry is named by a hash and never edited, so a new
    file is the only kind of change there is. That makes "upload what is not already up there"
    both safe and cheap.
  * XLA also writes lock files, which must NOT be uploaded; a stale lock in a restored cache is
    worse than a cold one.
  * the repo is a plain dataset repo. Nothing about it is HF-specific beyond that; it is just
    blob storage with a token.

Usage:
    python scripts/anima_xla_cache_hf.py pull <repo_id> <dest_dir>
    python scripts/anima_xla_cache_hf.py push <repo_id> <src_dir> [--state FILE]

HF_TOKEN comes from the environment. A missing token, an absent repo or a network failure on
pull is reported and treated as "start cold" -- a cache is an optimisation and must never stop
the server from booting. On push, failures are reported but never raise either: the run is
already serving by then.
"""
import argparse
import json
import os
import shutil
import sys

SKIP_SUFFIXES = (".lock", ".tmp", ".part")
STATE_NAME = "xla_cache_upload_state.json"


def _skip(name):
    return name.endswith(SKIP_SUFFIXES) or name.startswith(".cache")


def _walk_files(root):
    """Every cache file, relative to root, skipping locks and HF's own metadata dir."""
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if _skip(f):
                continue
            p = os.path.join(base, f)
            out.append((os.path.relpath(p, root).replace("\\", "/"), p))
    return sorted(out)


def pull(repo_id, dest, token=None, log=print):
    """Copy any cache entries the repo has that `dest` does not. Returns files copied."""
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(dest, exist_ok=True)
    if not token:
        log("xla cache: no HF token, starting cold")
        return 0
    try:
        files = [f for f in HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
                 if not _skip(f)]
    except Exception as e:
        log(f"xla cache: {repo_id} not readable ({type(e).__name__}: {e}); starting cold")
        return 0

    got = 0
    for rel in files:
        target = os.path.join(dest, rel)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            continue
        try:
            src = hf_hub_download(repo_id=repo_id, filename=rel,
                                  repo_type="dataset", token=token)
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            shutil.copyfile(src, target)
            got += 1
        except Exception as e:
            log(f"xla cache: skipping {rel} ({type(e).__name__}: {e})")
    log(f"xla cache: pulled {got} of {len(files)} entries from {repo_id} into {dest}")
    return got


def push(repo_id, src, token=None, state_path=None, log=print):
    """Upload cache entries not already uploaded. Returns (files_uploaded, bytes_uploaded)."""
    from huggingface_hub import HfApi

    if not token:
        log("xla cache: no HF token, not uploading")
        return 0, 0
    if not os.path.isdir(src):
        log(f"xla cache: {src} does not exist, nothing to upload")
        return 0, 0

    state_path = state_path or os.path.join(src, STATE_NAME)
    try:
        with open(state_path) as f:
            state = json.load(f).get("files", {})
    except Exception:
        state = {}

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
    except Exception as e:
        log(f"xla cache: could not ensure repo {repo_id} ({type(e).__name__}: {e})")
        return 0, 0

    files = _walk_files(src)
    n, total = 0, 0
    for rel, path in files:
        if rel == STATE_NAME:
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        # A cache entry is immutable, so size+mtime identifies its content well enough to skip
        # re-uploading, and re-uploading a stray partial is the only cost of being wrong.
        key = f"{st.st_size}:{int(st.st_mtime)}"
        if state.get(rel) == key:
            continue
        try:
            api.upload_file(path_or_fileobj=path, path_in_repo=rel, repo_id=repo_id,
                            repo_type="dataset",
                            commit_message=f"xla cache: {rel}")
            state[rel] = key
            n += 1
            total += st.st_size
        except Exception as e:
            log(f"xla cache: upload failed for {rel} ({type(e).__name__}: {e})")

    try:
        with open(state_path, "w") as f:
            json.dump({"repo": repo_id, "files": state}, f, indent=1)
    except Exception:
        pass
    log(f"xla cache: uploaded {n} file(s), {total / 1e6:.1f} MB to {repo_id} "
        f"({len(files)} in cache)")
    return n, total


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sync the XLA cache with a HF dataset repo")
    ap.add_argument("action", choices=["pull", "push"])
    ap.add_argument("repo_id")
    ap.add_argument("path")
    ap.add_argument("--state", default=None)
    args = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN", "").strip()
    if args.action == "pull":
        pull(args.repo_id, args.path, token=token)
    else:
        push(args.repo_id, args.path, token=token, state_path=args.state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
