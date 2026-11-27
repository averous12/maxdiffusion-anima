"""The XLA cache sync must be safe to run blindly.

It runs at boot, before the server exists, and its failure mode must be "start cold", never a
lost boot. It also must never upload lock files: a restored stale lock is worse than no cache.

Network and token are faked here on purpose -- this checks what the code decides to do, not
whether HuggingFace is reachable.

Run from the workspace root.
"""
import os
import sys
import tempfile
import types

sys.path.insert(0, "maxdiffusion/scripts")


def fake_hf(uploads, downloads=None, list_error=None, create_error=None):
    """A huggingface_hub stand-in that records calls instead of making them."""
    mod = types.ModuleType("huggingface_hub")

    class HfApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_files(self, repo_id, repo_type=None):
            if list_error:
                raise list_error
            return downloads or []

        def create_repo(self, repo_id, repo_type=None, exist_ok=True, private=True):
            if create_error:
                raise create_error
            return repo_id

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None,
                        commit_message=None):
            uploads.append(path_in_repo)

    def hf_hub_download(repo_id, filename, repo_type=None, token=None):
        raise RuntimeError("not used in this test")

    mod.HfApi, mod.hf_hub_download = HfApi, hf_hub_download
    return mod


def main():
    import anima_xla_cache_hf as xc

    fails = []

    # --- skip rules
    skip_ok = (xc._skip("a.lock") and xc._skip("b.tmp") and xc._skip("c.part")
               and not xc._skip("jit_xla_module.bin") and not xc._skip("xla_proto.txt"))
    print(f"{'PASS' if skip_ok else 'FAIL'}  locks/temp skipped, real cache files kept")
    if not skip_ok:
        fails.append("skip rules")

    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "xla_cache")
        os.makedirs(os.path.join(cache, "abc", "def"))
        for rel, data in [("abc/def/jit_a.bin", b"x" * 64),
                          ("abc/jit_b.bin", b"y" * 32),
                          ("abc/jit_a.bin.lock", b"lock"),
                          (".cache/huggingface/meta.json", b"meta")]:
            p = os.path.join(cache, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(data)

        files = xc._walk_files(cache)
        names = [r for r, _ in files]
        walk_ok = names == ["abc/def/jit_a.bin", "abc/jit_b.bin"]
        print(f"{'PASS' if walk_ok else 'FAIL'}  walk skips locks and .cache: {names}")
        if not walk_ok:
            fails.append("walk")

        # --- no token: pull and push are no-ops, not errors
        n = xc.pull("user/repo", os.path.join(tmp, "cold"), token=None, log=lambda *a: None)
        up, total = xc.push("user/repo", cache, token=None, log=lambda *a: None)
        notoken_ok = n == 0 and up == 0 and total == 0
        print(f"{'PASS' if notoken_ok else 'FAIL'}  no token -> does nothing, raises nothing")
        if notoken_ok is False:
            fails.append("no token")

        # --- missing repo on pull: cold start, not a crash
        sys.modules["huggingface_hub"] = fake_hf([], list_error=RuntimeError("404 repo not found"))
        n = xc.pull("user/repo", os.path.join(tmp, "cold2"), token="hf_fake", log=lambda *a: None)
        cold_ok = n == 0
        print(f"{'PASS' if cold_ok else 'FAIL'}  unreadable repo -> 0 entries, no raise")
        if not cold_ok:
            fails.append("cold start")

        # --- push uploads only real files, once, then skips unchanged ones
        uploads = []
        sys.modules["huggingface_hub"] = fake_hf(uploads)
        state = os.path.join(tmp, "state.json")
        up1, b1 = xc.push("user/repo", cache, token="hf_fake", state_path=state,
                          log=lambda *a: None)
        first_ok = sorted(uploads) == ["abc/def/jit_a.bin", "abc/jit_b.bin"] and up1 == 2
        print(f"{'PASS' if first_ok else 'FAIL'}  push uploads {up1} file(s), {b1} B: {sorted(uploads)}")
        if not first_ok:
            fails.append("first push")

        uploads.clear()
        up2, _ = xc.push("user/repo", cache, token="hf_fake", state_path=state,
                         log=lambda *a: None)
        second_ok = up2 == 0 and not uploads
        print(f"{'PASS' if second_ok else 'FAIL'}  second push re-uploads nothing ({up2})")
        if not second_ok:
            fails.append("incremental push")

        # --- a new entry is picked up
        open(os.path.join(cache, "abc", "jit_c.bin"), "wb").write(b"z" * 16)
        up3, _ = xc.push("user/repo", cache, token="hf_fake", state_path=state,
                         log=lambda *a: None)
        new_ok = up3 == 1 and uploads == ["abc/jit_c.bin"]
        print(f"{'PASS' if new_ok else 'FAIL'}  new cache entry uploaded: {uploads}")
        if not new_ok:
            fails.append("new entry")

        # --- repo creation failure must not raise
        sys.modules["huggingface_hub"] = fake_hf([], create_error=RuntimeError("403 no write"))
        up4, _ = xc.push("user/repo", cache, token="hf_fake", state_path=state,
                         log=lambda *a: None)
        create_ok = up4 == 0
        print(f"{'PASS' if create_ok else 'FAIL'}  unwritable repo -> 0 uploads, no raise")
        if not create_ok:
            fails.append("create failure")

    print()
    print("RESULT: XLA HF CACHE OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
