"""The named-tunnel argv must actually parse in cloudflared.

`--token` belongs to the `run` subcommand. Passing it to `tunnel` makes cloudflared answer
"Incorrect Usage. flag provided but not defined: -token" and exit, so the connector never
starts -- and from the outside that looks like "the tunnel is dead / expired, run it again".

The check below is empirical, not a string comparison: it runs the real binary with a bogus
token and reads the verdict. A valid parse rejects the TOKEN ("Tunnel token is not valid"); an
invalid parse rejects a FLAG ("flag provided but not defined"). It also runs the old buggy form
and requires it to fail, so the test is proven to discriminate.

Run from the workspace root.
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "maxdiffusion/scripts")


def find_cloudflared():
    for cand in ("cloudflared", "/usr/local/bin/cloudflared",
                 "/c/Users/avero/bin/cloudflared.exe", "/c/Users/avero/bin/cloudflared"):
        p = shutil.which(cand) or (cand if os.path.exists(cand) else None)
        if p:
            return p
    return None


def run(argv, timeout=20):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "<timeout>"


def main():
    import anima_cloudflared as cf

    fails = []
    argv = cf.named_argv("/usr/local/bin/cloudflared", "abc123")
    shape_ok = argv[:2] == ["/usr/local/bin/cloudflared", "tunnel"] and argv[2:4] == [
        "--no-autoupdate", "run"] and argv[4] == "--token"
    print(f"{'PASS' if shape_ok else 'FAIL'}  argv shape: {argv[1:]}")
    if not shape_ok:
        fails.append("shape")

    binp = find_cloudflared()
    if not binp:
        print("SKIP  no cloudflared binary on this host -- shape check only")
    else:
        # Real parse, valid order: the flags are accepted, the token is rejected.
        out = run([binp, "tunnel", "--no-autoupdate", "run", "--token", "abc123"])
        parsed = "not defined" not in out and "token is not valid" in out.lower()
        print(f"{'PASS' if parsed else 'FAIL'}  real binary accepts the order "
              f"({binp.split(os.sep)[-1]}): {out.strip().splitlines()[0][:70] if out.strip() else '<no output>'}")
        if not parsed:
            fails.append("valid order")

        # The old form must fail -- otherwise this test proves nothing.
        bad = run([binp, "tunnel", "--no-autoupdate", "--token", "abc123"])
        bad_fails = "not defined" in bad
        print(f"{'PASS' if bad_fails else 'FAIL'}  old form (token on 'tunnel') is rejected, "
              f"so the check discriminates")
        if not bad_fails:
            fails.append("old form")

    print()
    print("RESULT: CLOUDFLARED ARGV OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
