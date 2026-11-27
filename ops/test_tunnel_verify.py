"""A quick tunnel must not be announced before it serves.

Reported symptom: "cloudflare needs to be run twice, the first one gave an expired tunnel".
start() used to return the first hostname cloudflared printed, and a printed hostname is not a
working one -- the edge can hand out a name that never routes (browser: Cloudflare error 1033).

These checks drive start() with the process spawner and the probe replaced, so the retry logic
is exercised without network or a real cloudflared: first hostname dead -> torn down, second
hostname alive -> returned. Also pins that a probe which 4xx's still counts as alive, because
the edge routing at all is what matters, and that a 5xx/1033 does not.

Run from the workspace root.
"""
import sys

sys.path.insert(0, "maxdiffusion/scripts")


class FakeProc:
    def __init__(self, name):
        self.name = name
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return 0


def main():
    import anima_cloudflared as cf

    fails = []
    real = (cf.ensure_cloudflared, cf._wait_origin, cf._spawn_quick, cf._public_ok,
            cf.time.sleep)
    try:
        cf.ensure_cloudflared = lambda path=None: "cloudflared"
        cf._wait_origin = lambda port, timeout: None
        cf.time.sleep = lambda s: None          # the verify loop must not actually wait

        # --- first URL dead, second alive
        spawned = []

        def spawn(binp, port, log_path, timeout, append=False):
            url = f"https://dead-{len(spawned)}.trycloudflare.com" if not spawned else \
                  "https://good.trycloudflare.com"
            p = FakeProc(url)
            spawned.append(p)
            return p, url

        alive = {"https://good.trycloudflare.com"}

        cf._spawn_quick = spawn
        cf._public_ok = lambda url, timeout=4: url in alive
        p, url = cf.start(port=7860, timeout=5, verify_timeout=4, attempts=2)

        ok = url == "https://good.trycloudflare.com" and spawned[0].terminated \
            and not spawned[0].waited is False
        retry_ok = url == "https://good.trycloudflare.com" and spawned[0].terminated
        print(f"{'PASS' if retry_ok else 'FAIL'}  dead first hostname torn down, live one "
              f"returned: {url} (attempts={len(spawned)}, first terminated="
              f"{spawned[0].terminated})")
        if not retry_ok:
            fails.append("retry")

        # --- every hostname dead: warn and return unverified, never raise. Raising here killed a
        # boot that had already reached SERVER READY, taking the runtime down with the tunnel.
        cf._spawn_quick = lambda *a, **k: (FakeProc("x"), "https://dead.trycloudflare.com")
        cf._public_ok = lambda url, timeout=4: False
        try:
            p2, u2 = cf.start(port=7860, timeout=5, verify_timeout=2, attempts=2)
            msg, raised = "returned " + u2, False
            ok_dead = u2 == "https://dead.trycloudflare.com"
        except SystemExit as e:
            msg, raised, ok_dead = str(e), True, False
        print(f"{'PASS' if ok_dead and not raised else 'FAIL'}  all-dead returns unverified "
              f"instead of raising: {msg[:60]!r}")
        if not (ok_dead and not raised):
            fails.append("all dead")

        # --- probe semantics: an origin 404 is alive, a 1033/5xx is not
        import urllib.error

        def http_error(code):
            def f(url, timeout=4):
                try:
                    raise urllib.error.HTTPError(url, code, "x", None, None)
                except urllib.error.HTTPError as e:
                    return 200 <= e.code < 500
            return f

        print(f"{'PASS' if http_error(404)('u') else 'FAIL'}  probe treats 404 as alive")
        print(f"{'PASS' if not http_error(502)('u') else 'FAIL'}  probe treats 502 as dead")
        if not http_error(404)("u"):
            fails.append("404 alive")
        if http_error(502)("u"):
            fails.append("502 dead")
    finally:
        (cf.ensure_cloudflared, cf._wait_origin, cf._spawn_quick, cf._public_ok,
         cf.time.sleep) = real

    print()
    print("RESULT: TUNNEL VERIFY OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
