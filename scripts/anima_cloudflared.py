"""Expose the local Gradio server through a Cloudflare quick tunnel.

Replaces Gradio's `share=True` (which relies on Gradio's own frpc tunnel) with a
`cloudflared` quick tunnel, so the public URL is a trycloudflare.com hostname.

Only the stdlib is used, so this runs under any interpreter on the Colab host.
"""
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

BIN = "/usr/local/bin/cloudflared"
URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")
LOG = "/content/cloudflared.log"


def ensure_cloudflared(path=BIN):
    """Download the cloudflared binary once if it is not already present."""
    if os.path.exists(path) and os.access(path, os.X_OK):
        print(f"cloudflared already present: {path}", flush=True)
        return path
    if platform.system() != "Linux":
        raise SystemExit("this helper targets the Linux Colab runtime")
    arch = "amd64" if platform.machine() in ("x86_64", "amd64") else "arm64"
    url = ("https://github.com/cloudflare/cloudflared/releases/latest/download/"
           f"cloudflared-linux-{arch}")
    print(f"downloading cloudflared ({arch}) ...", flush=True)
    tmp = "/tmp/cloudflared.download"
    urllib.request.urlretrieve(url, tmp)
    os.chmod(tmp, 0o755)
    subprocess.run(["mv", tmp, path], check=True)
    print(f"installed {path}", flush=True)
    return path


def _public_ok(url, timeout=4):
    """True if the published URL actually serves something.

    A quick tunnel's hostname is printed by cloudflared before it is usable, and sometimes it
    never becomes usable at all -- the browser shows Cloudflare error 1033 and the obvious
    conclusion is "the tunnel expired, run it again". Probing turns that guess into a check.
    4xx counts as alive: the origin answering "not found" still means the edge is routing.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "anima-tunnel-probe"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 500
    except urllib.error.HTTPError as e:
        return 200 <= e.code < 500
    except Exception:
        return False


def _wait_origin(port, timeout):
    """Hold until the local Gradio server answers, so the tunnel is not proxying a void."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise SystemExit(f"origin http://127.0.0.1:{port}/ did not answer within {timeout}s")


def _spawn_quick(binp, port, log_path, timeout, append=False):
    """Start cloudflared for a quick tunnel and return (process, url) once a URL is logged."""
    log = open(log_path, "a" if append else "w")
    p = subprocess.Popen(
        [binp, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(1)
        txt = open(log_path, errors="replace").read()
        if p.poll() is not None:
            raise SystemExit("cloudflared exited rc=%s:\n%s"
                             % (p.returncode, txt[-2000:]))
        m = URL_RE.search(txt)
        if m:
            return p, m.group(0)
    raise SystemExit("no tunnel URL after %ss:\n%s"
                     % (timeout, open(log_path, errors="replace").read()[-2000:]))


def start(port=7860, log_path=LOG, timeout=90, verify_timeout=90, attempts=2):
    """Start a quick tunnel and return (process, url) for a URL that has been probed.

    Returning the first URL cloudflared logs is not the same as returning a working one. Each
    attempt is verified against the public edge; a hostname that never answers is torn down and
    replaced, which is what "run it twice" was doing by hand.
    """
    binp = ensure_cloudflared()
    _wait_origin(port, timeout)
    dead = []
    p = url = None
    for attempt in range(1, attempts + 1):
        # Truncate every attempt: appending made URL_RE match the PREVIOUS attempt's hostname, so
        # the retry probed a corpse and "restarted" nothing. A fresh file is the only way the
        # matcher can be sure it read this attempt's URL.
        p, url = _spawn_quick(binp, port, log_path, timeout, append=False)
        t0 = time.time()
        while time.time() - t0 < verify_timeout:
            if _public_ok(url):
                print(f"tunnel verified: {url} "
                      f"(attempt {attempt}, {time.time() - t0:.0f}s)", flush=True)
                return p, url
            time.sleep(2)
        dead.append(url)
        print(f"tunnel {url} never answered within {verify_timeout}s -- restarting it",
              flush=True)
        try:
            p.terminate()
            p.wait(timeout=10)
        except Exception:
            pass

    # Do not raise. A tunnel is an access path, not the runtime: raising here killed a boot that
    # had already reached SERVER READY, which took the server, the UI and the session record down
    # with it. Return the last hostname, unverified, and say so.
    print("WARNING: no quick tunnel answered within "
          f"{verify_timeout}s per attempt ({', '.join(dead)}); returning {url} UNVERIFIED. "
          "The local server is unaffected -- use the notebook/atp exec path, or retry the tunnel.",
          flush=True)
    return p, url


def bare_host(hostname):
    """The hostname alone, with any scheme or trailing slash removed.

    The URL is rebuilt as https://<host>, so a value pasted from the browser as
    "https://anima.example.com/" must not produce "https://https://anima.example.com/".
    Accepting both forms costs four lines and removes a whole class of silent breakage.
    """
    host = (hostname or "").strip().rstrip("/")
    for scheme in ("https://", "http://"):
        if host.lower().startswith(scheme):
            host = host[len(scheme):]
    return host.rstrip("/")


def named_argv(binp, token):
    """argv for a named tunnel.

    `--token` is a flag of the `run` SUBCOMMAND, not of `tunnel`: `cloudflared tunnel --token X`
    answers "Incorrect Usage. flag provided but not defined: -token" and exits, so the connector
    never starts. `--no-autoupdate` has to come before `run` as well -- after it, cloudflared
    says the flag is not defined.

    Verified against cloudflared 2026.10.2: this exact order parses, and a bogus token then
    fails at "Tunnel token is not valid", which is the parser accepting the flags.
    """
    return [binp, "tunnel", "--no-autoupdate", "run", "--token", token]


def start_named(token, hostname="", port=7860, log_path=LOG, timeout=120):
    """Run a NAMED tunnel from a connector token and return (process, url).

    A quick tunnel (`start`) gets a random *.trycloudflare.com name that changes on every
    restart, which is useless as a bookmark and means a new URL in every log line. A named
    tunnel is created once in the Cloudflare dashboard with a public hostname pointing at
    http://localhost:7860; its connector token is then reusable here, and the URL stays put.

    The token is what the dashboard shows under the tunnel's "Install connector" step. It is a
    credential: passed here it is never written to the log.
    """
    binp = ensure_cloudflared()
    log = open(log_path, "w")

    hostname = bare_host(hostname)

    _t0 = time.time()
    while time.time() - _t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise SystemExit(f"origin http://127.0.0.1:{port}/ did not answer within {timeout}s")

    p = subprocess.Popen(named_argv(binp, token),
                         stdout=log, stderr=subprocess.STDOUT)
    # With a token there is no URL to scrape -- the hostname lives in the dashboard. Wait for a
    # registered connection instead, which is the point where Cloudflare is actually routing.
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(1)
        txt = open(log_path, errors="replace").read()
        if p.poll() is not None:
            # Never echo the tail verbatim: it can contain the token.
            raise SystemExit(f"cloudflared exited rc={p.returncode} (see {log_path})")
        if "Registered tunnel connection" in txt or "Connection registered" in txt:
            url = f"https://{hostname}" if hostname else "(named tunnel: hostname is in the dashboard)"
            return p, url
    raise SystemExit(f"named tunnel did not register within {timeout}s (see {log_path})")


def stop(p):
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    _port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("ANIMA_UI_PORT", "7860"))
    _p, _url = start(_port)
    print("PUBLIC_URL:", _url, flush=True)
    print("(leave this cell running; the tunnel closes when the process is killed)", flush=True)
    try:
        _p.wait()
    except KeyboardInterrupt:
        stop(_p)
