"""Leaving the new params empty must behave exactly like before, not half-enable something.

Two "empty" cases matter:
  * CLOUDFLARE_TUNNEL_TOKEN empty  -> quick tunnel, random URL (the old behaviour)
  * XLA_HF_REPO empty              -> local-only cache, no hub traffic at all

Rather than reading the notebook and hoping, this lifts the real start_tunnel() and
push_xla_cache() out of the launch cell and calls them with stubs, so the branch that runs is
the branch the notebook would run.

Run from the workspace root.
"""
import ast
import json
import sys

sys.path.insert(0, "maxdiffusion/scripts")   # anima_cloudflared for the hostname helper

NB = "anima_tpu_project/anima_aesthetic_colab.ipynb"


def cell_with(nb, needle):
    for c in nb["cells"]:
        if c["cell_type"] == "code" and needle in "".join(c["source"]):
            return "".join(c["source"])
    raise SystemExit(f"no code cell containing {needle!r}")


def extract(src, name):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise SystemExit(f"function {name} not in the cell")


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return ("proc", "https://stub")


def build_ns(func_src, token, repo, call_log):
    """A namespace with just enough stubs to run the extracted function."""
    class CF:
        start = staticmethod(Recorder())
        start_named = staticmethod(Recorder())
        stop = staticmethod(lambda p: None)

    ns = {
        "cf": CF, "UI_PORT": 7860, "CF_LOG": "/tmp/cf.log",
        "CLOUDFLARE_TUNNEL_TOKEN": token, "CLOUDFLARE_HOSTNAME": "anima.example.com",
        "XLA_HF_REPO": repo, "XLA_HF_PUSH_S": 600, "XLA_CACHE_DIR": "/tmp/xla",
        "HF_TOKEN": "", "py": "python", "UI_LOG": "/tmp/ui.log",
        "ui_alive": lambda: True, "ui_answer": lambda: True,
        "ui_proc": object(), "cf_proc": None, "public_url": None,
        "LOG": lambda m: call_log.append(m),
        "dump_tail": lambda *a, **k: None,
        "subprocess": type("S", (), {"run": staticmethod(
            lambda *a, **k: type("R", (), {"stdout": "", "stderr": ""})())})(),
        "os": __import__("os"), "time": __import__("time"), "json": json,
        "_t_last_push": [0.0],
    }
    exec(compile(func_src, "<nb>", "exec"), ns)
    return ns


def main():
    nb = json.load(open(NB, encoding="utf-8"))
    launch = cell_with(nb, "def start_tunnel")
    start_src = extract(launch, "start_tunnel")
    push_src = extract(launch, "push_xla_cache")
    fails = []

    # --- empty token -> quick tunnel
    log = []
    ns = build_ns(start_src + "\n" + push_src, token="", repo="", call_log=log)
    ns["start_tunnel"]()
    quick = len(ns["cf"].start.calls) == 1 and not ns["cf"].start_named.calls
    print(f"{'PASS' if quick else 'FAIL'}  empty token -> cf.start (quick tunnel) called once")
    if not quick:
        fails.append("empty token fallback")

    # --- token set -> named tunnel, and NOT the quick one
    ns2 = build_ns(start_src + "\n" + push_src, token="tok123", repo="me/cache", call_log=[])
    ns2["start_tunnel"]()
    named = bool(ns2["cf"].start_named.calls) and not ns2["cf"].start.calls
    host_ok = bool(ns2["cf"].start_named.calls) and \
        ns2["cf"].start_named.calls[0][1].get("hostname") == "anima.example.com"
    print(f"{'PASS' if named else 'FAIL'}  token set -> cf.start_named, quick tunnel untouched")
    print(f"{'PASS' if host_ok else 'FAIL'}  hostname passed through to the named tunnel")
    if not named:
        fails.append("named tunnel")
    if not host_ok:
        fails.append("hostname")

    # --- empty repo -> push is a no-op (no subprocess, no hub call)
    ran = {"count": 0}
    ns3 = build_ns(start_src + "\n" + push_src, token="", repo="", call_log=[])
    ns3["subprocess"].run = staticmethod(lambda *a, **k: ran.__setitem__("count", ran["count"] + 1))
    ns3["push_xla_cache"](force=True)
    noop = ran["count"] == 0
    print(f"{'PASS' if noop else 'FAIL'}  empty XLA_HF_REPO -> push runs nothing")
    if not noop:
        fails.append("empty repo no-op")

    # --- repo set -> push runs the script once
    ns4 = build_ns(start_src + "\n" + push_src, token="", repo="me/cache", call_log=[])
    ran2 = {"count": 0}
    ns4["subprocess"].run = staticmethod(lambda *a, **k: ran2.__setitem__("count", ran2["count"] + 1))
    ns4["push_xla_cache"](force=True)
    pushes = ran2["count"] == 1
    print(f"{'PASS' if pushes else 'FAIL'}  repo set -> push runs the sync script once")
    if not pushes:
        fails.append("push with repo")

    # --- hostname accepted bare or pasted as a URL (the URL is rebuilt as https://<host>)
    import anima_cloudflared as cfu
    forms = ["anima.example.com", "https://anima.example.com", "http://anima.example.com/",
             "  anima.example.com/  ", "HTTPS://anima.example.com"]
    bad = [f for f in forms if cfu.bare_host(f) != "anima.example.com"]
    print(f"{'PASS' if not bad else 'FAIL'}  hostname accepted bare or as a URL "
          f"{'(all 5 forms)' if not bad else bad}")
    if bad:
        fails.append("hostname forms")

    # --- the boot cell must skip the pull entirely when the repo is empty
    boot = cell_with(nb, "xla cache: restoring")
    elifs = [n for n in ast.walk(ast.parse(boot)) if isinstance(n, ast.If)]
    has_else = any("XLA_HF_REPO" in ast.get_source_segment(boot, n.test) and n.orelse
                   for n in elifs)
    print(f"{'PASS' if has_else else 'FAIL'}  boot cell has an explicit else for an empty repo")
    if not has_else:
        fails.append("boot else")

    print()
    print("RESULT: FALLBACKS OK" if not fails else f"RESULT: FAILED {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
