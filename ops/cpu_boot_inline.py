"""Call boot() in-process to distinguish an exception from an OOM kill.

A silent process death with no traceback looks like the OOM killer; a real bug would raise and
print. This runs the same boot path with the same env, so whichever happens is attributable.
"""
import os
import sys
import time
import traceback

CONTENT = "/content"
REPO = f"{CONTENT}/maxdiffusion"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def meminfo():
    try:
        d = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            d[k] = v.strip()
        for k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            log(f"{k}: {d.get(k)}")
    except Exception as e:
        log(f"meminfo failed: {e}")


def main():
    meminfo()
    os.environ.update({
        "PYTHONPATH": f"{REPO}/src",
        "ANIMA_WARMUP_PX": "128",
        "ANIMA_LLLITE_DIR": CONTENT,
        "ANIMA_SNAPREF_DIR": CONTENT,
    })
    sys.path.insert(0, f"{REPO}/src")

    # The kernel keeps imported modules for its whole life, so a re-run after a git pull would
    # silently execute the OLD bytecode (tracebacks even pointed at pre-pull line numbers).
    # Purge anything maxdiffusion before importing.
    import importlib
    for name in [n for n in list(sys.modules) if n.startswith("maxdiffusion")]:
        del sys.modules[name]
    importlib.invalidate_caches()

    import maxdiffusion.runtime.pipeline as p
    log("module imported")
    try:
        p.boot()
        log("BOOT RETURNED OK")
        meminfo()
        return 0
    except MemoryError as e:
        log(f"MEMORY ERROR: {e}")
        return 2
    except Exception as e:
        log(f"EXCEPTION {type(e).__name__}: {e}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
