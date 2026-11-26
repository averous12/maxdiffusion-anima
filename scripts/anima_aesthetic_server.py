import os
import subprocess
import sys

repo = os.environ.get("ANIMA_REPO", "/content/maxdiffusion")
venv = repo + "/.venv"
# Prefer the repo venv when it exists; otherwise fall back to the running
# interpreter so the server also works in sessions that pip-installed into the
# system Python (the E2E runner path).
if os.path.exists(venv + "/bin/python"):
    py = venv + "/bin/python"
    _path_prefix = venv + "/bin:"
else:
    py = sys.executable
    _path_prefix = ""
env = {**os.environ, "PATH": _path_prefix + "/usr/local/bin:" + os.environ["PATH"],
       "PYTHONPATH": repo + "/src", "HF_HOME": os.environ.get("HF_HOME", "/content/hf_cache"),
       "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}
env.pop("UV_SYSTEM_PYTHON", None)
print("server interpreter:", py, flush=True)

# The runtime used to be a code = r'''...''' string shipped to python -c. On Windows that
# overflowed the 32767-character command line and needed a write-to-file workaround. It is a
# real module now (maxdiffusion.runtime.pipeline): the same program, with tracebacks that
# carry file:line, linters that can read it, and no command-line length limit.
r = subprocess.Popen([py, "-u", "-m", "maxdiffusion.runtime.pipeline"], cwd=repo, env=env)
print("server pid", r.pid, flush=True)
