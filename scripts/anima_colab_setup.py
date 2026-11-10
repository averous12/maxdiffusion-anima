"""Colab setup helper for the Anima-Aesthetic MaxDiffusion fork."""
import os
import subprocess
from pathlib import Path

REPO = Path(os.environ.get("ANIMA_REPO", "/content/maxdiffusion"))
VENV = REPO / ".venv"
PY = VENV / "bin/python"
RUNTIME_LOCK = REPO / "scripts" / "runtime_tpu_requirements.txt"
SETUP_SCRIPT = Path("/content/anima_setup_v5e.sh")
HF_CACHE = Path("/content/hf_cache")  # Deliberately local; XLA is the only Drive cache.

HF_CACHE.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "PATH": f"{VENV}/bin:/usr/local/bin:" + os.environ["PATH"],
       "PYTHONPATH": str(REPO / "src"), "HF_HOME": str(HF_CACHE),
       "HF_HUB_CACHE": str(HF_CACHE / "hub"), "ANIMA_HF_CACHE": str(HF_CACHE),
       "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}
env.pop("UV_SYSTEM_PYTHON", None)
env.pop("ENABLE_PJRT_COMPATIBILITY", None)

if not PY.exists():
    subprocess.run(["uv", "venv", "--python", "3.12", str(VENV), "--seed"], check=True)

setup = (REPO / "setup.sh").read_text().replace("python3", str(PY))
SETUP_SCRIPT.write_text(setup)
subprocess.run(["bash", str(SETUP_SCRIPT), "MODE=stable", "DEVICE=tpu",
                "JAX_VERSION=0.7.2"], cwd=REPO, env=env, check=True)

# The setup script may have left a PJRT plugin from a different install path.
# Remove direct installs first; then let the supported TPU extra resolve the
# matching jaxlib/libtpu pair together.
def remove_stale_pjrt_packages():
    subprocess.run(["uv", "pip", "uninstall", "--python", str(PY),
                    "libtpu", "libtpu-nightly", "jaxlib", "jax"],
                   env=env, check=False)


def install_runtime_lock():
    remove_stale_pjrt_packages()
    subprocess.run(["uv", "pip", "install", "--python", str(PY),
                    "-r", str(RUNTIME_LOCK),
                    "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"],
                   cwd=REPO, env=env, check=True)


install_runtime_lock()
subprocess.run(["uv", "pip", "install", "-q", "--python", str(PY),
                "transformers", "tokenizers", "accelerate", "safetensors",
                "huggingface_hub", "absl-py", "Pillow", "einops", "gradio"],
               env=env, check=True)
install_runtime_lock()

# A passing import is not enough: the PJRT ABI error can surface on the first
# device execution. Compile and run a tiny operation before the server boots.
verify = r'''
import glob
import importlib.metadata as metadata
import jax
import jax.numpy as jnp
import jaxlib
import os
import site
import sys
import traceback

for name in ("jax", "jaxlib", "libtpu"):
    try:
        print(name, metadata.version(name), flush=True)
    except Exception as e:
        print(name, type(e).__name__, e, flush=True)

print("jax file:", jax.__file__, flush=True)
print("jaxlib file:", jaxlib.__file__, flush=True)
print("PJRT_CLIENT_CREATE_OPTIONS:", os.environ.get("PJRT_CLIENT_CREATE_OPTIONS"), flush=True)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"), flush=True)
print("sys.executable:", sys.executable, flush=True)
print("plugin paths:", flush=True)
for root in site.getsitepackages():
    for pattern in ("*pjrt*", "*libtpu*"):
        for path in glob.glob(os.path.join(root, "**", pattern), recursive=True):
            print(" ", path, flush=True)

try:
    jax.print_environment_info()
    devices = jax.devices()
    if not devices or any(device.platform != "tpu" for device in devices):
        raise RuntimeError(f"expected TPU devices, got {devices!r}")
    result = jax.jit(lambda x: x + 1)(jnp.ones((1,), dtype=jnp.float32))
    result.block_until_ready()
except Exception:
    print("PJRT probe: FAIL", flush=True)
    traceback.print_exc()
    raise
else:
    print("PJRT probe: PASS", flush=True)
    print("TPU devices:", devices, flush=True)
'''
subprocess.run([str(PY), "-u", "-c", verify], env=env, check=True)
print("HF cache:", HF_CACHE, "(local only)")
print("ANIMA_SETUP_DONE")
