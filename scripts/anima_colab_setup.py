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

if not PY.exists():
    subprocess.run(["uv", "venv", "--python", "3.12", str(VENV), "--seed"], check=True)

setup = (REPO / "setup.sh").read_text().replace("python3", str(PY))
SETUP_SCRIPT.write_text(setup)
subprocess.run(["bash", str(SETUP_SCRIPT), "MODE=stable", "DEVICE=tpu",
                "JAX_VERSION=0.7.2"], cwd=REPO, env=env, check=True)

# MaxDiffusion's generated requirements can drift; apply the locked TPU stack
# after setup and again after installing UI dependencies to prevent upgrades.
def install_runtime_lock():
    subprocess.run(["uv", "pip", "install", "--python", str(PY),
                    "-r", str(RUNTIME_LOCK)], cwd=REPO, env=env, check=True)

install_runtime_lock()
subprocess.run(["uv", "pip", "install", "-q", "--python", str(PY),
                "transformers", "tokenizers", "accelerate", "safetensors",
                "huggingface_hub", "absl-py", "Pillow", "einops", "gradio"],
               env=env, check=True)
install_runtime_lock()

verify = r'''
from importlib.metadata import version
import jax
expected = {"jax": "0.7.2", "jaxlib": "0.7.2", "flax": "0.11.2", "libtpu": "0.0.34"}
actual = {name: version(name) for name in expected}
for name, wanted in expected.items():
    if actual[name] != wanted:
        raise SystemExit(f"runtime pin mismatch: {name}={actual[name]}, expected {wanted}")
print("TPU runtime versions:", actual)
print("TPU backend:", jax.default_backend(), "devices:", jax.devices())
'''
subprocess.run([str(PY), "-c", verify], env=env, check=True)
print("HF cache:", HF_CACHE, "(local only)")
print("ANIMA_SETUP_DONE")
