"""Colab setup helper for the Anima-Aesthetic MaxDiffusion fork."""
import os
import subprocess
from pathlib import Path

REPO = Path(os.environ.get("ANIMA_REPO", "/content/maxdiffusion"))
VENV = REPO / ".venv"
PY = VENV / "bin/python"

env = {**os.environ, "PATH": f"{VENV}/bin:/usr/local/bin:" + os.environ["PATH"],
       "PYTHONPATH": str(REPO / "src"), "HF_HOME": "/content/hf_cache",
       "PYTHONUNBUFFERED": "1", "MPLBACKEND": "agg"}
env.pop("UV_SYSTEM_PYTHON", None)

if not PY.exists():
    subprocess.run(["uv", "venv", "--python", "3.12", str(VENV), "--seed"], check=True)
setup = (REPO / "setup.sh").read_text().replace("python3", str(PY))
Path("/content/anima_setup_v5e.sh").write_text(setup)
subprocess.run(["bash", "/content/anima_setup_v5e.sh", "MODE=stable", "DEVICE=tpu"], cwd=REPO, env=env, check=True)
subprocess.run(["uv", "pip", "install", "-q", "--python", str(PY),
                "transformers", "tokenizers", "accelerate", "safetensors",
                "huggingface_hub", "absl-py", "Pillow", "einops"],
               check=True, env=env)
print("ANIMA_SETUP_DONE")
