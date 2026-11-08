"""Fetch the weights a local (CPU) Anima run needs — the same repos/paths the server uses.

Skips the 3.91 GB diffusers-format transformer: the server takes its transformer from the
single-file ComfyUI checkpoint (ANIMA_AESTHETIC_FILE), which is also how SnapRef v1a was
trained (base = anima-base-v1.0).
"""
import os
import sys
import time

from huggingface_hub import snapshot_download

OUT = os.environ.get("ANIMA_LOCAL_HF", "E:/Workspace/colab/hf_cache")
BASE = "circlestone-labs/Anima-Base-v1.0-Diffusers"
COMFY = "circlestone-labs/Anima"

os.makedirs(OUT, exist_ok=True)
t0 = time.time()
p = snapshot_download(
    BASE, cache_dir=OUT,
    allow_patterns=["text_encoder/*", "tokenizer/*", "tokenizer_2/*", "t5_tokenizer/*",
                    "vae/*", "text_conditioner/*", "scheduler/*", "*.json"],
)
print("base snapshot:", p, f"{time.time()-t0:.0f}s", flush=True)

t0 = time.time()
q = snapshot_download(COMFY, cache_dir=OUT,
                      allow_patterns=["split_files/diffusion_models/anima-base-v1.0.safetensors"])
print("comfy snapshot:", q, f"{time.time()-t0:.0f}s", flush=True)

for root, _dirs, files in os.walk(OUT):
    for f in files:
        fp = os.path.join(root, f)
        if os.path.getsize(fp) > 50_000_000:
            print(f"{os.path.getsize(fp)/1e9:.2f}G  {fp}", flush=True)
print("LOCAL_WEIGHTS_DONE", flush=True)
sys.exit(0)
