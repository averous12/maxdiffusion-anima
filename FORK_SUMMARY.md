# Anima on MaxDiffusion — Fork Summary

This repository is a fork of [`google/maxdiffusion`](https://github.com/google/maxdiffusion) that adds a complete, TPU-serving pipeline for the Anima Aesthetic v1.1 diffusion model.

## Why this fork exists

`google/maxdiffusion` is a training/inference framework for Stable Diffusion-style models on JAX/Flax/TPU. This fork adapts it to:

- Run the **Anima Aesthetic v1.1** transformer + conditioner
- Use the **Anima-Base-v1.0-Diffusers** Qwen3 text encoder, tokenizers, and VAE
- Serve interactive generation through a **persistent file-based server + Gradio UI** on Colab TPU v5e

## New files

| Path | Purpose |
|---|---|
| `scripts/anima_aesthetic_server.py` | Persistent server: loads models, warms up executables, watches `anima_request.json`, writes progress and images |
| `scripts/anima_gradio_ui.py` | Gradio UI: writes request JSON, polls progress + previews, archives finished images |
| `scripts/anima_cloudflared.py` | Publishes the Gradio port via a Cloudflare quick tunnel |
| `scripts/anima_aesthetic_tpu.py` | One-shot end-to-end generation script (pre-server) |
| `scripts/anima_scan_e2e.py` | End-to-end scan/block runner and CPU parity helper |
| `src/maxdiffusion/models/anima_cosmos_flax.py` | Flax AnimaCosmos transformer, attention blocks, weight conversion |
| `src/maxdiffusion/models/anima_text_conditioner_flax.py` | Text conditioner + adapter conversion |
| `src/maxdiffusion/models/qwen3_flax.py` | Qwen3 text encoder in Flax |
| `src/maxdiffusion/models/qwen3_utils.py` | Qwen3 safetensors → Flax weight conversion |
| `src/maxdiffusion/models/qwen_image_vae_utils.py` | VAE conversion utilities |

## Major modifications to existing code

- Added `src/maxdiffusion/models/anima_cosmos_flax.py` with `_nearest_resize_mask()` to handle non-multiple image dimensions
- Added `scripts/anima_aesthetic_server.py` as a self-contained server launcher with embedded server code string
- Added `scripts/anima_gradio_ui.py` for browser UI and headless in-kernel generation
- Added `scripts/anima_cloudflared.py` for tunneling

## Key technical decisions

### Dtype policy

- `BF16_BF16_F32`: bf16 operands, fp32 accumulate, fp32 activations
- Keeps memory low without full fp32 cost

### Transformer refactor

- Replaced unrolled blocks with `flax.nnx.scan`
- Host RSS dropped from 46 GB to ~11 GB
- CPU parity: max abs 1.32e-05 / cosine 1.0000001 (14-layer fp32)

### Dimension rounding

- Requested H/W rounded to **multiple of 16** (VAE stride 8 × patchify stride 2)

### Serving fixes

- Padding mask nearest-resize for non-multiple dims
- Warmup dtype matches real path (bf16 context, fp32 latents)
- Timing excludes preview decode cost
- `gen_id` handshake between UI and server
- `demo.queue()` for live preview streaming
- Cloudflared waits for origin before tunneling
- Random seed by default with resolved seed reported and archived

## Measured performance (TPU v5e-1)

- Boot: ~4.3 min
- Warm generation: 30 steps at 1024² in **~21 s**
- HBM peak: ~14 GB / 16.91 GB
- Host RSS: ~13 GB

## Usage

See the companion project `anima_tpu_project` for the Colab notebook, validators, and run docs.

## Repository

Fork: `https://github.com/averous12/maxdiffusion-anima`
Upstream: `https://github.com/google/maxdiffusion`
