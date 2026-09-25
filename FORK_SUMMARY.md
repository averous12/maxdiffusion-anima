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

### Prompt weighting

- `(tag:1.3)` syntax scales the attention of a prompt span
- Groups nest and weights multiply; weights clamped to **0..3**
- Plain parentheses (e.g. `fern (sousou no frieren)`) stay literal
- Backslash-escaped parentheses (`\(`, `\)`) produce literal parens inside a group
- Parsed and applied server-side where the tokenizer lives; both positive and negative prompts are processed

### Preview observability

- A failed live preview used to be invisible: the server caught the exception, logged one
  line, and the UI showed an empty box behind a healthy status line and a normal `done`.
- The server now publishes preview state on **every** progress record:
  `preview_ok`, `preview_fails` (consecutive), `preview_fails_total` (cumulative),
  `preview_total` (expected), `preview_error`, `preview_gaveup` — so the client can tell
  "no preview yet" from "previews keep failing".
- After **2 consecutive failures** the server stops attempting previews for that generation
  instead of paying a ~50 s XLA compile stall every N steps; the denoise loop still runs to
  completion. A failed VAE shape is remembered for the request in `_vae_failed` and not
  re-advertised as "compiling".
- The UI status line renders it: `previews 6/6`, `previews off`,
  `previews failing: <ErrorType>: <msg> (see /content/anima_server.log)`,
  `previews disabled after failures: ...`. The timeout branch now renders through the same
  status line instead of dumping the raw progress dict.
- Previews arrive **every `preview_every` steps**, not continuously: 30 steps at 5 = 6 decodes,
  and the final step always decodes. `preview_every=1` is available but costs a decode per step
  (a warm decode is ~0.5 s, so ~15 s on top of a ~17 s 30-step denoise); the default of 5 is 3
  decodes' worth of overhead.
- The preview handler logs a full traceback, and the line names the phase that broke —
  `preview failed at step N during decode|preview write` — so a full disk or a PIL/grid
  error is no longer mislabelled as a VAE decode failure.
- Both headless drivers (`anima_aesthetic_colab.ipynb` cell 6 and
  `ops/tpu_gradio_e2e.py`) were calling an obsolete 8-argument `start_gen` and unpacking 3
  values, so they raised `TypeError` before sending a request. They now match the current
  12-argument signature, consume the `(grid, gallery, snap, status)` 4-tuple, save PIL
  payloads, assert the arity, and print a `PREVIEW_VERDICT` line.

### True batching

- Request `batch` > 1 shares every denoise step across images on the TPU
- Each image uses `seed + k` so runs are still reproducible
- Server emits a preview grid during denoising and a final grid alongside individual PNGs
- Grid helpers adapt to non-square batches and image sizes

### Aspect-ratio sizing

- UI exposes 0.5 / 1.0 / 1.5 MP targets, locked to a user-supplied W:H aspect ratio
- Live shape preview before generation
- "Custom size" mode lets the user type exact pixel dimensions
- Server rounds requested H/W to multiples of 16 (VAE stride 8 × patchify stride 2)

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
