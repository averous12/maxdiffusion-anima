# Validation matrix — what is proven, and what still needs a TPU run

The refactor changed the **shared** inference path: one `_transformer_forward` now serves every
mode, and the runtime was reorganised into `boot()` / `serve()` / `handle_request()`. That means
everything downstream of the transformer needs re-validation — **not just LLLite**. LoRA,
SnapRef, regional and the WebUI all ride the same code that moved.

This document separates three very different states, because conflating them is how a
regression ships:

* **CPU-proven** — deterministic, needs no TPU, re-runnable here.
* **TPU-measured** — a number was actually recorded from a TPU run.
* **Unverified** — never run, or run only before the refactor.

Counts below are live references in `src/maxdiffusion/runtime/pipeline.py` and test files that
exist today.

---

## 1. Per-feature state

| Feature | Runtime refs | CPU proof | TPU proof | Verdict |
|---|---|---|---|---|
| **LLLite** | 68 | 4 real checks: parity vs ComfyUI **2.742e-06** (828 tensors, 0 missing), scan≡loop **1.144e-05**, strength-0 bit-exact, adapter≡direct-apply bitwise | control changes the image (**643,837 px**, mean 39.7); step cost **2303–2341 ms** vs 1896 baseline | **best-covered, still incomplete** |
| **SnapRef** | 145 | none — 20 `snapref_*` files in `ops/` are probes, not assertions | none | **unverified** |
| **LoRA** | 112 | **none — zero tests anywhere** | none | **unverified, untested** |
| **regional** | 49 | 2 UI-contract tests (mask editor → request JSON) | none | **unverified** |
| **WebUI** | — | `build()` → 203 components, 23 wirings, 0.6 s; 3 contract tests (7/7, 4/4, 18/19) | **Colab CPU: HTTP 200, renders 6 tabs in Chromium, Generate writes a valid request** | **verified — UI layer only, no generation** |

Two facts worth stating plainly:

* **LoRA has no test at all** — not a probe, not an assertion, nothing. It carries 112
  references in the runtime including the hot-swap path in `handle_request`. It is the
  largest untested surface in the project.
* **SnapRef's 20 `ops/` files are investigative probes**, not checks. They printed
  diagnostics during development; they assert nothing. Counting them as coverage would be
  wrong.

## 2. Combinations

Six modes are supported, one is refused by design.

| Mode | Ever run on TPU? | Notes |
|---|---|---|
| plain | **yes** | 30 steps @ 1896 ms/step, 1024²; image verified by eye |
| lllite | **yes** — but through the deleted one-shot driver, never the server | pose adherence still unproven |
| regional | no | |
| snapref | no | |
| regional + snapref | no | |
| lllite + snapref | **no** | the combined jit exists and its kwargs are unit-tested; it has never executed |
| regional + lllite | n/a — **refused** | `resolve_control_modes` raises: regional replaces the base prediction per region, so a whole-image control would be dropped silently |

The combination matrix is exactly where "it compiles" is most likely to be mistaken for "it
works": `lllite + snapref` passes a kwarg-mapping test and has never produced a pixel.

## 3. What CPU genuinely proves — and what it cannot

CPU gives certainty about arithmetic and wiring:

* the four jitted bodies are one body: adapter output is **bitwise identical** to a direct
  `apply()` with the same kwargs, using the real `anima_3dpose_6200` weights
* strength 0 is inert (`0.000e+00`), so enabling a control cannot perturb a plain run
* mode resolution: 15/15 cases, all six modes plus both refusals
* import is side-effect-free and the runtime surface is intact

CPU cannot prove: that a TPU kernel produces the same image, that step cost did not regress,
that memory fits, or that a control actually *steers* rather than merely perturbing.

**A caution learned the hard way this session:** all four `@jax.jit` decorators were silently
dropped by a refactor step and **every numeric test still passed** — JAX fell back to eager
execution, which is correct but dramatically slower. Only a decorator census caught it. Timing
and structure need their own checks; numerical equality is not enough.

## 4. TPU checklist

Ordered so a failure early invalidates less work. Each item has an acceptance criterion, not
just an action.

| # | Check | Acceptance | Why |
|---|---|---|---|
| 1 | `plain` at fixed seed | **sha256 matches** the pre-refactor `anima_perstep.png` | proves the refactor changed no arithmetic on the real device |
| 2 | step cost, `plain` | **~1896 ms/step** ± a few % | catches the dropped-jit class of regression, which numeric checks cannot |
| 3 | `lllite` through the **server** (not a driver) | image differs from plain at strength 1.0; **bit-identical** at strength 0 | the server path has never executed |
| 4 | `lllite` pose adherence | generated pose matches the control map — judged visually against the repo's own example sheet | the actual open question; effect ≠ correctness |
| 5 | `snapref` | reference character's features appear; no crash on encode/first-step | 145 runtime refs, zero runs |
| 6 | `lora` | LoRA visibly changes output at scale 1.0, and `lora_swap` reuses the executable (no recompile) | zero tests exist; largest blind spot |
| 7 | `regional` | region prompts affect only their masks; `regional_end` window respected | 2 UI tests, no inference run |
| 8 | `regional + snapref` | both effects visible, no crash | never run |
| 9 | `lllite + snapref` | both effects visible; no crash from the combined forward | jit exists, unit-tested, never executed |
| 10 | `regional + lllite` | **rejected** with the supported-combinations message | proves the guard runs in the real request path, not just in a unit test |
| 11 | WebUI **with the server** | submit from the browser, get an image back, progress + preview advance | the UI alone is verified on CPU (§6); the server half is not |

Items 1–2 are the regression gate and should run before anything else. Items 3–4 answer the
LLLite question that is actually open. Items 5–8 are first-ever runs for features that carry a
lot of runtime code and no tests.

## 5. Known defects that will bite during this

| Defect | Effect | Where |
|---|---|---|
| `adapter_block_map` assumes an iterable | `TypeError: 'int' object is not iterable` for a scalar layer count | `anima_snapref_flax.py:65` |
| LoRA has no test | regressions are invisible until a run looks wrong | `ops/`, `tests/` (empty) |
| `lllite + snapref` untested | the one combination most likely to be silently broken | — |

---

## 6. WebUI verified on a Colab CPU instance

The UI is deliberately independent of the model — it writes `/content/anima_request.json` and
polls `/content/anima_progress.json` — so it can be verified without booting 6.6 GB of weights.
Done on a plain CPU Colab VM with `ops/webui_cpu_*.py`:

| Step | Result |
|---|---|
| `gradio` installed | 6.31.0 |
| `build()` | **203 components, 23 event wirings, 0.6 s** |
| serve on `:7860` | **HTTP 200** |
| render in headless Chromium | title `Anima Aesthetic TPU`; **4 textareas, 21 inputs, 22 buttons, 6 tabs**; no page errors; screenshot 120,319 B |
| click Generate | `/content/anima_request.json` written |
| request contract | `prompt` carried verbatim; `negative_prompt` real; `height/width` 1024; `steps` 30; `guidance` 4.0; `seed` -1; `prompt_style` comfy; `preview_every` 5; one default regional box |

Tabs present: Settings, LoRAs, Decode, SnapRef, Regional, Help.

Not covered by this: an actual generation, since that needs the server and the weights. Also
blocked: the `cloudflared` quick tunnel returns **HTTP 429** from that host, so the screenshot
was taken inside the VM against `127.0.0.1:7860` rather than through a public URL. That is a
Cloudflare rate limit, not a defect in this code.

Two mistakes worth keeping: `networkidle` never fires because Gradio holds a websocket open
(wait for `gradio-app` instead), and Playwright's sync API refuses to start inside the Jupyter
asyncio loop that `colab exec` uses (run it in a child process).

---

## 7. Measured: the six-mode matrix on TPU v5e1

First run of every mode through the real server. 256x256, 1 step, seed 1234, identical prompt;
`sha256` of each output, first 16 hex characters. Timing is dominated by first-shape compile,
so only the hashes carry meaning.

| Mode | Result | sha256[:16] | Notes |
|---|---|---|---|
| plain | OK | `c187baa067855d48` | pays the first decode compile (10.8 s) |
| regional | OK | `1ff8ec05a9d617de` | |
| snapref | OK | `0eafb4feb24d97f5` | |
| lllite | OK | `0fb3cad2299205fc` | |
| regional + snapref | OK | `1ff8ec05a9d617de` | **identical to regional alone — SnapRef contributed nothing** |
| lllite + snapref | OK | `55c2888d904822d1` | distinct from both parents: composition works |
| regional + lllite | refused | — | `ControlModeError`, as designed |

6/7 produced an image, 5 distinct images. Boot 45 s warm / 195 s cold; the whole matrix 155 s.

### Two bugs this run found, both now fixed

1. **LLLite never worked through a request.** `lllite.image` arrives as a path, and
   `preprocess_control` did `Image.fromarray(np.asarray(image))` — a path becomes a `(1,1)` string
   array, so PIL raised `TypeError: Cannot handle this data type: (1, 1), <U21`. The width in
   that message (21) was literally the length of `/content/ctl_pose.png`. Every CPU probe passed
   decoded arrays, which is why parity and scan passed while the server path never once ran.
2. **The encoder needed the object, not the cache dict.** Once (1) cleared, the next error was
   `AttributeError: 'dict' object has no attribute 'config'`: `encode_conditioning` reads
   `.config`/`.cond`, but `get_lllite()` cached only derived fields.

### Outstanding

- **`regional + snapref` silently ignores the reference.** The image is bit-identical to
  `regional` alone, so SnapRef is either not applied or applied with zero effect on that path.
  Not root-caused. Every other combination behaves.
- Suite regressions found on the way: `serve()` crashed on the first request
  (`last_gen_id` never declared global — see `test_global_completeness.py`).
- **LoRA is still untested** — it needs `ANIMA_LORAS` and a LoRA file; neither was available.
- These are 256x256 one-step images: they prove routing and effect, not quality. Pose
  adherence for LLLite remains unjudged.
