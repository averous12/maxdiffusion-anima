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
| **WebUI** | — | 3 tests construct `build()`: DECODE_UI 7/7, REGIONAL_END_UI 4/4, HELP_CONTENT 18/19 (1 pre-existing doc gap) | never launched against a live server | **structurally checked only** |

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
| 11 | WebUI end-to-end | submit from the browser, get an image back, progress + preview behave | never launched against a live server |

Items 1–2 are the regression gate and should run before anything else. Items 3–4 answer the
LLLite question that is actually open. Items 5–8 are first-ever runs for features that carry a
lot of runtime code and no tests.

## 5. Known defects that will bite during this

| Defect | Effect | Where |
|---|---|---|
| `adapter_block_map` assumes an iterable | `TypeError: 'int' object is not iterable` for a scalar layer count | `anima_snapref_flax.py:65` |
| LoRA has no test | regressions are invisible until a run looks wrong | `ops/`, `tests/` (empty) |
| `lllite + snapref` untested | the one combination most likely to be silently broken | — |
