# Refactor plan — making this maintainable

Written after the LLLite work exposed how hard the current shape is to change safely.
Everything below is measured, not felt; the command that produced each number is named.

The goal is **not** a rewrite. The generation behaviour is validated and must not move.
The goal is that a change costs one edit in one place, and that a test can prove it.

---

## 1. What is actually wrong

### P1 — The runtime is a string (the root cause)

| File | Total lines | Inside a `code = r'''…'''` string |
|---|---|---|
| `scripts/anima_aesthetic_server.py` | 1996 | **1964 (98%)** |
| `scripts/anima_scan_e2e.py` | 257 | 223 (87%) |
| `scripts/anima_aesthetic_tpu.py` (deleted) | 312 | 312 (100%) |

The server is a thin launcher that ships its real body to `subprocess.run([py, "-c", code])`.

Consequences, all of which we hit during the LLLite work:

* **Not importable** — so the mode rules had to be pulled out into a real module
  (`anima_control_modes.py`) before they could be tested at all.
* **No static analysis** — linters, type checkers and `py_compile` see a string.
* **Tracebacks say `<string>`** — line numbers in a stack trace are meaningless.
* **Edits are string surgery** — a `sed` targeting line 95 of the wrapper silently matched
  nothing, which is how a strength sweep came to overwrite the same output file twice and
  produce a false "control is ignored" conclusion.
* **Duplication was structural** — the TPU driver used the same pattern, so sharing code
  between the two was impossible by construction.

### P2 — Seven jits for what is one forward pass

```
cond_forward  transformer_forward  snapref_target_forward  snapref_reference_forward
lllite_forward  lllite_snapref_target_forward  pixel_tap_forward
```
plus three predictors (`cfg_prediction`, `snapref_cfg_prediction`, `lllite_cfg_prediction`).

Each mode got its own copy of the same body. `lllite_forward` and `snapref_target_forward`
differ from `lllite_snapref_target_forward` only in which kwargs are passed — that is four
bodies where one parameterised call would do. Every new combination now costs another jit.

### P3 — Mutable module-level state shared across requests

`_SNAPREF_STATE` (9 refs) and `_LLLITE_CACHE` (4 refs) are globals mutated during a
generation. Two concurrent or interleaved requests would corrupt each other, and no test can
reset them cleanly.

### P4 — Nesting depth 16

Maximum indentation inside the served code is **16 levels**. The request path nests
`while True → try → if → for → if → …`. This is where the inline `if regional and lllite:
raise` lived, four levels down, unreachable by any test.

### P5 — One 356-line function

`anima_gradio_ui.py` is a normal module (good), but `build()` is **356 lines** — layout,
widget definitions, event wiring and defaults in one scope.

### P6 — `ops/` has no ownership

34 scripts: 5 real tests, 20 SnapRef probes (including four superseded
`snapref_scan_probe{,2,3,4}.py`), 6 UI/e2e scripts, 1 bootstrap. No marker of which are
load-bearing, so a reader cannot tell a check from a scratchpad.

### P7 — No test covers request → image

There is no test that exercises the server's generation path end to end. Everything proven
about LLLite was proven against modules and against the (now deleted) driver. The server's
own path is untested.

---

## 2. Target shape

```
src/maxdiffusion/runtime/          ← importable, testable, no subprocess
    request.py        parse + validate a request dict        (pure)
    modes.py          control-mode rules                     (already exists, pure)
    prepare.py        text/ref/control encoding -> context
    denoise.py        the loop + ONE parameterised forward
    decode.py         VAE / pixel head
    server.py         the file-IPC loop, ~200 lines, thin
scripts/anima_aesthetic_server.py   ← launcher only: env, subprocess or direct call
```

The rule that keeps it honest: **if it can be imported, it can be tested.** Anything that
must stay in a subprocess stays tiny.

---

## 3. Phased steps

Each phase is independently shippable and independently verifiable. Do them in order; do
not start the next until the acceptance check passes.

### Phase 0 — Lift the string, change nothing

Move the body of `code = r'''…'''` into `src/maxdiffusion/runtime/` as a real module tree,
keeping the logic **byte-for-byte equivalent**. The launcher imports it.

*Acceptance:* the module imports cleanly; every top-level name the launcher used still
resolves; a one-request smoke run produces a byte-identical image (compare sha256) to the
pre-refactor server.

*Risk:* low. This is a move, not a rewrite, and the hash check catches any slip.

### Phase 1 — Extract the request path into functions

Split the `while True` body into `parse → validate → prepare → denoise → decode → write`.
Kill the depth-16 nesting: target max depth 4.

*Acceptance:* max indentation ≤ 4 in the served code (measure with the AST walker already
used above); same sha256 as Phase 0.

### Phase 2 — One forward, not seven

Replace the seven jits with a single `forward(params, latents, t, ctx, control)` where
`control` is a pytree carrying `lora`, `ref_kv`, `ref_bias`, `lllite_w`, `cond_emb`,
`strength`. Absent features are `None`.

*Acceptance:* same sha256 per mode; and the count of compiled executables does not increase
(JAX caches per pytree structure, so each mode still gets its own trace — that is fine and
expected, the win is one body to maintain).

*Note:* verify with a mode matrix run, one image per mode, all six hashes recorded.

### Phase 3 — Kill the globals

Replace `_SNAPREF_STATE` / `_LLLITE_CACHE` with a per-request context object plus an
explicitly-scoped cache passed in.

*Acceptance:* zero module-level mutable dicts in the runtime (AST check); same sha256.

### Phase 4 — Break up `build()`

Split the 356-line `build()` into `_layout()`, `_controls()`, `_events()`, `_defaults()`.

*Acceptance:* no function over 120 lines in `anima_gradio_ui.py`; UI still starts and a
request round-trips (screenshot + vision check, per this project's standard).

### Phase 5 — Tame `ops/`

Move probes to `ops/probes/`, keep tests at `ops/`, add a one-table index at the top of
`ops/README.md` saying what is load-bearing. Delete the four superseded
`snapref_scan_probe{2,3,4}.py`.

*Acceptance:* every file in `ops/` appears in the index; nothing referenced by docs or CI is
deleted.

### Phase 6 — First end-to-end test

A test that drives the runtime with a stubbed transformer (tiny dims) through
request → latents, asserting shapes and that strength 0 equals no control. No TPU needed.

*Acceptance:* the test fails if `lllite_strength` is wired to zero, proving it has teeth.

---

## 4. What NOT to do

* **No big-bang rewrite.** The generation path is validated; a rewrite would invalidate
  every number in `MODEL_ARCHITECTURE.md` and we would not know which change broke what.
* **No re-forking.** The driver already taught this lesson: a fork of the runtime goes
  stale silently and duplicates the logic it was meant to reuse.
* **No phase without a hash check.** "It looks right" is what produced a wrong conclusion
  about the control already.

## 5. Why this order

Phase 0 unlocks everything else: until the code is importable, none of the later phases can
be tested, only eyeballed. Phase 1 makes the code navigable. Phase 2 removes the growth
path that makes every new feature another jit. Phases 3–4 are hygiene that stops silent
cross-request bugs. Phases 5–6 are what keep it from rotting again.

The cheapest phase with the largest payoff is **Phase 0**, and it changes no behaviour.
Start there.
