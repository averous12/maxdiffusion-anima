# ops/ — checks, probes and one-off tooling

Nothing here is library code. This index exists so a reader can tell a **check** (wired to
the refactor plan, expected to pass) from a **probe** (a past diagnostic kept for
reproducibility) from **scratch** (superseded, safe to delete).

## Checks — expect these to pass

| File | What it proves |
|---|---|
| `test_control_modes.py` | the six legal control combinations resolve; `regional+lllite` and 3-way refused. Pure CPU, no model |
| `test_forward_adapters.py` | the one forward forwards the right kwargs per mode, and no `None` leaks into the trace. Pure CPU |
| `test_forward_equivalence.py` | **inference is unchanged**: adapters are bitwise identical to a direct `apply`, with the real LLLite checkpoint. Pure CPU |
| `lllite_parity.py` | LLLite matches the ComfyUI reference (conditioning ~2.7e-06, 828 tensors, 0 missing) |
| `lllite_scan_test.py` | LLLite rides `nn.scan`; scan ≡ loop (1.1e-05); strength 0 is bit-exact |
| `lllite_full_forward.py` | the real transformer forward with LLLite; effect linear in strength (set `LL_LAYERS=28` for production depth) |
| `lllite_block_hook.py` | block-level hook wiring |
| `test_server_json_contract.py`, `test_decode_ui_contract.py`, `test_help_content.py`, `test_regional_editor.py`, `test_regional_end_ui.py`, `test_stall_watchdog.py` | server/UI contracts and the watchdog |

## Tooling — used, load-bearing

| File | Purpose |
|---|---|
| `anima_tpu_bootstrap.py` | Colab VM bootstrap: clone, overlay model files, setup, import check. **Setup only** — generation is the server's job |

## Probes — past diagnostics, kept for reproducibility

`snapref_*` (20 files). The `*.js` ones drive a browser over CDP for UI captures; the
`.py` ones inspect or exercise SnapRef internals. Superceded and safe to delete:

- `snapref_scan_probe{2,3,4}.py` — iterations of `snapref_scan_probe.py`
- `snapref_dd_{find,inspect,pick}.js` — three steps of one investigation
- `snapref_ui_{drive,fill,final,full,inspect,proof,shot}.js` — successive UI-capture scripts now that `test_*_ui_contract.py` covers the contracts

`browser_test.py` — standalone browser smoke test.

## Running them

The `lllite_*` checks resolve the HF cache relative to the current directory, so run them
from the workspace root (one level above this repo):

```bash
PYTHONPATH=maxdiffusion/src python maxdiffusion/ops/lllite_scan_test.py
```

`test_control_modes.py`, `test_forward_adapters.py` and `test_forward_equivalence.py` resolve
paths from `__file__` and run from any directory. The `lllite_*` checks still expect the
workspace root; making their checkpoint paths absolute is tracked in `docs/REFACTOR_PLAN.md`.

## Deleting

Pruning the superseded files is a judgement call, not a mechanical one: they are not
referenced by code, CI or the docs, but they are also the only remaining record of how
certain SnapRef behaviour was established. Decide per file; do not bulk-delete.
