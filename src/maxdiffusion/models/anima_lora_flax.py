"""
LoRA merger for linen (flat dict) parameters — Anima-Cosmos transformer.

The upstream nnx merger expects a module graph; our params are a flat dict
produced by `flax.traverse_util.flatten_dict`. This module works directly on
the flat dict, merging LoRA deltas in place before `nnx.update`/`unflatten_dict`.

Two on-disk LoRA formats are supported:

1) ComfyUI / AI-toolkit (trained Anima LoRAs):
     lora_unet_blocks_{i}_<suffix>.lora_down.weight  (rank, in)
     lora_unet_blocks_{i}_<suffix>.lora_up.weight    (out, rank)
     lora_unet_blocks_{i}_<suffix>.alpha             (scalar, optional)

2) diffusers / PEFT-style (Civitai "adaptive rank" releases):
     diffusion_model.blocks.{i}.<module>.lora_A.weight  (rank, in)
     diffusion_model.blocks.{i}.<module>.lora_B.weight  (out, rank)
     diffusion_model.blocks.{i}.<module>.alpha          (optional; often absent)

   In format 2 the module name uses dots (`adaln_modulation_self_attn.1`,
   `mlp.layer1`) and each module may carry a DIFFERENT rank — the "adaptive
   rank" convention. When `alpha` is absent the effective scale is 1.0, i.e.
   alpha == rank.

   `diffusion_model.llm_adapter.blocks.*` keys belong to the text-encoder
   adapter, not the diffusion transformer, and are intentionally ignored.

Our linen params (scanned) live under keys:
  ('scan', 'b', <rest>, 'kernel')  with shape (layers, out, in)

Mapping table (LoRA suffix -> linen rest):
  adaln_modulation_self_attn_1     -> ('norm1', 'linear_1')
  adaln_modulation_self_attn_2     -> ('norm1', 'linear_2')
  adaln_modulation_cross_attn_1    -> ('norm2', 'linear_1')
  adaln_modulation_cross_attn_2    -> ('norm2', 'linear_2')
  adaln_modulation_mlp_1           -> ('norm3', 'linear_1')
  adaln_modulation_mlp_2           -> ('norm3', 'linear_2')
  self_attn_q_proj                 -> ('attn1', 'to_q')
  self_attn_k_proj                 -> ('attn1', 'to_k')
  self_attn_v_proj                 -> ('attn1', 'to_v')
  self_attn_output_proj            -> ('attn1', 'to_out')
  cross_attn_q_proj                -> ('attn2', 'to_q')
  cross_attn_k_proj                -> ('attn2', 'to_k')
  cross_attn_v_proj                -> ('attn2', 'to_v')
  cross_attn_output_proj           -> ('attn2', 'to_out')
  mlp_layer1                       -> ('ff_in',)
  mlp_layer2                       -> ('ff_out',)

All target weights are 2D (Linear kernel). `up @ down` is (out, in) in the
PyTorch/diffusers convention, while a linen `Dense` kernel is (in, out), so the
delta is transposed before being added. The transpose is mandatory: without it
every non-square target (AdaLN modulations, cross-attn k/v, both MLP layers)
fails its shape check and is skipped, which still leaves the square attention
projections merged and makes the LoRA look like it "mostly worked".

MEMORY / MULTI-LORA
-------------------
Several LoRAs can be merged in a single pass. Merging them one file at a time
would copy every target tensor once per file; here each target tensor is copied
exactly once and every LoRA's delta is accumulated into it.

Tensors are streamed from the safetensors files on demand: the index pass reads
only key names (plus the tiny alpha scalars), so the resident LoRA cost is one
target's worth of deltas rather than a whole file. Params are stored back as
plain numpy arrays and the JAX array is released as soon as the copy exists, so
peak memory is roughly one copy of the largest single target tensor.
"""

import contextlib
import os
import re
import numpy as np
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors import safe_open


LORA_SUFFIX_TO_LINEN = {
    # AdaLN modulations: each has two linears (linear_1, linear_2)
    "adaln_modulation_self_attn_1": ("norm1", "linear_1"),
    "adaln_modulation_self_attn_2": ("norm1", "linear_2"),
    "adaln_modulation_cross_attn_1": ("norm2", "linear_1"),
    "adaln_modulation_cross_attn_2": ("norm2", "linear_2"),
    "adaln_modulation_mlp_1": ("norm3", "linear_1"),
    "adaln_modulation_mlp_2": ("norm3", "linear_2"),
    # Self-attention projections
    "self_attn_q_proj": ("attn1", "to_q"),
    "self_attn_k_proj": ("attn1", "to_k"),
    "self_attn_v_proj": ("attn1", "to_v"),
    "self_attn_output_proj": ("attn1", "to_out"),
    # Cross-attention projections
    "cross_attn_q_proj": ("attn2", "to_q"),
    "cross_attn_k_proj": ("attn2", "to_k"),
    "cross_attn_v_proj": ("attn2", "to_v"),
    "cross_attn_output_proj": ("attn2", "to_out"),
    # MLP (ff_in, ff_out)
    "mlp_layer1": ("ff_in",),
    "mlp_layer2": ("ff_out",),
}

# Legacy constant, kept for backwards compatibility with older callers/logs.
# Scaling now uses the per-module rank read from the tensor itself.
EXPECTED_LORA_RANK = 32

# Anima is 28 scanned layers x 16 modules.
EXPECTED_TARGETS_ANIMA = 448

# --- key patterns -----------------------------------------------------------
# ComfyUI / AI-toolkit
_COMFY_RE = re.compile(
    r"^lora_unet_blocks_(\d+)_(.+?)\.(lora_down|lora_up|alpha)(\.weight)?$")
# diffusers / PEFT (adaptive rank)
_DIFFUSERS_RE = re.compile(
    r"^diffusion_model\.blocks\.(\d+)\.(.+?)\.(lora_A|lora_B|alpha)(\.weight)?$")
# Text-encoder adapter: not part of the diffusion transformer.
_LLM_ADAPTER = "diffusion_model.llm_adapter."


def _suffix_from_diffusers_module(module):
    """'adaln_modulation_self_attn.1' -> 'adaln_modulation_self_attn_1'."""
    return module.replace(".", "_")


def detect_lora_format(path):
    """Return 'comfyui', 'diffusers', 'mixed' or 'unknown' for a LoRA file."""
    comfy = diffusers = 0
    with safe_open(path, framework="numpy") as s:
        for key in s.keys():
            if key.startswith(_LLM_ADAPTER):
                continue
            if _COMFY_RE.match(key):
                comfy += 1
            elif _DIFFUSERS_RE.match(key):
                diffusers += 1
    if comfy and diffusers:
        return "mixed"
    if comfy:
        return "comfyui"
    if diffusers:
        return "diffusers"
    return "unknown"


def _new_slot(targets, layer_idx, linen_rest):
    target_key = ("scan", "b") + linen_rest + ("kernel",)
    targets.setdefault(target_key, {})
    return targets[target_key].setdefault(
        layer_idx, {"down": None, "up": None, "alpha": None})


def _index_lora(handle):
    """Read a LoRA's key layout WITHOUT materialising tensor data.

    Returns {"targets": {target_key: {layer_idx: {"down","up","alpha"}}},
             "format": str}

    `down`/`up` are tensor key names to fetch later; `alpha` is the real value
    (a scalar costs nothing to read).
    """
    targets = {}
    formats = set()

    for key in handle.keys():
        if key.startswith(_LLM_ADAPTER):
            continue

        m = _COMFY_RE.match(key)
        if m:
            layer_idx, suffix, kind = int(m.group(1)), m.group(2), m.group(3)
            if suffix not in LORA_SUFFIX_TO_LINEN:
                continue
            formats.add("comfyui")
            slot = _new_slot(targets, layer_idx, LORA_SUFFIX_TO_LINEN[suffix])
            if kind == "lora_down":
                slot["down"] = key
            elif kind == "lora_up":
                slot["up"] = key
            else:
                slot["alpha"] = float(np.asarray(handle.get_tensor(key)).reshape(-1)[0])
            continue

        m = _DIFFUSERS_RE.match(key)
        if m:
            layer_idx = int(m.group(1))
            suffix = _suffix_from_diffusers_module(m.group(2))
            kind = m.group(3)
            if suffix not in LORA_SUFFIX_TO_LINEN:
                continue
            formats.add("diffusers")
            slot = _new_slot(targets, layer_idx, LORA_SUFFIX_TO_LINEN[suffix])
            if kind == "lora_A":
                slot["down"] = key
            elif kind == "lora_B":
                slot["up"] = key
            else:
                slot["alpha"] = float(np.asarray(handle.get_tensor(key)).reshape(-1)[0])
            continue

    return {"targets": targets,
            "format": "+".join(sorted(formats)) if formats else "unknown"}


def _normalise_loras(loras):
    """Accept a path, (path, scale), or a list of either."""
    if loras is None:
        return []
    if isinstance(loras, (str, bytes)):
        return [(loras, 1.0)]
    if isinstance(loras, tuple) and len(loras) == 2 and isinstance(loras[0], str):
        return [(loras[0], float(loras[1]))]
    out = []
    for item in loras:
        if isinstance(item, (str, bytes)):
            out.append((item, 1.0))
        else:
            path, scale = item
            out.append((path, float(scale)))
    return out


def _parse_anima_lora_file(path):
    """Compatibility helper: materialise one LoRA's deltas as arrays.

    Kept for tests and callers that want everything in memory. The merge path
    does NOT use this — it streams per target so resident memory stays flat.
    """
    parsed = {}
    with safe_open(path, framework="numpy") as s:
        idx = _index_lora(s)
        for target_key, per_layer in idx["targets"].items():
            for layer_idx, e in per_layer.items():
                entry = {"down": None, "up": None, "alpha": e["alpha"]}
                if e["down"] is not None:
                    entry["down"] = s.get_tensor(e["down"]).astype(np.float32)
                if e["up"] is not None:
                    entry["up"] = s.get_tensor(e["up"]).astype(np.float32)
                parsed.setdefault(layer_idx, {})[target_key] = entry
    return parsed, ({idx["format"]} if idx["format"] != "unknown" else set())


def _merge_anima_loras_numpy(flat_params, loras, dtype=jnp.bfloat16):
    """REFERENCE implementation — merge one or more LoRAs into a flat dict.

    Kept verbatim as the oracle for the CPU gate tests (`backend="numpy"`).
    It is not the production path: every target is copied and up-cast to
    float32 with `np.asarray(...).astype(np.float32)` on a per-layer slice,
    and numpy has no native bfloat16 — that cast runs through ml_dtypes'
    scalar loop with no SIMD, which is what made a 470 M-element merge cost
    112 s on the TPU boot path. See `_merge_anima_loras_jax` for the fast path.

    Args:
        flat_params: dict from flatten_dict(model_params). MUTATED in place;
            touched entries are replaced with numpy arrays in `dtype`.
        loras: [(path, scale), ...] — a bare path or (path, scale) also works.
        dtype: dtype for the merged weights.

    Returns:
        (flat_params, stats)

    One pass over the target tensors and one copy per tensor, regardless of how
    many LoRAs are supplied.
    """
    specs = _normalise_loras(loras)
    specs = [(p, s) for p, s in specs if p]
    if not specs:
        return flat_params, {"merged": 0, "skipped": 0, "format": "unknown",
                             "error": "no LoRA paths"}

    merged = 0
    skipped = 0
    details = []
    ranks = set()
    used = []
    formats = set()
    all_targets = []

    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(safe_open(p, framework="numpy")) for p, _ in specs]
        indexes = [_index_lora(h) for h in handles]
        formats.update(idx["format"] for idx in indexes)

        # union of target keys, in a stable order
        seen = set()
        for idx in indexes:
            for tk in idx["targets"]:
                if tk not in seen:
                    seen.add(tk)
                    all_targets.append(tk)

        for target_key in all_targets:
            param = flat_params.get(target_key)
            if param is None:
                skipped += sum(len(idx["targets"].get(target_key, {})) for idx in indexes)
                details.append(f"missing param: {target_key}")
                continue
            if param.ndim != 3:
                skipped += 1
                details.append(f"bad shape {target_key}: {param.shape} (expected 3D)")
                continue

            # One copy, then release the JAX array: replacing the dict entry
            # drops the last reference, so the original and the copy are never
            # both resident beyond this point.
            buf = np.array(param)
            flat_params[target_key] = buf
            del param

            slice_shape = tuple(buf.shape[1:])
            n_layers = buf.shape[0]
            touched = 0

            for layer_idx in range(n_layers):
                acc = None
                for idx, handle, (_, lora_scale) in zip(indexes, handles, specs):
                    entry = idx["targets"].get(target_key, {}).get(layer_idx)
                    if not entry or entry["down"] is None or entry["up"] is None:
                        continue

                    down = handle.get_tensor(entry["down"]).astype(np.float32)
                    up = handle.get_tensor(entry["up"]).astype(np.float32)
                    rank = down.shape[0]
                    ranks.add(rank)
                    # Standard LoRA scaling: alpha / rank. Missing alpha == rank.
                    a = entry["alpha"] if entry["alpha"] is not None else float(rank)

                    delta = up @ down                  # (out, in)
                    delta *= (lora_scale * a / rank)   # in place, no extra tensor
                    if delta.T.shape == slice_shape:
                        add = delta.T            # (in, out): correct for linen
                        used.append("T")
                    elif delta.shape == slice_shape:
                        add = delta
                        used.append("raw")
                    else:
                        skipped += 1
                        details.append(
                            f"delta {delta.shape} (or T {delta.T.shape}) != param "
                            f"slice {slice_shape} for {target_key}[{layer_idx}]")
                        continue

                    if acc is None:
                        acc = buf[layer_idx].astype(np.float32)
                    acc += add

                if acc is not None:
                    try:
                        np.copyto(buf[layer_idx], acc, casting="unsafe")
                    except (TypeError, ValueError):
                        buf[layer_idx] = acc.astype(buf.dtype)
                    merged += 1
                    touched += 1

            if not touched:
                details.append(f"no delta applied to {target_key}")

    return flat_params, {
        "merged": merged,
        "skipped": skipped,
        "targets": len(all_targets),
        "loras": [{"path": p, "scale": s} for p, s in specs],
        "ranks": sorted(ranks),
        "convention": sorted(set(used)),
        "format": "+".join(sorted(formats)),
        "details": details,
    }


def _read_lora_tensor(handle, key):
    """Fetch one LoRA tensor WITHOUT an elementwise cast.

    safetensors returns the stored dtype, and a same-dtype copy is a memcpy.
    Casting here would be the slow path whenever the file is bfloat16, so every
    cast is deferred to a single XLA op over a whole stacked target.
    """
    return np.asarray(handle.get_tensor(key))


def _merge_anima_loras_jax(flat_params, handles, indexes, specs):
    """Production merge: batched einsum + XLA casts, no numpy bf16 round-trip.

    Differences from `_merge_anima_loras_numpy`, all of them speed-only:

    * the per-layer Python loop is replaced by one einsum over stacked
      (L, out, R) x (L, R, in) factors, with ranks zero-padded to this LoRA's
      largest rank for the target (zero rows/cols leave `up @ down` exact);
    * the base tensor is up-cast once per target inside XLA instead of once per
      layer through numpy, and the result is down-cast once inside XLA;
    * target shapes are read from the safetensors headers, so no tensor is
      fetched twice.

    The transpose convention is resolved exactly as the reference does — the
    transposed layout is checked FIRST, so square targets keep using 'T'.
    """
    merged = 0
    skipped = 0
    details = []
    ranks = set()
    used = []
    formats = set(idx["format"] for idx in indexes)

    all_targets = []
    seen = set()
    for idx in indexes:
        for tk in idx["targets"]:
            if tk not in seen:
                seen.add(tk)
                all_targets.append(tk)

    # A TPU-only jax build has no CPU backend at all ("Unknown backend cpu"), and
    # the project's .venv is exactly that, so fall back to whatever device exists
    # rather than assuming a CPU one is present.
    try:
        cpu = jax.devices("cpu")[0]
    except (RuntimeError, ValueError):
        cpu = jax.devices()[0]

    for target_key in all_targets:
        param = flat_params.get(target_key)
        if param is None:
            skipped += sum(len(idx["targets"].get(target_key, {})) for idx in indexes)
            details.append(f"missing param: {target_key}")
            continue

        param_np = np.asarray(param)
        if param_np.ndim != 3:
            skipped += 1
            details.append(f"bad shape {target_key}: {param_np.shape} (expected 3D)")
            continue

        n_layers = param_np.shape[0]
        slice_shape = tuple(param_np.shape[1:])
        base_dtype = param_np.dtype

        # Shapes come from the file headers — no tensor data is read for this.
        in_dim = out_dim = None
        for idx, handle in zip(indexes, handles):
            for e in (idx["targets"].get(target_key) or {}).values():
                if e["down"] and e["up"]:
                    in_dim = int(handle.get_slice(e["down"]).get_shape()[1])
                    out_dim = int(handle.get_slice(e["up"]).get_shape()[0])
                    break
            if in_dim is not None:
                break
        if in_dim is None:
            continue

        # Stack each LoRA's factors for this target once (a few MB), then walk the
        # layers in chunks. A whole-target float32 delta for an ff layer is
        # ~1.9 GB; materialising that plus its transpose and sum makes an
        # unbounded batched merge SLOWER than the per-slice reference — measured
        # at 0.6x on the 1937.8 M-element model. Chunking keeps the float32
        # working set at a few hundred MB while every cast still happens inside
        # XLA instead of through ml_dtypes.
        chunk = max(1, int(os.environ.get("ANIMA_MERGE_CHUNK", "4")))
        prepared = []

        for idx, handle, (_, lora_scale) in zip(indexes, handles, specs):
            per_layer = {
                li: e for li, e in (idx["targets"].get(target_key) or {}).items()
                if e["down"] is not None and e["up"] is not None
            }
            if not per_layer:
                continue

            entry_ranks = {}
            for li, e in per_layer.items():
                r = int(handle.get_slice(e["down"]).get_shape()[0])
                entry_ranks[li] = r
                ranks.add(r)
            max_rank = max(entry_ranks.values())

            first = _read_lora_tensor(handle, next(iter(per_layer.values()))["down"])
            stack_dtype = first.dtype
            up = np.zeros((n_layers, out_dim, max_rank), dtype=stack_dtype)
            dn = np.zeros((n_layers, max_rank, in_dim), dtype=stack_dtype)
            scale_vec = np.zeros((n_layers,), np.float32)
            ok_layers = []

            for li, e in per_layer.items():
                d = _read_lora_tensor(handle, e["down"])   # (rank, in)
                u = _read_lora_tensor(handle, e["up"])     # (out, rank)
                r = d.shape[0]
                if d.shape != (r, in_dim) or u.shape != (out_dim, r):
                    skipped += 1
                    details.append(
                        f"delta {u.shape}@{d.shape} != param slice {slice_shape} "
                        f"for {target_key}[{li}]")
                    continue
                # Same dtype on both sides -> plain memcpy, no ml_dtypes loop.
                dn[li, :r] = d
                up[li, :, :r] = u
                a = e["alpha"] if e["alpha"] is not None else float(r)
                scale_vec[li] = lora_scale * a / r
                ok_layers.append(li)

            if ok_layers:
                prepared.append((up, dn, scale_vec, ok_layers))

        if not prepared:
            details.append(f"no delta applied to {target_key}")
            continue

        # Where the arithmetic runs. "cpu" keeps the production behaviour: params
        # are still host-resident at this point and go to the device right after.
        # "tpu"/"device" keeps each target on the accelerator end to end, which is
        # the point of the exercise — bf16 is native there, so the emulated casts
        # that dominate on CPU disappear.
        use_device = os.environ.get("ANIMA_MERGE_DEVICE", "cpu").lower() in (
            "tpu", "device", "gpu")
        if use_device:
            try:
                dev = jax.devices()[0]
            except Exception:
                dev = cpu
        else:
            dev = cpu

        def _chunk_delta(up, dn, scale_vec, sl):
            with jax.default_device(dev):
                # Batched matmul, not einsum: on the dominant ff shape (28x8192x2048,
                # rank 32) einsum reaches 15 GFLOP/s while jnp.matmul reaches 58 —
                # a measured 3.4x on the product, which is the single largest term
                # in the merge. Both compute (L,out,r) @ (L,r,in) -> (L,out,in).
                d = jnp.matmul(jnp.asarray(up[sl]).astype(jnp.float32),
                               jnp.asarray(dn[sl]).astype(jnp.float32))
                return d * jnp.asarray(scale_vec[sl])[:, None, None]

        # The layout is uniform across layers, so resolve it from layer 0 once.
        # 'T' is checked first, exactly like the reference: square targets are
        # transposed, and transposing vs not is not a no-op.
        probe = _chunk_delta(prepared[0][0], prepared[0][1], prepared[0][2],
                             slice(0, 1))
        if tuple(probe.transpose(0, 2, 1).shape[1:]) == slice_shape:
            transpose = True
        elif tuple(probe.shape[1:]) == slice_shape:
            transpose = False
        else:
            skipped += sum(len(ok) for *_, ok in prepared)
            details.append(
                f"delta {tuple(probe.shape)} (or T) != param slice "
                f"{slice_shape} for {target_key}")
            continue
        used.append("T" if transpose else "raw")

        if use_device:
            # Device-resident: upload each target once, do every chunk's matmul,
            # cast and add on the accelerator, download once. bf16 is native here,
            # so the emulated-cast cost that dominates on CPU does not exist.
            with jax.default_device(dev):
                base_dev = jnp.asarray(param_np)
                parts = []
                for start in range(0, n_layers, chunk):
                    sl = slice(start, min(start + chunk, n_layers))
                    acc = None
                    for up, dn, scale_vec, _ok in prepared:
                        d = _chunk_delta(up, dn, scale_vec, sl)
                        acc = d if acc is None else acc + d
                    if acc is None:
                        parts.append(base_dev[sl])
                        continue
                    if transpose:
                        acc = acc.transpose(0, 2, 1)
                    merged_chunk = (base_dev[sl].astype(jnp.float32) + acc)
                    parts.append(merged_chunk.astype(base_dtype))
                combined = jnp.concatenate(parts, axis=0)
                # Callers that hand the result straight to the accelerator (the
                # live-swap path) do not need this download at all; set
                # ANIMA_MERGE_KEEP_DEVICE=1 to keep the merged weights resident.
                out = (combined if os.environ.get("ANIMA_MERGE_KEEP_DEVICE")
                       else np.asarray(combined))

            flat_params[target_key] = out
            merged += len({li for *_, ok in prepared for li in ok})
            continue

        out = np.array(param_np)
        for start in range(0, n_layers, chunk):
            sl = slice(start, min(start + chunk, n_layers))
            acc = None
            for up, dn, scale_vec, _ok in prepared:
                d = _chunk_delta(up, dn, scale_vec, sl)
                acc = d if acc is None else acc + d
            if acc is None:
                continue
            # XLA wins the batched matmul (58 vs 38 GFLOP/s), so the product stays
            # on XLA. The casts are done here in numpy: standalone they measure
            # bf16->fp32 at 1.05 ns/elem and fp32->bf16 at 2.73, against XLA's
            # 10-20. End to end, though, this split measured NEUTRAL (32.8 vs
            # 31.9 s across runs, inside the +-5 s spread), so it is kept for the
            # shorter data path, not as a speed claim.
            acc_np = np.asarray(acc)
            if transpose:
                acc_np = acc_np.transpose(0, 2, 1)
            base_chunk = out[sl].astype(np.float32)
            base_chunk += acc_np
            out[sl] = base_chunk.astype(base_dtype)

        flat_params[target_key] = out
        merged += len({li for *_, ok in prepared for li in ok})

    return flat_params, {
        "merged": merged,
        "skipped": skipped,
        "targets": len(all_targets),
        "loras": [{"path": p, "scale": s} for p, s in specs],
        "ranks": sorted(ranks),
        "convention": sorted(set(used)),
        "format": "+".join(sorted(formats)),
        "details": details,
        "backend": "jax",
    }


def merge_anima_loras_into_flat_params(flat_params, loras, dtype=jnp.bfloat16,
                                       backend="jax"):
    """Merge one or more LoRAs into a flat linen parameter dict.

    Args:
        flat_params: dict from flatten_dict(model_params). MUTATED in place;
            touched entries are replaced, keeping each array's own dtype.
        loras: [(path, scale), ...] — a bare path or (path, scale) also works.
        dtype: accepted for API compatibility; the stored dtype follows the
            input array (unchanged from the reference implementation).
        backend: "jax" (default) merges in XLA — one einsum per chunk of layers
            and vectorised casts each way, with the float32 working set bounded
            by ANIMA_MERGE_CHUNK (default 4). "numpy" runs the reference loop,
            kept so the gate tests have an independent oracle.

    MEASURED (2026-11-04, CPU, both real LoRA files, 1937.8 M elements, bf16):
    the reference runs 39-56 s (run-to-run spread is +-5 s on this box); this
    path lands at 31.9 s with chunk=4, i.e. ~1.5x, and 38.6 s when the product
    was still an einsum. Coverage, transpose convention and skip counts are
    identical; max |diff| is 1.907e-06, i.e. last-bit bf16 drift from a different
    summation order.

    Where the time actually goes, all measured (bench/bench_bf16_conv.py and
    bench/bench_rank_gemm.py): bf16<->fp32 conversion is only 3.78 ns/element
    (1.05 up, 2.73 down), and the low-rank product is ~4.5 s for two LoRAs once
    it is a batched matmul instead of an einsum (58 vs 15 GFLOP/s). Those two
    account for roughly 12 s of the 32; the remainder is memory traffic across
    the ~6 passes a merge makes over 1.94 B elements. The casts and the product
    were both wrong first guesses. Getting to seconds needs the merge to run
    where bf16 is native (the TPU), or no merge at all.

    Returns:
        (flat_params, stats)

    One pass over the target tensors and one copy per tensor, regardless of how
    many LoRAs are supplied.
    """
    specs = _normalise_loras(loras)
    specs = [(p, s) for p, s in specs if p]
    if not specs:
        return flat_params, {"merged": 0, "skipped": 0, "format": "unknown",
                             "error": "no LoRA paths", "backend": backend}

    if backend == "numpy":
        return _merge_anima_loras_numpy(flat_params, specs, dtype=dtype)
    if backend != "jax":
        raise ValueError(f"unknown backend {backend!r} (expected 'jax' or 'numpy')")

    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(safe_open(p, framework="numpy"))
                   for p, _ in specs]
        indexes = [_index_lora(h) for h in handles]
        return _merge_anima_loras_jax(flat_params, handles, indexes, specs)


def merge_anima_lora_into_flat_params(flat_params, lora_path, scale=1.0, dtype=jnp.bfloat16):
    """Single-LoRA wrapper kept for existing callers."""
    return merge_anima_loras_into_flat_params(flat_params, [(lora_path, scale)], dtype=dtype)


def load_and_merge_anima_loras(model_params, loras, dtype=jnp.bfloat16):
    """Flatten, merge one or more LoRAs, unflatten.

    Args:
        model_params: nested pytree (from nnx.State or linen init). Must be the
            NESTED tree — this function flattens internally, so an already-flat
            dict would double-flatten its tuple keys.
        loras: [(path, scale), ...] or a single path.
        dtype: dtype for the merged weights.

    Returns:
        (new_model_params, stats)
    """
    flat = flatten_dict(model_params)
    flat, stats = merge_anima_loras_into_flat_params(flat, loras, dtype=dtype)
    return unflatten_dict(flat), stats


def load_and_merge_anima_lora(model_params, lora_path, scale=1.0, dtype=jnp.bfloat16):
    """Single-LoRA wrapper kept for existing callers."""
    return load_and_merge_anima_loras(model_params, [(lora_path, scale)], dtype=dtype)
