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
import re
import numpy as np
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


def merge_anima_loras_into_flat_params(flat_params, loras, dtype=jnp.bfloat16):
    """Merge one or more LoRAs into a flat linen parameter dict.

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
