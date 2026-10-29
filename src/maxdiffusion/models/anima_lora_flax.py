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
  ('scan', 'b', <rest>, 'kernel')  with shape (28, out, in) or (28, out, in, kH, kW)

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

All target weights are 2D (Linear kernel) so LoRA delta = up @ down  (out, in)
"""

import re
import jax.numpy as jnp
import numpy as np
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


def _parse_anima_lora_file(path):
    """
    Parse an Anima LoRA safetensors file into a dict:
      {layer_idx: {linen_rest_tuple: {"down": arr, "up": arr, "alpha": float|None}}}
    Also returns the set of detected formats.
    """
    parsed = {}
    formats = set()

    def _slot(layer_idx, linen_rest):
        target_key = ("scan", "b") + linen_rest + ("kernel",)
        parsed.setdefault(layer_idx, {})
        parsed[layer_idx].setdefault(
            target_key, {"down": None, "up": None, "alpha": None})
        return parsed[layer_idx][target_key]

    with safe_open(path, framework="numpy") as s:
        for key in s.keys():
            if key.startswith(_LLM_ADAPTER):
                continue

            m = _COMFY_RE.match(key)
            if m:
                layer_idx, suffix, kind = int(m.group(1)), m.group(2), m.group(3)
                if suffix not in LORA_SUFFIX_TO_LINEN:
                    continue
                formats.add("comfyui")
                slot = _slot(layer_idx, LORA_SUFFIX_TO_LINEN[suffix])
                val = s.get_tensor(key)
                if kind == "lora_down":
                    slot["down"] = val.astype(np.float32)
                elif kind == "lora_up":
                    slot["up"] = val.astype(np.float32)
                else:
                    slot["alpha"] = float(np.asarray(val).reshape(-1)[0])
                continue

            m = _DIFFUSERS_RE.match(key)
            if m:
                layer_idx = int(m.group(1))
                suffix = _suffix_from_diffusers_module(m.group(2))
                kind = m.group(3)
                if suffix not in LORA_SUFFIX_TO_LINEN:
                    continue
                formats.add("diffusers")
                slot = _slot(layer_idx, LORA_SUFFIX_TO_LINEN[suffix])
                val = s.get_tensor(key)
                if kind == "lora_A":
                    slot["down"] = val.astype(np.float32)
                elif kind == "lora_B":
                    slot["up"] = val.astype(np.float32)
                else:
                    slot["alpha"] = float(np.asarray(val).reshape(-1)[0])
                continue

    return parsed, formats


def merge_anima_lora_into_flat_params(flat_params, lora_path, scale=1.0, dtype=jnp.bfloat16):
    """
    Merge LoRA weights into a flat linen parameter dict.

    Args:
        flat_params: dict from flatten_dict(model_params) — will be MUTATED.
        lora_path: path to .safetensors LoRA file.
        scale: global LoRA scale multiplier.
        dtype: target dtype for the merged weights (params are updated in this dtype).

    Returns:
        (updated_flat_params, stats_dict)
    """
    parsed, formats = _parse_anima_lora_file(lora_path)
    if not parsed:
        return flat_params, {"merged": 0, "skipped": 0, "format": "unknown",
                             "error": "no matching keys"}

    # Regroup by target param so each scanned tensor is copied exactly once.
    # Layout: {target_key: {layer_idx: {"down","up","alpha"}}}
    by_target = {}
    for layer_idx, targets in parsed.items():
        for target_key, parts in targets.items():
            by_target.setdefault(target_key, {})[layer_idx] = parts

    merged = 0
    skipped = 0
    details = []
    ranks = set()

    for target_key, per_layer in by_target.items():
        if target_key not in flat_params:
            skipped += len(per_layer)
            details.append(f"missing param: {target_key}")
            continue

        param = flat_params[target_key]
        if param.ndim != 3:
            skipped += len(per_layer)
            details.append(f"shape mismatch {target_key}: {param.shape} (expected 3D scanned)")
            continue

        out = np.array(param)  # exactly one copy per scanned tensor
        slice_shape = tuple(param.shape[1:])
        touched = 0

        for layer_idx, parts in sorted(per_layer.items()):
            down, up, alpha = parts["down"], parts["up"], parts["alpha"]
            if down is None or up is None:
                skipped += 1
                details.append(f"incomplete LoRA for {target_key}[{layer_idx}]")
                continue
            if layer_idx >= param.shape[0]:
                skipped += 1
                details.append(f"layer {layer_idx} out of range for {target_key}")
                continue

            rank = down.shape[0]
            ranks.add(rank)
            # Standard LoRA scaling: alpha / rank. Missing alpha == rank -> factor 1.
            a = alpha if alpha is not None else float(rank)

            delta = up @ down  # (out, in)
            if delta.shape != slice_shape:
                skipped += 1
                details.append(
                    f"delta {delta.shape} != param slice {slice_shape} for "
                    f"{target_key}[{layer_idx}]")
                continue

            base = param[layer_idx].astype(np.float32)
            out[layer_idx] = (base + delta * (scale * a / rank)).astype(out.dtype)
            merged += 1
            touched += 1

        if touched:
            flat_params[target_key] = jnp.asarray(out, dtype=dtype)

    return flat_params, {
        "merged": merged,
        "skipped": skipped,
        "layers_touched": len(parsed),
        "targets": len(by_target),
        "ranks": sorted(ranks),
        "format": "+".join(sorted(formats)) if formats else "unknown",
        "details": details,
    }


def load_and_merge_anima_lora(model_params, lora_path, scale=1.0, dtype=jnp.bfloat16):
    """
    High-level helper: flatten, merge, unflatten.

    Args:
        model_params: nested pytree (from nnx.State or linen init)
        lora_path: path to .safetensors
        scale: global LoRA scale
        dtype: dtype for merged weights

    Returns:
        new_model_params (nested pytree, same structure as input)
    """
    flat = flatten_dict(model_params)
    flat, stats = merge_anima_lora_into_flat_params(flat, lora_path, scale, dtype)
    return unflatten_dict(flat), stats