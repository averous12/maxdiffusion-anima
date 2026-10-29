"""
LoRA merger for linen (flat dict) parameters — Anima-Cosmos transformer.

The upstream nnx merger expects a module graph; our params are a flat dict
produced by `flax.traverse_util.flatten_dict`. This module works directly on
the flat dict, merging LoRA deltas in place before `nnx.update`/`unflatten_dict`.

LoRA file format (from trained Anima LoRAs on disk):
  lora_unet_blocks_{i}_<suffix>.lora_down.weight  (rank, in)
  lora_unet_blocks_{i}_<suffix>.lora_up.weight    (out, rank)
  lora_unet_blocks_{i}_<suffix>.alpha             (scalar)

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

# All are 2D Linear kernels; no convs in Anima transformer
EXPECTED_LORA_RANK = 32  # from the trained files; can be inferred


def _parse_anima_lora_file(path):
    """
    Parse an Anima LoRA safetensors file into a dict:
      {layer_idx: {linen_rest_tuple: (down, up, alpha)}}
    """
    parsed = {}
    with safe_open(path, framework="numpy") as s:
        for key in s.keys():
            # Key format: lora_unet_blocks_{i}_{suffix}.{lora_down|lora_up|alpha}.weight?
            m = re.match(r"^lora_unet_blocks_(\d+)_(.+?)\.(lora_down|lora_up|alpha)(\.weight)?$", key)
            if not m:
                # Could be non-block keys; ignore for now
                continue
            layer_idx = int(m.group(1))
            suffix = m.group(2)
            kind = m.group(3)

            if suffix not in LORA_SUFFIX_TO_LINEN:
                # Unknown suffix — could be from a different architecture
                continue

            linen_rest = LORA_SUFFIX_TO_LINEN[suffix]
            target_key = ("scan", "b") + linen_rest + ("kernel",)

            if layer_idx not in parsed:
                parsed[layer_idx] = {}
            if target_key not in parsed[layer_idx]:
                parsed[layer_idx][target_key] = {"down": None, "up": None, "alpha": None}

            val = s.get_tensor(key)
            if kind == "lora_down":
                parsed[layer_idx][target_key]["down"] = val.astype(np.float32)
            elif kind == "lora_up":
                parsed[layer_idx][target_key]["up"] = val.astype(np.float32)
            elif kind == "alpha":
                parsed[layer_idx][target_key]["alpha"] = float(val.item())

    return parsed


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
    parsed = _parse_anima_lora_file(lora_path)
    if not parsed:
        return flat_params, {"merged": 0, "skipped": 0, "error": "no matching keys"}

    merged = 0
    skipped = 0
    details = []

    for layer_idx, targets in parsed.items():
        for target_key, parts in targets.items():
            if target_key not in flat_params:
                skipped += 1
                details.append(f"missing param: {target_key}")
                continue

            down = parts["down"]
            up = parts["up"]
            alpha = parts["alpha"] if parts["alpha"] is not None else EXPECTED_LORA_RANK

            if down is None or up is None:
                skipped += 1
                details.append(f"incomplete LoRA for {target_key}")
                continue

            # LoRA delta: up (out, rank) @ down (rank, in) -> (out, in)
            # down shape: (rank, in), up shape: (out, rank)
            delta = up @ down  # (out, in)
            delta = delta * (scale * alpha / EXPECTED_LORA_RANK)

            # Our scanned param has shape (28, out, in) — add to the layer_idx slice
            param = flat_params[target_key]
            if param.ndim != 3 or param.shape[0] <= layer_idx:
                skipped += 1
                details.append(f"shape mismatch {target_key}: {param.shape} vs layer {layer_idx}")
                continue

            # Update in place (param is a jax.Array; we need to create new array)
            # Since jax arrays are immutable, we replace the entry in the dict
            param_np = np.array(param)
            param_np[layer_idx] = param_np[layer_idx] + delta.astype(param_np.dtype)
            flat_params[target_key] = jnp.asarray(param_np, dtype=dtype)
            merged += 1
            details.append(f"merged {target_key}[{layer_idx}] scale={scale*alpha/EXPECTED_LORA_RANK:.4f}")

    return flat_params, {
        "merged": merged,
        "skipped": skipped,
        "layers_touched": len(parsed),
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