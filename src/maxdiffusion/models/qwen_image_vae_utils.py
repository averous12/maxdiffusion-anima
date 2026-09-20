import os

import numpy as np
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .. import max_logging


def _wan_vae_key(key: str) -> str:
  """Rename chain from MaxDiffusion's load_wan_vae (structure only, no dtype renames)."""
  renamed = key
  renamed = renamed.replace("up_blocks_", "up_blocks.")
  renamed = renamed.replace("mid_block_", "mid_block.")
  renamed = renamed.replace("down_blocks_", "down_blocks.")
  renamed = renamed.replace("conv_in.bias", "conv_in.conv.bias")
  renamed = renamed.replace("conv_in.weight", "conv_in.conv.weight")
  renamed = renamed.replace("conv_out.bias", "conv_out.conv.bias")
  renamed = renamed.replace("conv_out.weight", "conv_out.conv.weight")
  renamed = renamed.replace("attentions_", "attentions.")
  renamed = renamed.replace("resnets_", "resnets.")
  renamed = renamed.replace("upsamplers_", "upsamplers.")
  renamed = renamed.replace("resample_", "resample.")
  renamed = renamed.replace("conv1.bias", "conv1.conv.bias")
  renamed = renamed.replace("conv1.weight", "conv1.conv.weight")
  renamed = renamed.replace("conv2.bias", "conv2.conv.bias")
  renamed = renamed.replace("conv2.weight", "conv2.conv.weight")
  renamed = renamed.replace("time_conv.bias", "time_conv.conv.bias")
  renamed = renamed.replace("time_conv.weight", "time_conv.conv.weight")
  renamed = renamed.replace("quant_conv", "quant_conv.conv")
  renamed = renamed.replace("conv_shortcut", "conv_shortcut.conv")
  if "decoder" in renamed:
    renamed = renamed.replace("resample.1.bias", "resample.layers.1.bias")
    renamed = renamed.replace("resample.1.weight", "resample.layers.1.weight")
  if "encoder" in renamed:
    renamed = renamed.replace("resample.1", "resample.conv")
  return renamed


def _convert_tensor(key: tuple, value: np.ndarray, target: dict) -> tuple:
  """Match one PyTorch tensor to a flat target leaf and lay it out Flax-style."""
  # Candidate final-component renames, tried in order.
  last = key[-1]
  candidates = [key]
  if last == "weight":
    if value.ndim >= 4:  # conv kernel: (o,i,k1,...,kn) -> (k1,...,kn,i,o)
      candidates.append(key[:-1] + ("kernel",))
    elif value.ndim == 2:  # linear: (o,i) -> (i,o)
      candidates.append(key[:-1] + ("kernel",))
  for cand in candidates:
    if cand in target:
      shape = target[cand]
      if value.ndim >= 4:
        value = np.transpose(value, tuple(range(2, value.ndim)) + (1, 0))
      elif value.ndim == 2:
        value = value.T
      if tuple(value.shape) != tuple(shape.shape):
        if value.ndim >= 1 and tuple(value.shape[-(value.ndim - value.ndim):]) != tuple(shape.shape) and value.size == int(np.prod(shape.shape)):
          value = value.reshape(shape.shape)
        else:
          raise ValueError(f"Shape mismatch for {cand}: ckpt {value.shape} vs target {shape}")
      return cand, value
  raise KeyError(f"No target leaf for converted key {key} (candidates {candidates})")


def load_qwen_image_vae(
    pretrained_model_name_or_path: str,
    eval_shapes: dict,
    device: str = "cpu",
    hf_download: bool = True,
):
  """Strictly load the official Qwen Image VAE into the WAN-compatible NNX tree.

  The architecture is intentionally supplied by AutoencoderKLWan; this function
  only handles Qwen's checkpoint naming and tensor layout.
  """
  if os.path.isdir(pretrained_model_name_or_path):
    path = os.path.join(pretrained_model_name_or_path, "vae", "diffusion_pytorch_model.safetensors")
  elif hf_download:
    path = hf_hub_download(pretrained_model_name_or_path, subfolder="vae", filename="diffusion_pytorch_model.safetensors")
  else:
    raise FileNotFoundError(pretrained_model_name_or_path)

  # Accept either a nested pure dict or an already-flat dict of shapes.
  if eval_shapes and not isinstance(next(iter(eval_shapes.values())), dict):
    flat_target = {tuple(str(x) for x in (k if isinstance(k, tuple) else tuple(k.split(".")))): v for k, v in eval_shapes.items()}
  else:
    flat_target = {tuple(str(x) for x in k): v for k, v in flatten_dict(eval_shapes).items()}

  converted = {}
  with safe_open(path, framework="pt", device="cpu") as tensors:
    for source_key in tensors.keys():
      renamed = _wan_vae_key(source_key)
      pt_tuple = tuple(renamed.split("."))
      value = tensors.get_tensor(source_key).float().numpy()
      flax_key, flax_value = _convert_tensor(pt_tuple, value, flat_target)
      converted[flax_key] = jnp.asarray(flax_value)

  unmatched = [k for k in converted if k not in flat_target]
  missing = [k for k in flat_target if k not in converted]
  if unmatched or missing:
    raise KeyError(
        f"Qwen VAE mapping incomplete: {len(unmatched)} unmatched (e.g. {unmatched[:5]}), "
        f"{len(missing)} unfilled (e.g. {missing[:5]})"
    )
  max_logging.log(f"Loaded and validated Qwen Image VAE tensors: {len(converted)} leaves")
  return unflatten_dict(converted)
