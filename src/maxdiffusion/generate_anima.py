# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate images with Anima (Circlestone Anima-Base) on MaxDiffusion + TPU.

Text encoding runs in PyTorch on CPU (Qwen3 last_hidden_state + T5 ids);
the diffusion loop and VAE decode run in JAX. Reports warm images/minute.

Example:
  python -m maxdiffusion.generate_anima \
    --pretrained_model_name_or_path=circlestone-labs/Anima-Base-v1.0-Diffusers \
    --prompt="masterpiece, best quality, 1girl, solo, city lights" \
    --negative_prompt="low quality, blurry" --num_inference_steps=30
"""

import gc
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from flax import nnx
from huggingface_hub import snapshot_download

try:
  import torch
except ImportError:
  torch = None

from maxdiffusion import max_logging, pyconfig
from maxdiffusion.models.anima_cosmos_flax import FlaxAnimaCosmosTransformer, convert_anima_cosmos_weights
from maxdiffusion.models.anima_text_conditioner_flax import (
  AnimaTextConditionerConfig,
  FlaxAnimaTextConditioner,
  load_and_convert_anima_text_conditioner_weights,
)
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model
from maxdiffusion.models.qwen3_utils import load_and_convert_qwen3_weights
from maxdiffusion.models.qwen_image_vae_utils import load_qwen_image_vae
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan
from maxdiffusion.pipelines.anima.anima_pipeline import FlaxAnimaPipeline
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler


def encode_texts(prompts, snapshot_dir, max_qwen_len=512, max_t5_len=512):
  """Qwen3 last_hidden_state (+mask) and T5 token ids via PyTorch CPU encoders."""
  from transformers import AutoModel, AutoTokenizer

  tok = AutoTokenizer.from_pretrained(os.path.join(snapshot_dir, "tokenizer"))
  t5_tok = AutoTokenizer.from_pretrained(os.path.join(snapshot_dir, "t5_tokenizer"))
  text_encoder = AutoModel.from_pretrained(
    os.path.join(snapshot_dir, "text_encoder"), torch_dtype=torch.float32
  ).eval()
  out = []
  with torch.no_grad():
    for prompt in prompts:
      text_inputs = tok(prompt, padding="max_length", max_length=max_qwen_len, truncation=True, return_tensors="pt")
      qwen_embeds = text_encoder(
        input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask
      ).last_hidden_state
      qwen_embeds = qwen_embeds * text_inputs.attention_mask.to(qwen_embeds.dtype).unsqueeze(-1)
      t5_inputs = t5_tok(
        prompt, padding="max_length", max_length=max_t5_len, truncation=True, return_tensors="pt"
      )
      out.append(
        (qwen_embeds.cpu().numpy(), text_inputs.attention_mask.cpu().numpy(), t5_inputs.input_ids.cpu().numpy(), t5_inputs.attention_mask.cpu().numpy())
      )
  del text_encoder
  gc.collect()
  return out


def main(argv):
  jax.config.update("jax_use_shardy_partitioner", True)
  pyconfig.initialize([None, "src/maxdiffusion/configs/base_anima.yml"])
  config = pyconfig.config
  os.makedirs(config.output_dir, exist_ok=True)

  repo_id = config.pretrained_model_name_or_path
  snapshot_dir = repo_id if os.path.exists(repo_id) else snapshot_download(repo_id=repo_id)
  max_logging.log(f"Model: {snapshot_dir} on {jax.devices()}")

  # ---- Qwen3 (config from checkpoint, Anima 0.6B-ish dims, not Klein defaults) ----
  from transformers import AutoConfig as HFAutoConfig

  pt_config = HFAutoConfig.from_pretrained(os.path.join(snapshot_dir, "text_encoder"))
  rope_theta = pt_config.rope_theta if hasattr(pt_config, "rope_theta") else pt_config.rope_parameters["rope_theta"]
  qwen_cfg = FlaxQwen3Config(
    vocab_size=pt_config.vocab_size,
    hidden_size=pt_config.hidden_size,
    intermediate_size=pt_config.intermediate_size,
    num_hidden_layers=pt_config.num_hidden_layers,
    num_attention_heads=pt_config.num_attention_heads,
    num_key_value_heads=pt_config.num_key_value_heads,
    head_dim=getattr(pt_config, "head_dim", pt_config.hidden_size // pt_config.num_attention_heads),
    rms_norm_eps=pt_config.rms_norm_eps,
    rope_theta=rope_theta,
    max_position_embeddings=pt_config.max_position_embeddings,
    dtype=jnp.bfloat16,
    max_layer_to_run=None,
    is_causal=True,
  )
  qwen3_model = FlaxQwen3Model(qwen_cfg)
  qwen3_vars = qwen3_model.init(
    jax.random.key(0), jnp.zeros((1, 8), dtype=jnp.int32), jnp.zeros((1, 8), dtype=jnp.int32)
  )
  qwen3_params = load_and_convert_qwen3_weights(
    os.path.join(snapshot_dir, "text_encoder"), qwen3_vars["params"], qwen_cfg
  )

  # ---- Conditioner ----
  cond_cfg = AnimaTextConditionerConfig(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
  conditioner = FlaxAnimaTextConditioner(cond_cfg)
  cond_vars = conditioner.init(
    jax.random.key(1), jnp.zeros((1, 8, 1024), dtype=jnp.bfloat16), np.zeros((1, 8), dtype=np.int32)
  )
  conditioner_params = load_and_convert_anima_text_conditioner_weights(
    os.path.join(snapshot_dir, "text_conditioner", "diffusion_pytorch_model.safetensors"),
    cond_vars["params"],
    dtype=jnp.bfloat16,
  )

  # ---- Transformer (28 blocks) ----
  transformer = FlaxAnimaCosmosTransformer(layers=28)
  t_vars = transformer.init(
    jax.random.key(2),
    jnp.zeros((1, 16, 1, 8, 8), dtype=jnp.bfloat16),
    jnp.zeros((1,), dtype=jnp.bfloat16),
    jnp.zeros((1, 8, 1024), dtype=jnp.bfloat16),
  )
  transformer_params = convert_anima_cosmos_weights(
    os.path.join(snapshot_dir, "transformer", "diffusion_pytorch_model.safetensors"),
    t_vars["params"],
    dtype=jnp.bfloat16,
    num_layers=28,
  )

  # ---- VAE (proven merge: string-path match + from_flat_path + update) ----
  from flax.traverse_util import flatten_dict as _flatten_dict
  vae = AutoencoderKLWan(nnx.Rngs(0), dtype=jnp.bfloat16, weights_dtype=jnp.bfloat16)
  _state = nnx.state(vae, nnx.Param)
  _flat_state = _state.flat_state()
  _flat_target = {k: v.value for k, v in _flat_state.items()}
  _converted = load_qwen_image_vae(repo_id, _flat_target)
  _conv_flat = _flatten_dict(_converted)
  _new_flat = {}
  _missing = []
  for _k, _vs in _flat_state.items():
    _p = "/".join(str(x) for x in _k)
    if _p in _conv_flat:
      _new_flat[_k] = _vs.replace(jnp.asarray(_conv_flat[_p], dtype=_vs.value.dtype))
    else:
      _missing.append(_p)
  if _missing:
    raise KeyError(f"VAE merge incomplete: {_missing[:8]}")
  nnx.update(vae, nnx.State.from_flat_path(_new_flat))

  scheduler = FlaxFlowMatchScheduler()
  pipeline = FlaxAnimaPipeline(
    qwen3_model, qwen3_params, conditioner, conditioner_params, transformer, transformer_params, vae, scheduler
  )

  prompts = [config.prompt]
  neg_prompts = [config.negative_prompt]
  texts = encode_texts(prompts + neg_prompts, snapshot_dir)
  (qwen_embeds, qwen_mask, t5_ids, t5_mask) = texts[0]
  (neg_embeds, neg_mask, neg_t5_ids, neg_t5_mask) = texts[1]
  _ = qwen_mask, neg_mask

  trace: dict = {}
  img = pipeline(
    jnp.asarray(qwen_embeds, dtype=jnp.bfloat16),
    jnp.asarray(qwen_mask),
    jnp.asarray(neg_embeds, dtype=jnp.bfloat16),
    jnp.asarray(neg_mask),
    t5_ids,
    t5_mask,
    neg_t5_ids,
    neg_t5_mask,
    height=config.height,
    width=config.width,
    num_inference_steps=config.num_inference_steps,
    guidance_scale=config.guidance_scale,
    seed=config.seed,
    trace=trace,
  )
  img.save(os.path.join(config.output_dir, "anima_warmup.png"))
  warmup_total = trace.get("conditioning", 0.0) + trace.get("denoise", 0.0) + trace.get("vae_decode", 0.0)
  max_logging.log(f"Warmup (compile+run): {warmup_total:.2f}s {trace}")

  times = []
  for rep in range(config.num_reps):
    t0 = time.perf_counter()
    trace = {}
    img = pipeline(
      jnp.asarray(qwen_embeds, dtype=jnp.bfloat16),
      jnp.asarray(qwen_mask),
      jnp.asarray(neg_embeds, dtype=jnp.bfloat16),
      t5_ids,
      neg_t5_ids,
      height=config.height,
      width=config.width,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=config.guidance_scale,
      seed=config.seed,
      trace=trace,
    )
    img.save(os.path.join(config.output_dir, f"anima_{rep}.png"))
    dt = time.perf_counter() - t0
    times.append(dt)
    max_logging.log(f"Rep {rep + 1}/{config.num_reps}: {dt:.2f}s {trace}")

  warm = min(times)
  max_logging.log(f"Warm {config.num_inference_steps}-step CFG image: {warm:.2f}s = {60.0 / warm:.2f} images/min")


def run_main(argv):
  main(argv)


if __name__ == "__main__":
  app.run(run_main)
