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
"""Flax Anima text-to-image pipeline (Circlestone Anima-Base on MaxDiffusion).

Assembly of the verified components:
- Qwen3 text encoder (FlaxQwen3Model): last_hidden_state * attention mask
  (diffusers modular_pipelines/anima/encoders.py::_get_qwen_prompt_embeds).
- T5 ids + FlaxAnimaTextConditioner -> (B, 512, 1024) Cosmos context.
- FlaxAnimaCosmosTransformer (28 blocks, bf16) with the Anima schedule:
  raw sigmas linspace(1, 1/N, N), shift 3.0, terminal 0 (scheduler sigmas= path);
  transformer timestep = t/1000 fraction; no latent preconditioning;
  noise_pred straight into Euler step.
- CFG: standard uncond + guidance*(cond - uncond), guidance 4.0 default,
  two transformer evals per step.
- Qwen Image VAE via AutoencoderKLWan + qwen_image_vae_utils converter:
  latents/std + mean denormalization, decode, first frame.
"""

import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from maxdiffusion import max_logging
from maxdiffusion.models.anima_cosmos_flax import FlaxAnimaCosmosTransformer
from maxdiffusion.models.anima_text_conditioner_flax import (
  AnimaTextConditionerConfig,
  FlaxAnimaTextConditioner,
)
from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model
from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache


class FlaxAnimaPipeline:
  """Assembled Anima pipeline. Modules are Linen; params are pytrees."""

  def __init__(
      self,
      qwen3_model: FlaxQwen3Model,
      qwen3_params: dict,
      conditioner: FlaxAnimaTextConditioner,
      conditioner_params: dict,
      transformer: FlaxAnimaCosmosTransformer,
      transformer_params: dict,
      vae: AutoencoderKLWan,
      scheduler,
      vae_scale_factor: int = 8,
      dtype=jnp.bfloat16,
  ):
    self.qwen3_model = qwen3_model
    self.qwen3_params = qwen3_params
    self.conditioner = conditioner
    self.conditioner_params = conditioner_params
    self.transformer = transformer
    self.transformer_params = transformer_params
    self.vae = vae
    self.scheduler = scheduler
    self.vae_scale_factor = vae_scale_factor
    self.dtype = dtype
    self._jitted: dict = {}

  def _setup_jit_functions(self):
    if self._jitted:
      return

    @jax.jit
    def qwen3_forward(q_params, ids, mask):
      last_hidden, _ = self.qwen3_model.apply({"params": q_params}, input_ids=ids, attention_mask=mask)
      # Reference multiplies embeddings by the attention mask (encoders.py).
      return (last_hidden * mask.astype(last_hidden.dtype)[..., None]).astype(self.dtype)

    @jax.jit
    def conditioner_forward(c_params, source_hidden, target_ids):
      return self.conditioner.apply(
        {"params": c_params},
        source_hidden_states=source_hidden,
        target_input_ids=target_ids,
      ).astype(self.dtype)

    @jax.jit
    def transformer_forward(t_params, latents, timestep, context, padding_mask):
      return self.transformer.apply(
        {"params": t_params},
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=context,
        padding_mask=padding_mask,
      )

    def make_cfg_loop():
      @jax.jit
      def cfg_denoise_loop(t_params, latents, context, neg_context, padding_mask, timesteps, sigmas, guidance):
        sigmas_padded = jnp.concatenate([sigmas, jnp.zeros((1,), dtype=sigmas.dtype)])

        def scan_body(cur_latents, step_idx):
          t_val = timesteps[step_idx]
          t_vec = jnp.broadcast_to(t_val / 1000.0, (cur_latents.shape[0],)).astype(self.dtype)
          cur = cur_latents.astype(self.dtype)
          noise_cond = transformer_forward(t_params, cur, t_vec, context, padding_mask)
          noise_uncond = transformer_forward(t_params, cur, t_vec, neg_context, padding_mask)
          noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
          sigma = sigmas_padded[step_idx]
          sigma_next = sigmas_padded[step_idx + 1]
          next_latents = cur_latents + (sigma_next - sigma).astype(cur_latents.dtype) * noise_pred.astype(
            cur_latents.dtype
          )
          return next_latents, None

        final, _ = jax.lax.scan(scan_body, latents, jnp.arange(timesteps.shape[0]))
        return final

      return cfg_denoise_loop

    self._jitted = {
      "qwen3": qwen3_forward,
      "conditioner": conditioner_forward,
      "transformer": transformer_forward,
      "cfg_loop": make_cfg_loop(),
    }

  def encode_qwen3(self, ids: np.ndarray, mask: np.ndarray) -> jax.Array:
    self._setup_jit_functions()
    out = self._jitted["qwen3"](self.qwen3_params, jnp.asarray(ids), jnp.asarray(mask))
    return out

  def encode_conditioner(self, source_hidden: jax.Array, target_ids: np.ndarray) -> jax.Array:
    self._setup_jit_functions()
    return self._jitted["conditioner"](self.conditioner_params, source_hidden, jnp.asarray(target_ids))

  def decode_latents(self, latents: jax.Array) -> np.ndarray:
    """Denormalize (latents/std + mean), VAE-decode, return first-frame HWC uint8."""
    latents_mean = jnp.array(self.vae.latents_mean, dtype=latents.dtype).reshape(1, self.vae.z_dim, 1, 1, 1)
    latents_std = jnp.array(self.vae.latents_std, dtype=latents.dtype).reshape(1, self.vae.z_dim, 1, 1, 1)
    z = latents / (1.0 / latents_std) + latents_mean
    graphdef, state, rest = nnx.split(self.vae, nnx.Param, ...)
    merged = nnx.merge(graphdef, state, rest)
    video = merged.decode(z, AutoencoderKLWanCache(merged), return_dict=False)[0]
    video = jnp.clip(video / 2.0 + 0.5, 0.0, 1.0)
    img = np.asarray(video)
    # channels-last (B, T, H, W, C) -> first frame HWC
    if img.ndim == 5:
      img = img[:, 0]
    if img.shape[-1] not in (1, 3):
      img = np.moveaxis(img, 1, -1)
    return (img[0] * 255.0).round().astype(np.uint8)

  def __call__(
      self,
      qwen_embeds: jax.Array,
      qwen_mask: jax.Array,
      neg_qwen_embeds: jax.Array,
      t5_ids: np.ndarray,
      neg_t5_ids: np.ndarray,
      height: int,
      width: int,
      num_inference_steps: int = 30,
      guidance_scale: float = 4.0,
      seed: int = 0,
      trace: Optional[dict] = None,
  ) -> Image.Image:
    """Run the denoising loop + decode. Text encoding is done outside (or via encode_*)."""
    self._setup_jit_functions()
    t0 = time.perf_counter()
    context = self.encode_conditioner(qwen_embeds, t5_ids)
    neg_context = self.encode_conditioner(neg_qwen_embeds, neg_t5_ids)
    if trace is not None:
      trace["conditioning"] = time.perf_counter() - t0

    from maxdiffusion.schedulers.scheduling_flow_match_flax import FlowMatchSchedulerState

    sigmas_raw = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps).astype(np.float32)
    state = FlowMatchSchedulerState.create()
    state = self.scheduler.set_timesteps(state, sigmas=jnp.asarray(sigmas_raw))
    timesteps = np.asarray(state.timesteps)
    sigmas = np.asarray(state.sigmas[:-1])

    latent_h, latent_w = height // self.vae_scale_factor, width // self.vae_scale_factor
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((1, 16, 1, latent_h, latent_w)).astype(np.float32)
    padding_mask = np.zeros((1, 1, height, width), dtype=np.float32)

    t1 = time.perf_counter()
    denoised = self._jitted["cfg_loop"](
      self.transformer_params,
      jnp.asarray(latents),
      jnp.asarray(context),
      jnp.asarray(neg_context),
      jnp.asarray(padding_mask),
      jnp.asarray(timesteps),
      jnp.asarray(sigmas),
      jnp.asarray(np.float32(guidance_scale)),
    )
    denoised.block_until_ready()
    if trace is not None:
      trace["denoise"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    img_u8 = self.decode_latents(denoised)
    if trace is not None:
      trace["vae_decode"] = time.perf_counter() - t2
    return Image.fromarray(img_u8)
