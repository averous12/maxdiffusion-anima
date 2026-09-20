import jax
import jax.numpy as jnp

from maxdiffusion.models.anima_text_conditioner_flax import AnimaTextConditionerConfig, FlaxAnimaTextConditioner


def test_anima_text_conditioner_pads_to_512_tokens():
  config = AnimaTextConditionerConfig(
      source_dim=8,
      target_dim=8,
      model_dim=8,
      num_layers=1,
      num_attention_heads=2,
      target_vocab_size=32,
      min_sequence_length=512,
  )
  model = FlaxAnimaTextConditioner(config)
  variables = model.init(
      jax.random.key(0),
      jnp.ones((1, 5, 8), dtype=jnp.float32),
      jnp.ones((1, 7), dtype=jnp.int32),
      jnp.ones((1, 5), dtype=jnp.bool_),
      jnp.ones((1, 7), dtype=jnp.bool_),
  )
  output = model.apply(variables, jnp.ones((1, 5, 8)), jnp.ones((1, 7), dtype=jnp.int32),
                       jnp.ones((1, 5), dtype=jnp.bool_), jnp.ones((1, 7), dtype=jnp.bool_))
  assert output.shape == (1, 512, 8)
  assert output.dtype == jnp.bfloat16


def test_anima_text_conditioner_masks_padded_source_and_target_tokens():
  config = AnimaTextConditionerConfig(
      source_dim=8,
      target_dim=8,
      model_dim=8,
      num_layers=1,
      num_attention_heads=2,
      target_vocab_size=32,
      min_sequence_length=7,
  )
  model = FlaxAnimaTextConditioner(config)
  source = jax.random.normal(jax.random.key(1), (1, 5, 8))
  ids = jnp.array([[1, 2, 3, 4, 5, 6, 7]], dtype=jnp.int32)
  source_mask = jnp.array([[1, 1, 1, 0, 0]], dtype=jnp.bool_)
  target_mask = jnp.array([[1, 1, 1, 0, 0, 0, 0]], dtype=jnp.bool_)
  variables = model.init(jax.random.key(2), source, ids, source_mask, target_mask)
  baseline = model.apply(variables, source, ids, source_mask, target_mask)
  changed_source = source.at[:, 3:, :].set(999.0)
  changed_ids = ids.at[:, 3:].set(jnp.array([31, 30, 29, 28], dtype=jnp.int32))
  changed = model.apply(variables, changed_source, changed_ids, source_mask, target_mask)
  assert jnp.max(jnp.abs(baseline[:, :3] - changed[:, :3])) < 1e-5
