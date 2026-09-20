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
