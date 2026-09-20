import jax.numpy as jnp

from maxdiffusion.models.anima_cosmos_flax import cosmos_patchify, cosmos_unpatchify


def test_cosmos_image_patch_round_trip_preserves_temporal_axis():
  x = jnp.arange(1 * 16 * 1 * 4 * 6, dtype=jnp.float32).reshape(1, 16, 1, 4, 6)
  tokens = cosmos_patchify(x, patch_size=(1, 2, 2))
  assert tokens.shape == (1, 6, 64)
  restored = cosmos_unpatchify(tokens, output_channels=16, spatial_shape=(1, 4, 6), patch_size=(1, 2, 2))
  assert restored.shape == x.shape
  assert jnp.array_equal(restored, x)
