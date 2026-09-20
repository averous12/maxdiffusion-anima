import numpy as np

from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler


def reference_sigmas(num_inference_steps=4, shift=3.0, num_train_timesteps=1000):
  # Diffusers FlowMatchEulerDiscreteScheduler.set_timesteps() with the
  # official Anima scheduler config: no dynamic shifting, no special sigma
  # schedule, and an appended terminal zero.
  sigmas = np.linspace(1.0, 0.001, num_inference_steps, dtype=np.float32)
  sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
  return np.concatenate([sigmas, np.array([0.0], dtype=np.float32)])


def test_anima_flow_match_schedule_matches_diffusers_reference():
  scheduler = FlaxFlowMatchScheduler(
      num_train_timesteps=1000,
      shift=3.0,
      # Diffusers initializes the FlowMatch schedule from 1.0 down to
      # 1/num_train_timesteps, so Anima's sigma_min is 0.001.
      sigma_min=1.0 / 1000.0,
      sigma_max=1.0,
      dtype=np.float32,
  )
  state = scheduler.set_timesteps(scheduler.create_state(), num_inference_steps=4)
  actual = np.asarray(state.sigmas)
  expected = reference_sigmas()
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
