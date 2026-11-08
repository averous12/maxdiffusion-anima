"""Probe 3: which extra-argument shapes does nn.scan actually accept?

Variant A: (x, w)                      -> today's project pattern (must work)
Variant B: (x, w, extra_bc)            -> a second broadcast arg
Variant C: (x, w, extra_scan0)         -> a per-layer arg, in_axes 0
Variant D: (x, w, extra_scan0, extra_bc)
Variant E: (x, w, none_arg)            -> a None-valued arg, in_axes None
"""
import traceback

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.axes_scan import broadcast as BC

L, W = 4, 8


class Body(nn.Module):
  width: int

  @nn.compact
  def __call__(self, x, *args):
    y = nn.Dense(self.width, use_bias=False, name="proj")(x)
    acc = y
    for a in args:
      if a is None:
        continue
      if isinstance(a, tuple):
        for t in a:
          acc = acc + 0.0 * t.sum()
      else:
        acc = acc + 0.0 * a.sum() if a.ndim > 2 else acc + 0.0 * a.sum()
    return y + acc, y + acc


def run(name, in_axes, args, y_shape):
  class Net(nn.Module):
    @nn.compact
    def __call__(self, x, *a):
      S = nn.scan(Body, variable_axes={"params": 0}, split_rngs={"params": False},
                  length=L, in_axes=in_axes)
      return S(W, name="s")(x, *a)

  x = jnp.ones(y_shape)
  net = Net()
  try:
    params = net.init(jax.random.PRNGKey(0), x, *args)
    out, ys = net.apply(params, x, *args)
    print("%-28s OK  out %s ys %s" % (name, out.shape, ys.shape))
  except Exception as e:
    print("%-28s FAIL %s: %s" % (name, type(e).__name__, str(e).splitlines()[0][:110]))


x1 = jnp.ones((1, 5, W))
run("A (x,w) BC", (BC,), (jnp.ones((1,)),), (1, 5, W))
run("B (x,w,extra) BC", (BC, BC), (jnp.ones((1,)), jnp.ones((3, 2))), (1, 5, W))
run("C (x,w,per-layer) 0", (BC, 0), (jnp.ones((1,)), jnp.ones((L, 3, 2))), (1, 5, W))
run("D 0 + BC", (BC, 0, BC), (jnp.ones((1,)), jnp.ones((L, 3, 2)), jnp.ones((3, 2))), (1, 5, W))
run("E None-valued", (BC, None), (jnp.ones((1,)), None), (1, 5, W))
run("F none then 0", (BC, None, 0), (jnp.ones((1,)), None, jnp.ones((L, 3, 2))), (1, 5, W))
