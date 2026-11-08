"""Probe 2: nn.scan with per-layer extra args (in_axes 0 / None / broadcast), plus the
convention the Anima ScanBlock must follow: __call__ returns (carry, output).
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.axes_scan import broadcast as BC


class Body(nn.Module):
  width: int
  record: bool = False

  @nn.compact
  def __call__(self, x, w, lora, refk, bias):
    y = nn.Dense(self.width, use_bias=False, name="proj")(x)
    if lora is not None:
      A, B = lora
      y = y + (x @ A) @ B
    if refk is not None:
      y = y + 0.0 * (refk[0].sum() + refk[1].sum())
    out = y * w + bias.sum()
    if self.record:
      return out, (y, out * 2.0)          # carry, ys -> ys stacked per layer
    return out, out                       # carry, ys


class Net(nn.Module):
  width: int
  layers: int
  scan: bool = True
  record: bool = False

  def _axes(self, lora, refk):
    return (BC,
            None if lora is None else jax.tree_util.tree_map(lambda _: 0, lora),
            None if refk is None else jax.tree_util.tree_map(lambda _: 0, refk),
            BC)

  @nn.compact
  def __call__(self, x, w, lora, refk, bias):
    if self.scan:
      S = nn.scan(Body, variable_axes={"params": 0}, split_rngs={"params": False},
                  length=self.layers, in_axes=self._axes(lora, refk))
      return S(self.width, record=self.record, name="s")(x, w, lora, refk, bias)
    out, ys = x, None
    for i in range(self.layers):
      out, ys = Body(self.width, record=self.record, name=f"b{i}")(out, w[i], lora, refk, bias)
    return out, ys


def copy_params(params, layers):
  return {"params": {f"b{i}": jax.tree_util.tree_map(lambda a: a[i], params["params"]["s"]["b"])
                     for i in range(layers)}}


def main():
  L, W, R = 4, 8, 3
  x = jnp.ones((1, 5, W))
  w = jnp.arange(L, dtype=jnp.float32)[:, None, None] + 1.0
  A = jnp.ones((L, W, R)) * 0.1
  B = jnp.ones((L, R, W)) * 0.1
  k = jnp.arange(L * 1 * 3 * 2, dtype=jnp.float32).reshape(L, 1, 3, 2)
  v = k * 2
  bias = jnp.array([0.5, -0.25])

  for rec in (False, True):
    net = Net(W, L, record=rec)
    params = net.init(jax.random.PRNGKey(0), x, w, None, None, jnp.zeros((1,)))
    print("record=%s param %s" % (rec, params["params"]["s"]["b"]["proj"]["kernel"].shape))

    plain = net.apply(params, x, w, None, None, jnp.zeros((1,)))[0]
    lo = net.apply(params, x, w, (A, B), None, jnp.zeros((1,)))[0]
    rk = net.apply(params, x, w, None, (k, v), bias)[0]
    both = net.apply(params, x, w, (A, B), (k, v), bias)
    print("  lora delta %.6f  refkv delta %.6f" % (float(jnp.abs(lo - plain).max()),
                                                   float(jnp.abs(rk - plain).max())))
    un = Net(W, L, scan=False, record=rec).apply(copy_params(params, L), x, w, (A, B), (k, v), bias)
    print("  scan vs unrolled out %.3e" % float(jnp.abs(both[0] - un[0]).max()))
    if rec:
      print("  ys k %s k[0,0,0,:4]=%s" % (both[1][0].shape, both[1][0][0, 0, 0, :4]))
      print("  ys scan-vs-unrolled %.3e" % float(jnp.abs(both[1][0] - un[1][0]).max()))
  print("PROBE2 OK")


if __name__ == "__main__":
  main()
