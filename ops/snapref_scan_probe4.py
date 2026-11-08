"""Probe 4: the exact scan structure SnapRef needs.

- lora: nested dict of per-layer (A, B) leaf pairs + a broadcast scalar
- refkv: per-layer (k, v) tuples
- record: the body returns (carry, (k, v)) so the scan stacks per-layer K/V
- all compared against the unrolled path on the same weights
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.axes_scan import broadcast as BC

L, W, R = 4, 8, 3


class Body(nn.Module):
  width: int
  record: bool = False

  @nn.compact
  def __call__(self, x, lora, refk, bias):
    q = nn.Dense(self.width, use_bias=False, name="to_q")(x)
    if lora is not None:
      A, B = lora["to_q"]
      q = q + (x @ A) @ B
    k = q + bias * 0.0
    if refk is not None:
      k = jnp.concatenate([refk[0], k], axis=1)
    v = k * 2.0
    out = q + 0.0 * k.sum()
    if self.record:
      return out, (k, v)
    return out, out


class Net(nn.Module):
  layers: int
  scan: bool = True
  record: bool = False

  def _axes(self, lora, refk):
    lp = None if lora is None else {"to_q": (0, 0)}
    rp = None if refk is None else (0, 0)
    return (lp, rp, BC)

  @nn.compact
  def __call__(self, x, lora, refk, bias):
    if self.scan:
      S = nn.scan(Body, variable_axes={"params": 0}, split_rngs={"params": False},
                  length=self.layers, in_axes=self._axes(lora, refk))
      return S(W, record=self.record, name="s")(x, lora, refk, bias)
    out, ys = x, None
    for i in range(self.layers):
      li = None if lora is None else {"to_q": (lora["to_q"][0][i], lora["to_q"][1][i])}
      ri = None if refk is None else (refk[0][i], refk[1][i])
      out, ys = Body(W, record=self.record, name=f"b{i}")(out, li, ri, bias)
    return out, ys


def copy_params(params, layers):
  return {"params": {f"b{i}": jax.tree_util.tree_map(lambda a: a[i], params["params"]["s"])
                     for i in range(layers)}}


def main():
  x = jnp.ones((1, 5, W))
  A = jnp.ones((L, W, R)) * 0.1
  B = jnp.ones((L, R, W)) * 0.1
  kref = jnp.arange(L * 1 * 3 * W, dtype=jnp.float32).reshape(L, 1, 3, W) * 0.01
  vref = kref * 2
  bias = jnp.float32(0.5)
  lora = {"to_q": (A, B)}
  refk = (kref, vref)

  for rec in (False, True):
    net = Net(L, record=rec)
    params = net.init(jax.random.PRNGKey(0), x, None, None, bias)
    plain = net.apply(params, x, None, None, bias)
    lo = net.apply(params, x, lora, None, bias)
    rk = net.apply(params, x, None, refk, bias)
    both = net.apply(params, x, lora, refk, bias)
    print("record=%s  lora delta %.6f  refk delta %.6f" % (
        rec, float(jnp.abs(lo[0] - plain[0]).max()), float(jnp.abs(rk[0] - plain[0]).max())))
    un = Net(L, scan=False, record=rec).apply(copy_params(params, L), x, lora, refk, bias)
    print("   scan vs unrolled out %.3e" % float(jnp.abs(both[0] - un[0]).max()))
    if rec:
      print("   recorded k %s vs unrolled %.3e" % (both[1][0].shape,
                                                   float(jnp.abs(both[1][0] - un[1][0]).max())))
  print("PROBE4 OK")


if __name__ == "__main__":
  main()
