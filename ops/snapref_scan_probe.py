"""Probe: can nn.scan take per-layer extra *arguments* (in_axes 0 / None / broadcast)?

Decides whether the SnapRef LoRA + cached reference K/V can ride through the
scanned Anima block as scan inputs (cheap) or must live in the params tree.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.axes_scan import broadcast as BC


class Body(nn.Module):
  width: int

  @nn.compact
  def __call__(self, x, w, lora, refk, bias):
    # lora: (A, B) per layer (in, r) / (r, out); refk: (k, v) per layer
    y = nn.Dense(self.width, use_bias=False, name="proj")(x)
    if lora is not None:
      A, B = lora
      y = y + (x @ A) @ B
    out = y * w
    if refk is not None:
      out = out + refk[0].sum() * 0.0 + refk[1].sum() * 0.0
    return out + bias.sum() * 0.0, (y[0, 0], out[0, 0])


class Net(nn.Module):
  width: int
  layers: int
  scan: bool = True

  @nn.compact
  def __call__(self, x, w, lora, refk, bias):
    if self.scan:
      S = nn.scan(Body, variable_axes={"params": 0}, split_rngs={"params": False}, length=self.layers,
                  in_axes=(BC,
                           None if lora is None else jax.tree_util.tree_map(lambda _: 0, lora),
                           None if refk is None else jax.tree_util.tree_map(lambda _: 0, refk),
                           BC))
      out, aux = S(self.width, name="s")(x, w, lora, refk, bias)
    else:
      for i in range(self.layers):
        out, aux = Body(self.width, name=f"b{i}")(x, w, lora, refk, bias)
    return out, aux


def main():
  L, W, R = 4, 8, 3
  net = Net(W, L)
  x = jnp.ones((1, 5, W))
  w = jnp.arange(L, dtype=jnp.float32)[:, None, None]
  params = net.init(jax.random.PRNGKey(0), x, w, None, None, jnp.zeros((1,)))
  print("params ok:", jax.tree_util.tree_map(lambda a: a.shape, params["params"]))

  # --- no extras: must be unchanged from today's call convention
  out0, aux0 = net.apply(params, x, w, None, None, jnp.zeros((1,)))
  print("plain out", out0.shape, "aux", aux0)

  # --- LoRA only
  A = jnp.ones((L, W, R)) * 0.1
  B = jnp.ones((L, R, W)) * 0.1
  out1, _ = net.apply(params, x, w, (A, B), None, jnp.zeros((1,)))
  print("lora applied, delta:", float(jnp.abs(out1 - out0).max()))

  # --- ref K/V only
  k = jnp.ones((L, 1, 3, 2))
  v = jnp.ones((L, 1, 3, 2)) * 2
  out2, aux2 = net.apply(params, x, w, None, (k, v), jnp.array([0.5, -0.25]))
  print("refkv ok, aux:", aux2)

  # --- both, and compare against the unrolled path on the same params
  out3, aux3 = net.apply(params, x, w, (A, B), (k, v), jnp.array([0.5, -0.25]))
  up = net.apply(params, x, w, (A, B), (k, v), jnp.array([0.5, -0.25]), method=None) if False else None
  unrolled = Net(W, L, scan=False)
  # copy the scanned params into the unrolled layout for a like-for-like check
  uparams = {"params": {}}
  for i in range(L):
    uparams["params"][f"b{i}"] = jax.tree_util.tree_map(lambda a: a[i], params["params"]["s"]["b"])
  out4, aux4 = unrolled.apply(uparams, x, w, (A, B), (k, v), jnp.array([0.5, -0.25]))
  print("scan vs unrolled max diff:", float(jnp.abs(out3 - out4).max()))
  print("OK")


if __name__ == "__main__":
  main()
