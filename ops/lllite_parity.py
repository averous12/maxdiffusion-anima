"""Numerical parity: JAX port vs a PyTorch transcription of ComfyUI's anima/lllite.py.

The PyTorch side is built so its attribute names match the checkpoint keys exactly, which
lets `load_state_dict(..., strict=True)` consume the real file. That means the reference is
running the real weights through the real module structure, and any disagreement with the
JAX port is a bug in the port rather than in the test.

Run:  python lllite_parity.py [checkpoint.safetensors] [control.png]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "maxdiffusion/src")
from maxdiffusion.models import anima_lllite_flax as LL  # noqa: E402


# ---- PyTorch reference, transcribed from comfy/ldm/anima/lllite.py -------------
def _gn(c):
  g = 8
  while g > 1 and c % g != 0:
    g //= 2
  return nn.GroupNorm(g, c)


class RefResBlock(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.norm1, self.conv1 = _gn(c), nn.Conv2d(c, c, 3, padding=1)
    self.norm2, self.conv2 = _gn(c), nn.Conv2d(c, c, 3, padding=1)

  def forward(self, x):
    h = self.conv1(F.silu(self.norm1(x)))
    return x + self.conv2(F.silu(self.norm2(h)))


class RefASPP(nn.Module):
  def __init__(self, c, dilations):
    super().__init__()
    self.branches = nn.ModuleList()
    for d in dilations:
      conv = nn.Conv2d(c, c, 1) if d == 1 else nn.Conv2d(c, c, 3, padding=d, dilation=d)
      self.branches.append(nn.Sequential(conv, _gn(c), nn.SiLU()))
    self.global_pool = nn.AdaptiveAvgPool2d(1)
    self.global_conv = nn.Sequential(nn.Conv2d(c, c, 1), _gn(c), nn.SiLU())
    self.proj = nn.Sequential(nn.Conv2d(c * (len(dilations) + 1), c, 1), _gn(c), nn.SiLU())

  def forward(self, x):
    h, w = x.shape[-2:]
    outs = [b(x) for b in self.branches]
    outs.append(F.interpolate(self.global_conv(self.global_pool(x)), size=(h, w),
                              mode="bilinear", align_corners=False))
    return self.proj(torch.cat(outs, dim=1))


class RefConditioning(nn.Module):
  def __init__(self, cin, cdim, cemb, nres, dilations):
    super().__init__()
    half = cdim // 2
    self.conv1, self.norm1 = nn.Conv2d(cin, half, 4, stride=4), _gn(half)
    self.conv2, self.norm2 = nn.Conv2d(half, half, 3, padding=1), _gn(half)
    self.conv3, self.norm3 = nn.Conv2d(half, cdim, 4, stride=4), _gn(cdim)
    self.resblocks = nn.ModuleList([RefResBlock(cdim) for _ in range(nres)])
    self.aspp = RefASPP(cdim, dilations) if dilations else None
    self.proj = nn.Conv2d(cdim, cemb, 1)
    self.out_norm = nn.LayerNorm(cemb)

  def forward(self, x):
    x = F.silu(self.norm1(self.conv1(x)))
    x = F.silu(self.norm2(self.conv2(x)))
    x = F.silu(self.norm3(self.conv3(x)))
    for b in self.resblocks:
      x = b(x)
    if self.aspp is not None:
      x = self.aspp(x)
    x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
    return self.out_norm(x)


class RefModule(nn.Module):
  def __init__(self, ind, cemb, mlp):
    super().__init__()
    self.down = nn.Linear(ind, mlp)
    self.mid = nn.Linear(mlp + cemb, mlp)
    self.cond_to_film = nn.Linear(cemb, 2 * mlp)
    self.up = nn.Linear(mlp, ind)
    self.depth_embed = nn.Parameter(torch.empty(cemb), requires_grad=False)

  def forward(self, x, cond_emb, strength):
    cond_local = cond_emb + self.depth_embed
    hidden = F.silu(self.down(x))
    gamma, beta = self.cond_to_film(cond_local).chunk(2, dim=-1)
    hidden = self.mid(torch.cat((cond_local, hidden), dim=-1))
    hidden = F.silu(hidden * (1 + gamma) + beta)
    return x + self.up(hidden) * strength


class RefLLLite(nn.Module):
  def __init__(self, cfg, module_names):
    super().__init__()
    self.lllite_conditioning1 = RefConditioning(
        cfg.cond_in_channels, cfg.cond_dim, cfg.cond_emb_dim, cfg.cond_resblocks,
        cfg.aspp_dilations)
    for name in module_names:
      setattr(self, name, RefModule(cfg.model_dim, cfg.cond_emb_dim, cfg.mlp_dim))


def main():
  ckpt = sys.argv[1] if len(sys.argv) > 1 else (
      "hf_cache/models--toyxyz--Anima_ControlNet-LLLite/snapshots/"
      "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d/anima_3dpose_6200.safetensors")
  ctrl = sys.argv[2] if len(sys.argv) > 2 else (
      "hf_cache/models--toyxyz--Anima_ControlNet-LLLite/snapshots/"
      "37f3a4b6574f1dce3dfe3b5440285a39b11b3d8d/example_workflow/3dpose_example.png")

  import jax.numpy as jnp
  from PIL import Image
  from safetensors.torch import load_file

  print(f"checkpoint: {ckpt.split('/')[-1]}")
  jl = LL.load_lllite(ckpt)
  cfg = jl.config

  sd = load_file(ckpt)
  ref = RefLLLite(cfg, [n for n in {k.split('.', 1)[0] for k in sd
                                    if k.startswith('lllite_dit_blocks_')}])
  missing, unexpected = ref.load_state_dict(sd, strict=False)
  print(f"reference load: {len(sd)} tensors, missing={len(missing)} unexpected={len(unexpected)}")
  if missing:
    print("  MISSING:", missing[:5])
  if unexpected:
    print("  UNEXPECTED:", unexpected[:5])
  ref.eval()

  # ---- conditioning net
  img = Image.open(ctrl).convert("RGB")
  hw = (256, 256)
  x_np = LL.preprocess_control(img, hw)
  x_nchw = torch.from_numpy(x_np.transpose(0, 3, 1, 2)).contiguous()
  with torch.no_grad():
    t0 = time.time()
    ref_emb = ref.lllite_conditioning1(x_nchw).numpy()
    t_torch = time.time() - t0
  jax_emb = np.asarray(LL.encode_conditioning(jl, jnp.asarray(x_np)))

  print(f"\nconditioning  shape jax {jax_emb.shape} / torch {ref_emb.shape}")
  d = np.abs(jax_emb - ref_emb)
  rel = d.max() / max(np.abs(ref_emb).max(), 1e-9)
  print(f"  max abs diff {d.max():.3e}   mean {d.mean():.3e}   rel max {rel:.3e}")
  print(f"  torch {t_torch*1000:.0f} ms")

  # ---- per-block module
  gens = torch.Generator().manual_seed(0)
  xs = torch.randn(1, 256, cfg.model_dim, generator=gens)
  xj = jnp.asarray(xs.numpy())
  emb_t = torch.from_numpy(ref_emb)
  emb_j = jnp.asarray(ref_emb)

  ok = True
  for block in (0, 13, cfg.block_count - 1):
    for target in ("self_attn_q", "self_attn_k", "self_attn_v"):
      suffix = {v: k for k, v in LL._SUFFIX_TO_TARGET.items()}[target]
      mod = getattr(ref, f"lllite_dit_blocks_{block}_{suffix}")
      with torch.no_grad():
        yt = mod(xs, emb_t, 0.8).numpy()
      yj = np.asarray(LL.apply_module(jl, xj, emb_j, block, target, 0.8))
      dd = np.abs(yj - yt)
      scale = max(np.abs(yt).max(), 1e-9)
      good = dd.max() / scale < 1e-4
      ok &= good
      print(f"  block {block:2d} {target:12s} max {dd.max():.3e}  rel {dd.max()/scale:.3e}"
            f"  {'PASS' if good else 'FAIL'}")

  # ---- strength scaling / gating
  with torch.no_grad():
    y0 = mod(xs, emb_t, 0.0).numpy()
  print(f"\nstrength 0 returns input exactly: {np.abs(y0 - xs.numpy()).max():.3e}")
  print(f"sigma_gate: in-window {LL.sigma_gate(0.5, 1.0, 0.0, 1.0)}"
        f"  out-window {LL.sigma_gate(1.5, 1.0, 0.0, 1.0)}"
        f"  zero-strength {LL.sigma_gate(0.5, 1.0, 0.0, 0.0)}")

  print("\nRESULT:", "PARITY OK" if ok else "PARITY FAILED")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
