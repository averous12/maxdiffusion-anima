"""Small, backend-neutral contract helpers for scheduled Anima regional prompting.

A region is a normalized box plus a prompt schedule. The server owns the expensive
conditioning and SnapRef preparation; this module only validates the public shape,
selects active regions, and builds latent-grid masks.
"""
from __future__ import annotations

import math
from typing import Any


class RegionalPromptError(ValueError):
  pass


def _finite(value: Any, name: str) -> float:
  if isinstance(value, bool):
    raise RegionalPromptError(f"{name} must be a number")
  try:
    result = float(value)
  except (TypeError, ValueError) as exc:
    raise RegionalPromptError(f"{name} must be a number") from exc
  if not math.isfinite(result):
    raise RegionalPromptError(f"{name} must be finite")
  return result


def normalize_region(spec: Any, index: int = 0) -> dict:
  """Validate one public region and return a canonical copy.

  ``box`` is normalized ``[x, y, width, height]``. ``start``/``end`` are
  fractions of the denoise schedule. ``snapref`` is passed through for the
  server's existing reference preparation path.
  """
  if not isinstance(spec, dict):
    raise RegionalPromptError(f"region {index} must be an object")
  prompt = str(spec.get("prompt", "")).strip()
  if not prompt:
    raise RegionalPromptError(f"region {index} prompt is empty")
  box = spec.get("box")
  if isinstance(box, dict):
    values = [box.get(k) for k in ("x", "y", "w", "h")]
  elif isinstance(box, (list, tuple)) and len(box) == 4:
    values = list(box)
  else:
    raise RegionalPromptError(f"region {index} box must be [x, y, width, height]")
  x, y, w, h = (_finite(v, f"region {index} box[{n}]") for n, v in enumerate(values))
  start = _finite(spec.get("start", 0.0), f"region {index} start")
  end = _finite(spec.get("end", 1.0), f"region {index} end")
  feather = _finite(spec.get("feather", 0.0), f"region {index} feather")
  strength = _finite(spec.get("strength", 1.0), f"region {index} strength")
  if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 0.0 and h > 0.0):
    raise RegionalPromptError(f"region {index} box must start in [0,1] and have positive size")
  if x + w > 1.0 or y + h > 1.0:
    raise RegionalPromptError(f"region {index} box extends outside the canvas")
  if not (0.0 <= start < end <= 1.0):
    raise RegionalPromptError(f"region {index} requires 0 <= start < end <= 1")
  if not (0.0 <= feather <= 0.5):
    raise RegionalPromptError(f"region {index} feather must be between 0 and 0.5")
  if strength < 0.0:
    raise RegionalPromptError(f"region {index} strength cannot be negative")
  return {
    "prompt": prompt,
    "negative_prompt": str(spec.get("negative_prompt", "")).strip(),
    "box": [x, y, w, h],
    "start": start,
    "end": end,
    "feather": feather,
    "strength": strength,
    "snapref": spec.get("snapref") if isinstance(spec.get("snapref"), dict) else None,
  }


def normalize_regions(regions: Any, maximum: int = 4) -> list[dict]:
  if regions in (None, "", []):
    return []
  if not isinstance(regions, list):
    raise RegionalPromptError("regional must be a list")
  if len(regions) > maximum:
    raise RegionalPromptError(f"at most {maximum} regional prompts are supported")
  return [normalize_region(item, i) for i, item in enumerate(regions)]


def active(region: dict, step: int, steps: int) -> bool:
  fraction = (int(step) + 1) / max(1, int(steps))
  return region["start"] <= fraction < region["end"] or (
      step == int(steps) - 1 and fraction <= region["end"]
  )


def box_mask(region: dict, height: int, width: int):
  """Return a float32 mask at the latent grid resolution.

  Feathering is a linear edge ramp in normalized image coordinates. The mask
  shape is ``(1, 1, height, width)`` and broadcasts over the batch/channel.
  """
  import numpy as np
  x, y, w, h = region["box"]
  yy = (np.arange(int(height), dtype=np.float32) + 0.5) / max(1, int(height))
  xx = (np.arange(int(width), dtype=np.float32) + 0.5) / max(1, int(width))
  dx = np.minimum(xx - x, x + w - xx)
  dy = np.minimum(yy - y, y + h - yy)
  inside = np.minimum(dy[:, None], dx[None, :])
  feather = float(region.get("feather", 0.0))
  if feather <= 0.0:
    mask = (inside >= 0.0).astype(np.float32)
  else:
    mask = np.clip(inside / feather, 0.0, 1.0).astype(np.float32)
  return mask[None, None, None, :, :] * float(region.get("strength", 1.0))
