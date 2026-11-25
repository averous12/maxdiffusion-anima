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

  ``box`` is normalized ``[x, y, width, height]``. Alternatively, ``mask``
  is a request-local PNG path, normalized by the API before the server sees it.
  ``start``/``end`` are schedule fractions; ``snapref`` is passed through.
  """
  if not isinstance(spec, dict):
    raise RegionalPromptError(f"region {index} must be an object")
  prompt = str(spec.get("prompt", "")).strip()
  if not prompt:
    raise RegionalPromptError(f"region {index} prompt is empty")
  mask = spec.get("mask")
  box = spec.get("box")
  if mask:
    canonical_box = None
  else:
    if isinstance(box, dict):
      values = [box.get(k) for k in ("x", "y", "w", "h")]
    elif isinstance(box, (list, tuple)) and len(box) == 4:
      values = list(box)
    else:
      raise RegionalPromptError(f"region {index} needs either mask or box [x, y, width, height]")
    x, y, w, h = (_finite(v, f"region {index} box[{n}]") for n, v in enumerate(values))
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and w > 0.0 and h > 0.0):
      raise RegionalPromptError(f"region {index} box must start in [0,1] and have positive size")
    if x + w > 1.0 or y + h > 1.0:
      raise RegionalPromptError(f"region {index} box extends outside the canvas")
    canonical_box = [x, y, w, h]
  start = _finite(spec.get("start", 0.0), f"region {index} start")
  end = _finite(spec.get("end", 1.0), f"region {index} end")
  feather = _finite(spec.get("feather", 0.0), f"region {index} feather")
  strength = _finite(spec.get("strength", 1.0), f"region {index} strength")
  if not (0.0 <= start < end <= 1.0):
    raise RegionalPromptError(f"region {index} requires 0 <= start < end <= 1")
  if mask:
    if feather != 0.0:
      raise RegionalPromptError(f"region {index} feather belongs in the uploaded mask")
  elif not (0.0 <= feather <= 0.5):
    raise RegionalPromptError(f"region {index} feather must be between 0 and 0.5")
  if strength < 0.0:
    raise RegionalPromptError(f"region {index} strength cannot be negative")
  return {
    "prompt": prompt,
    "negative_prompt": str(spec.get("negative_prompt", "")).strip(),
    "box": canonical_box,
    "mask": str(mask) if mask else None,
    "start": start,
    "end": end,
    "feather": feather,
    "strength": strength,
    "snapref": spec.get("snapref") if isinstance(spec.get("snapref"), dict) else None,
  }


def normalize_schedule(schedule: Any, maximum: int = 16) -> list[dict]:
  """Flatten a stage schedule while preserving arbitrary interval overlap.

  Accepted forms:
    * legacy list of region objects;
    * {"stages": [{"id", "start", "end", "regions": [...]}]}.

  Empty stages are valid composition-only phases and produce no regional branch.
  Region intervals override stage bounds when explicitly supplied.
  """
  if schedule in (None, "", []):
    return []
  if isinstance(schedule, list):
    if len(schedule) > maximum:
      raise RegionalPromptError(f"at most {maximum} regional prompts are supported")
    out = []
    for i, item in enumerate(schedule):
      region = normalize_region(item, i)
      region["stage_id"] = "legacy"
      out.append(region)
    return out
  if not isinstance(schedule, dict) or not isinstance(schedule.get("stages"), list):
    raise RegionalPromptError("regional must be a list or an object with stages")
  out = []
  for stage_index, stage in enumerate(schedule["stages"]):
    if not isinstance(stage, dict):
      raise RegionalPromptError(f"stage {stage_index} must be an object")
    stage_id = str(stage.get("id", f"stage_{stage_index}")).strip() or f"stage_{stage_index}"
    start = _finite(stage.get("start", 0.0), f"stage {stage_index} start")
    end = _finite(stage.get("end", 1.0), f"stage {stage_index} end")
    if not 0.0 <= start < end <= 1.0:
      raise RegionalPromptError(f"stage {stage_id!r} requires 0 <= start < end <= 1")
    regions = stage.get("regions") or []
    if not isinstance(regions, list):
      raise RegionalPromptError(f"stage {stage_id!r} regions must be a list")
    for region_index, raw in enumerate(regions):
      if not isinstance(raw, dict):
        raise RegionalPromptError(f"stage {stage_id!r} region {region_index} must be an object")
      item = dict(raw)
      item.setdefault("start", start)
      item.setdefault("end", end)
      region = normalize_region(item, len(out))
      region["stage_id"] = stage_id
      out.append(region)
      if len(out) > maximum:
        raise RegionalPromptError(f"at most {maximum} regional prompts are supported")
  return out


def normalize_regions(regions: Any, maximum: int = 4) -> list[dict]:
  return normalize_schedule(regions, maximum=maximum)


def active_indices(regions: list[dict], step: int, steps: int) -> tuple[int, ...]:
  """Return the stable active-region key used for lazy stage activation."""
  return tuple(i for i, region in enumerate(regions) if active(region, step, steps))


def active(region: dict, step: int, steps: int) -> bool:
  fraction = (int(step) + 1) / max(1, int(steps))
  return region["start"] <= fraction < region["end"] or (
      step == int(steps) - 1 and fraction <= region["end"]
  )


def box_mask(region: dict, height: int, width: int):
  """Return normalized box-mask at latent grid resolution (5-D broadcast shape)."""
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


def mask_file_mask(region: dict, height: int, width: int, canvas_hw):
  """Load an explicit grayscale mask and nearest-resize it to latent resolution."""
  from PIL import Image
  import numpy as np
  path = region.get("mask")
  if not path:
    raise RegionalPromptError("region has no mask path")
  try:
    with Image.open(path) as image:
      image = image.convert("L")
      if image.size != (int(canvas_hw[1]), int(canvas_hw[0])):
        raise RegionalPromptError(
            f"mask dimensions {image.size} do not match canvas {(int(canvas_hw[1]), int(canvas_hw[0]))}")
      image = image.resize((int(width), int(height)), Image.Resampling.NEAREST)
      array = np.asarray(image, dtype=np.float32) / 255.0
  except RegionalPromptError:
    raise
  except Exception as exc:
    raise RegionalPromptError(f"cannot read region mask {path!r}: {exc}") from exc
  return array[None, None, None, :, :] * float(region.get("strength", 1.0))
