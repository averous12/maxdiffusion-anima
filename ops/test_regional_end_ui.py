"""UI contract checks for the regional routing window (regional_end).

Measured on TPU at 1024x1024/20 steps: window 0.35 gave two separate characters in
18.6 s, full window gave a split figure in 50.4 s. The slider therefore defaults to
0.35 and its value must reach the request, clamped to [0, 1].

Run: python ops/test_regional_end_ui.py
"""

import json
import os
import sys

CONTENT = os.environ.setdefault("ANIMA_CONTENT_ROOT", os.path.join(os.environ.get("TEMP", "/tmp"),
                                                                  "anima_ui_contract"))
os.makedirs(CONTENT, exist_ok=True)
os.makedirs(os.path.join(CONTENT, "anima_pixel_heads"), exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import anima_gradio_ui as ui  # noqa: E402

REGION = [{"prompt": "a girl in red", "box": [0.03, 0.06, 0.44, 0.88],
           "start": 0.10, "end": 1.0, "enabled": True}]


class _Captured(Exception):
  pass


def _capture_request():
  box = {}
  orig = ui._atomic_json

  def fake(path, obj):
    box["req"] = obj
    raise _Captured()
  ui._atomic_json = fake
  return box, orig


def _call(box, orig, **kw):
  args = dict(prompt="a cat", neg="", aspect_w=1, aspect_h=1, mp_label="1.0 MP",
              width=1024, height=1024, steps=8, seed=7, guidance=4.0, preview_every=0,
              batch_count=1)
  args.update(kw)
  try:
    for _ in ui.start_gen(**args):
      pass
  except _Captured:
    pass
  finally:
    ui._atomic_json = orig
  return box.get("req")


def _components(demo):
  cfg = demo.get_config_file()
  return {c.get("props", {}).get("label"): c.get("props", {}) for c in cfg.get("components", [])}


def test_slider_exists_with_0_35_default():
  props = _components(ui.build())
  assert "Regional routing window" in props, sorted(k for k in props if k)
  slider = props["Regional routing window"]
  assert slider.get("value") == 0.35, slider.get("value")
  assert (slider.get("minimum"), slider.get("maximum")) == (0.1, 1.0), slider
  print("slider present with default 0.35 and range 0.1-1.0")


def test_window_reaches_request():
  box, orig = _capture_request()
  req = _call(box, orig, regional_json=json.dumps(REGION), regional_end=0.35)
  assert req is not None, "no request captured"
  assert req["regional_end"] == 0.35, req.get("regional_end")
  assert len(req["regional"]) == 1
  print("regional_end 0.35 reaches the request")


def test_window_is_clamped():
  for sent, want in ((2.0, 1.0), (-3.0, 0.0), ("nonsense", 1.0), (None, 1.0)):
    box, orig = _capture_request()
    req = _call(box, orig, regional_json=json.dumps(REGION), regional_end=sent)
    assert req["regional_end"] == want, (sent, req.get("regional_end"), want)
  print("out-of-range and garbage windows clamp instead of crashing")


def test_no_regions_means_no_window():
  box, orig = _capture_request()
  req = _call(box, orig, regional_end=0.35)
  assert req is not None
  assert "regional_end" not in req, req.get("regional_end")
  print("no regions -> no regional_end key (plain inference untouched)")


if __name__ == "__main__":
  tests = [test_slider_exists_with_0_35_default, test_window_reaches_request,
           test_window_is_clamped, test_no_regions_means_no_window]
  failed = 0
  for t in tests:
    try:
      t()
      print(f"PASS {t.__name__}")
    except AssertionError as e:
      failed += 1
      print(f"FAIL {t.__name__}: {e}")
  print(f"REGIONAL_END_UI {len(tests) - failed}/{len(tests)}")
  sys.exit(1 if failed else 0)
