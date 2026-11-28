"""UI contract checks for the decode path (VAE vs pixel head).

Builds the real Gradio Blocks, then calls start_gen with request writing intercepted so the
payload can be asserted without a server. Run: python ops/test_decode_ui_contract.py
"""

import os
import sys
import uuid

CONTENT = os.environ.setdefault("ANIMA_CONTENT_ROOT", os.path.join(os.environ.get("TEMP", "/tmp"),
                                                                  "anima_ui_contract"))
os.makedirs(CONTENT, exist_ok=True)
os.makedirs(os.path.join(CONTENT, "anima_pixel_heads"), exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import anima_gradio_ui as ui  # noqa: E402


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
  # start_gen is a generator (Gradio streams its updates), so the body only runs on
  # iteration.
  try:
    for _ in ui.start_gen(**args):
      pass
  except _Captured:
    pass
  finally:
    ui._atomic_json = orig
  return box.get("req")


def test_blocks_build():
  demo = ui.build()
  assert demo is not None
  print(f"blocks ok; pixel head choices: {ui.pixel_head_names()}")


def test_regional_json_is_sent_and_validated():
  box, orig = _capture_request()
  regional = [{"prompt": "blue-haired boy", "box": [0.5, 0, 0.5, 1],
               "start": 0.25, "end": 0.75, "feather": 0.05,
               "snapref": {"enabled": True, "ref_adapter": "snapref_v2s_pre1.safetensors",
                           "mode": "style", "images": ["/tmp/ref.png"]}}]
  _call(box, orig, regional_json=__import__("json").dumps(regional))
  assert box["req"]["regional"] == regional, box["req"].get("regional")
  box, orig = _capture_request()
  _call(box, orig, regional_json="not json")
  assert "req" not in box
  print("regional UI: structured list sent; invalid JSON rejected")


def test_default_request_has_no_decode_key_without_head():
  """Default is pixel, but with no head selected the UI sends nothing and the server falls
  back. A fresh install must still be able to generate."""
  box, orig = _capture_request()
  req = _call(box, orig)
  assert req is not None, "request was not written"
  assert "decode" not in req, req.keys()
  assert "pixel_head" not in req
  print("default (pixel, no head installed): no decode key -> server fallback")


def test_explicit_vae_request_has_no_decode_key():
  box, orig = _capture_request()
  req = _call(box, orig, decode_mode="vae", pixel_head="base.safetensors")
  assert "decode" not in req, req.keys()
  print("vae request: no decode key")


def test_pixel_request_sets_decode():
  box, orig = _capture_request()
  req = _call(box, orig, decode_mode="pixel", pixel_head="base.safetensors")
  assert req["decode"] == "pixel", req.get("decode")
  assert req["pixel_head"] == "base.safetensors", req.get("pixel_head")
  print("pixel request: decode=pixel, head=base.safetensors")


def test_generate_event_wires_reference_budget_slider():
  demo = ui.build()
  config = demo.config
  budget_ids = {c["id"] for c in config["components"]
                if c.get("props", {}).get("label") == "Reference budget"}
  # The button is wired to start_gen_runs, the wrapper that repeats start_gen "Batch count"
  # times, so match either rather than pinning the wrapper away.
  generate = next(d for d in config["dependencies"]
                  if str(d.get("api_name") or "").startswith("start_gen"))
  assert len(budget_ids) == 1, budget_ids
  budget_id = next(iter(budget_ids))
  assert budget_id in generate["inputs"], (budget_id, generate["inputs"])
  print(f"Generate event wires reference budget component {budget_id}")


def test_reference_budget_is_per_request():
  """The combined separate-reference budget must travel in the request, not the env."""
  png = os.path.join(CONTENT, "ref.png")
  if not os.path.isfile(png):
    from PIL import Image
    Image.new("RGB", (32, 32), (200, 120, 90)).save(png)
  base = dict(enabled=True, adapter="a.safetensors", images=[png])
  loose = ui.build_snapref_config(**base, mp_budget=None)
  assert isinstance(loose, dict) and "ref_mp_budget" not in loose, loose
  tight = ui.build_snapref_config(**base, mp_budget=3.0)
  assert isinstance(tight, dict) and tight["ref_mp_budget"] == 3.0, tight
  neg = ui.build_snapref_config(**base, mp_budget=-1)
  assert isinstance(neg, str) and "negative" in neg, neg
  bad = ui.build_snapref_config(**base, mp_budget="lots")
  assert isinstance(bad, str), bad
  print("reference budget: unset omits, 3.0 passes, -1 and 'lots' rejected")


def main():
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  bad = 0
  for t in tests:
    try:
      t()
      print(f"PASS {t.__name__}")
    except Exception as e:
      bad += 1
      print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
  print(f"DECODE_UI_TESTS {len(tests) - bad}/{len(tests)}")
  return 1 if bad else 0


if __name__ == "__main__":
  sys.exit(main())
