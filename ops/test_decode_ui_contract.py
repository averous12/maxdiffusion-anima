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


def test_plain_request_has_no_decode_key():
  box, orig = _capture_request()
  req = _call(box, orig)
  assert req is not None, "request was not written"
  assert "decode" not in req, req.keys()
  assert "pixel_head" not in req
  print("plain request: no decode key")


def test_pixel_request_sets_decode():
  box, orig = _capture_request()
  req = _call(box, orig, decode_mode="pixel", pixel_head="base.safetensors")
  assert req["decode"] == "pixel", req.get("decode")
  assert req["pixel_head"] == "base.safetensors", req.get("pixel_head")
  print("pixel request: decode=pixel, head=base.safetensors")


def test_pixel_without_head_is_rejected():
  box, orig = _capture_request()
  req = _call(box, orig, decode_mode="pixel", pixel_head="(none found)")
  assert req is None, "expected a validation error, got a request"
  print("pixel without head: rejected")


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
