"""Unit tests for the regional request contract and latent mask geometry."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from maxdiffusion.models import anima_regional as r


def test_normalize_and_active_schedule():
    x = r.normalize_region({"prompt": "left person", "box": [0, 0, .5, 1], "start": .26, "end": .76})
    assert x["box"] == [0.0, 0.0, .5, 1.0]
    assert not r.active(x, 0, 4)
    assert r.active(x, 1, 4)
    assert r.active(x, 2, 4)
    assert not r.active(x, 3, 4)


def test_box_mask_has_correct_extent_and_feather():
    x = r.normalize_region({"prompt": "left", "box": [0, 0, .5, 1], "feather": .25})
    m = r.box_mask(x, 8, 8)
    assert m.shape == (1, 1, 1, 8, 8)
    row = m[0, 0, 0, 3]
    assert 0.0 < row[0] < row[1] <= 1.0
    assert np.all(m[0, 0, 0, :, 4:] == 0.0)
    assert np.all((m >= 0.0) & (m <= 1.0))


def test_invalid_region_payloads_fail_closed():
    for payload in (
        {"prompt": "", "box": [0, 0, 1, 1]},
        {"prompt": "x", "box": [0, 0, 2, 1]},
        {"prompt": "x", "box": [0, 0, 1, 1], "start": .8, "end": .2},
    ):
        try:
            r.normalize_region(payload)
        except r.RegionalPromptError:
            continue
        raise AssertionError(f"accepted invalid region: {payload}")


def test_max_regions():
    xs = [{"prompt": str(i), "box": [i / 5, 0, .1, 1]} for i in range(5)]
    try:
        r.normalize_regions(xs)
    except r.RegionalPromptError:
        return
    raise AssertionError("accepted >4 regions")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
    print(f"REGIONAL {len(tests)}/{len(tests)}")
