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


def test_mask_region_contract(tmp_path=None):
    from PIL import Image
    import tempfile
    root = tempfile.mkdtemp()
    path = os.path.join(root, "mask.png")
    Image.new("L", (16, 16), 255).save(path)
    x = r.normalize_region({"prompt": "masked", "mask": path})
    m = r.mask_file_mask(x, 4, 4, (16, 16))
    assert m.shape == (1, 1, 1, 4, 4)
    assert np.all(m == 1.0)
    try:
        r.normalize_region({"prompt": "masked", "mask": path, "feather": .1})
    except r.RegionalPromptError:
        pass
    else:
        raise AssertionError("mask + feather should be rejected")


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


def test_normalize_arbitrary_stage_schedule():
    schedule = r.normalize_schedule({"stages": [
        {"id": "base", "start": 0.0, "end": .1, "regions": []},
        {"id": "chars", "start": .1, "end": .6, "regions": [
            {"prompt": "left", "box": [0, 0, .5, 1]},
            {"prompt": "right", "box": [.5, 0, .5, 1]},
        ]},
        {"id": "merge", "start": .6, "end": 1.0, "regions": [
            {"prompt": "two characters together", "box": [0, 0, 1, 1]}
        ]},
    ]})
    assert [x["stage_id"] for x in schedule] == ["chars", "chars", "merge"]
    assert [(x["start"], x["end"]) for x in schedule] == [(.1, .6), (.1, .6), (.6, 1.0)]
    assert r.active_indices(schedule, 1, 10) == (0, 1)
    assert r.active_indices(schedule, 6, 10) == (2,)


def test_flat_regions_remain_backward_compatible():
    schedule = r.normalize_schedule([
        {"prompt": "left", "box": [0, 0, .5, 1], "start": .1, "end": .6}
    ])
    assert len(schedule) == 1 and schedule[0]["stage_id"] == "legacy"


def test_overlapping_region_windows_are_independent():
    regions = r.normalize_regions([
        {"prompt": "A", "box": [0, 0, .3, 1], "start": .3, "end": .6},
        {"prompt": "B", "box": [.3, 0, .3, 1], "start": .1, "end": .5},
        {"prompt": "C", "box": [.6, 0, .4, 1], "start": .4, "end": .8},
    ])
    assert r.active_indices(regions, 0, 10) == (1,)
    assert r.active_indices(regions, 3, 10) == (0, 1, 2)
    assert r.active_indices(regions, 6, 10) == (2,)


def test_always_on_region_survives_midrun_addition_and_exit():
    regions = r.normalize_regions([
        {"prompt": "persistent", "box": [0, 0, 1, 1], "start": 0.0, "end": 1.0},
        {"prompt": "temporary", "box": [0, 0, .5, 1], "start": .3, "end": .6},
    ])
    assert r.active_indices(regions, 0, 10) == (0,)
    assert r.active_indices(regions, 3, 10) == (0, 1)
    assert r.active_indices(regions, 7, 10) == (0,)


def test_active_indices_are_stage_keys():
    regions = r.normalize_regions([
        {"prompt": "left", "box": [0, 0, .5, 1], "start": .11, "end": .6},
        {"prompt": "right", "box": [.5, 0, .5, 1], "start": .11, "end": .6},
        {"prompt": "stitched", "box": [0, 0, 1, 1], "start": .6, "end": 1.0},
    ])
    assert r.active_indices(regions, 0, 10) == ()
    assert r.active_indices(regions, 1, 10) == (0, 1)
    assert r.active_indices(regions, 6, 10) == (2,)


def test_max_regions():
    xs = [{"prompt": str(i), "box": [i / 5, 0, .1, 1]} for i in range(5)]
    try:
        r.normalize_regions(xs)
    except r.RegionalPromptError:
        return
    raise AssertionError("accepted >4 regions")


def test_full_coverage_tiling_regions():
    """Halves and thirds tile the canvas, so the base prediction can be skipped."""
    h = w = 16
    assert r.full_coverage([{"box": [0, 0, .5, 1]}, {"box": [.5, 0, .5, 1]}], h, w)
    thirds = [{"box": [0, 0, .34, 1]}, {"box": [.34, 0, .32, 1]},
              {"box": [.66, 0, .34, 1]}]
    assert r.full_coverage(thirds, h, w)
    # Overlapping regions exceed mass 1 everywhere: still a safe skip.
    assert r.full_coverage([{"box": [0, 0, .7, 1]}, {"box": [.3, 0, .7, 1]}], h, w)


def test_full_coverage_rejects_incomplete():
    """Anything short of complete coverage must keep the base prediction."""
    h = w = 16
    assert not r.full_coverage([], h, w)
    assert not r.full_coverage([{"box": [0, 0, .5, 1]}], h, w)
    assert not r.full_coverage([{"box": [0, 0, .5, 1]}, {"box": [.6, 0, .4, 1]}], h, w)
    # A vertical gap is just as disqualifying as a horizontal one.
    assert not r.full_coverage([{"box": [0, 0, 1, .5]}, {"box": [0, .6, 1, .4]}], h, w)
    # Strength < 1 scales the mask, so coverage is not complete even when it tiles.
    assert not r.full_coverage([{"box": [0, 0, .5, 1], "strength": 0.5},
                                {"box": [.5, 0, .5, 1]}], h, w)


def test_common_block_is_prepended_to_every_region():
    """A shared quality/artist preamble must reach every region, exactly once."""
    common = {"prompt": "masterpiece, best quality, @sw33t,",
              "negative_prompt": "worst quality, low quality"}
    regions = r.normalize_schedule([
        {"prompt": "1girl, red coat", "box": [0, 0, .5, 1]},
        {"prompt": "1boy, blue coat", "box": [.5, 0, .5, 1]},
    ], common=common)
    assert regions[0]["prompt"] == "masterpiece, best quality, @sw33t, 1girl, red coat"
    assert regions[1]["prompt"] == "masterpiece, best quality, @sw33t, 1boy, blue coat"
    for region in regions:
        assert region["negative_prompt"] == "worst quality, low quality"


def test_common_from_schedule_object():
    """The dict form carries `common` alongside `stages`."""
    schedule = {"common": {"prompt": "masterpiece, best quality"},
                "stages": [{"id": "s", "start": 0.0, "end": 1.0,
                            "regions": [{"prompt": "1girl", "box": [0, 0, 1, 1]}]}]}
    regions = r.normalize_schedule(schedule)
    assert regions[0]["prompt"] == "masterpiece, best quality, 1girl"
    assert regions[0]["negative_prompt"] == ""


def test_disabled_region_is_dropped():
    """The UI checkmark must cost nothing: no encode, no branch, no region slot."""
    regions = r.normalize_schedule([
        {"prompt": "1girl", "box": [0, 0, .5, 1]},
        {"prompt": "1boy", "box": [.5, 0, .5, 1], "enabled": False},
    ])
    assert len(regions) == 1
    assert regions[0]["prompt"] == "1girl"
    # Disabled regions do not count against the region cap either.
    many = [{"prompt": "x", "box": [0, 0, 1, 1], "enabled": False} for _ in range(40)]
    assert r.normalize_schedule(many) == []


def test_region_may_rely_on_common_prompt():
    """A blank region prompt is legal when the shared block supplies the text."""
    regions = r.normalize_schedule([{"prompt": "", "box": [0, 0, 1, 1]}],
                                   common={"prompt": "masterpiece, 1girl"})
    assert regions[0]["prompt"] == "masterpiece, 1girl"
    try:
        r.normalize_schedule([{"prompt": "", "box": [0, 0, 1, 1]}])
    except r.RegionalPromptError:
        return
    raise AssertionError("empty prompt with no common block should fail")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for test in tests:
        test()
    print(f"REGIONAL {len(tests)}/{len(tests)}")
