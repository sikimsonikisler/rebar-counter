"""Phase 1 checks. Real photos do not exist yet, so accuracy is pinned on synthetic
images whose count is known exactly (see make_synthetic.py)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

import eval as evaluation
from count import count_image
from make_synthetic import VARIANTS, render, write_image
from pipeline import detect, draw, preprocess, segment

# The pipeline is not exact by construction; this is the band phase 1 must hold.
MAX_ERROR_PCT = 5.0


def _write(tmp_path: Path, name: str, **kwargs) -> tuple[Path, int]:
    scene = render(seed=7, **kwargs)
    path = tmp_path / f"20260101_1200_{scene.count}_12_{name}.png"
    write_image(path, scene.image)
    return path, scene.count


@pytest.mark.parametrize("variant", ["seyrek", "degen", "sikisik", "kalabalik"])
def test_counts_synthetic_stacks(tmp_path, cfg, variant):
    """Touching and overlapping bars must still be counted one by one."""
    path, truth = _write(tmp_path, variant, **VARIANTS[variant])
    result = count_image(path, cfg)
    error_pct = 100.0 * abs(result["count"] - truth) / truth
    assert error_pct <= MAX_ERROR_PCT, f"{variant}: got {result['count']}, expected {truth}"


def test_json_contract(tmp_path, cfg):
    """SPEC 6: {count, points[[x, y, r]], method, elapsed_ms}."""
    path, _ = _write(tmp_path, "degen", **VARIANTS["degen"])
    result = count_image(path, cfg)

    assert set(result) == {"count", "points", "method", "elapsed_ms"}
    assert result["method"] == "classical"
    assert result["count"] == len(result["points"])
    assert isinstance(result["elapsed_ms"], int)
    assert all(len(point) == 3 for point in result["points"])
    assert all(r > 0 for _, _, r in result["points"])


def test_debug_writes_every_stage(tmp_path, cfg):
    """--debug is the only way to tune this pipeline, so the stages are part of the contract."""
    path, _ = _write(tmp_path, "degen", **VARIANTS["degen"])
    debug_dir = tmp_path / "debug"
    count_image(path, cfg, debug_dir=debug_dir)

    expected = {f"{i:02d}_{stage}.png" for i, stage in enumerate(draw.STAGES, start=1)}
    assert {p.name for p in debug_dir.iterdir()} == expected


def test_points_are_in_original_image_coordinates(tmp_path, cfg):
    """A photo bigger than io.max_long_edge is processed small but reported full size."""
    scene = render(count=180, size=(2400, 1800), seed=3)
    path = tmp_path / f"20260101_1200_{scene.count}_12_buyuk.png"
    write_image(path, scene.image)

    cfg["io"]["max_long_edge"] = 1200
    result = count_image(path, cfg)

    xs = [x for x, _, _ in result["points"]]
    ys = [y for _, y, _ in result["points"]]
    assert max(xs) > 1200, "points look like they are still in processing coordinates"
    assert max(xs) <= 2400 and max(ys) <= 1800


def test_ml_method_reports_missing_weights_clearly(tmp_path, cfg):
    """Asking for the ML engine without a trained model must say so, not crash obscurely."""
    path, _ = _write(tmp_path, "degen", **VARIANTS["degen"])
    with pytest.raises(FileNotFoundError, match="train first"):
        count_image(path, cfg, method="ml", weights=tmp_path / "missing.pt")


def test_ground_truth_parsing(cfg):
    """SPEC 7 filenames only; anything else must be reported, never silently counted."""
    parsed = evaluation.parse_ground_truth("20260910_1430_412_12_islak.jpg", cfg)
    assert parsed == {"count": 412, "diameter": 12, "condition": "islak"}

    for bad in ["IMG_1234.jpg", "20260910_412_12_islak.jpg", "20260910_1430_412_12.jpg"]:
        assert evaluation.parse_ground_truth(bad, cfg) is None


def test_summary_metrics(cfg):
    rows = [
        {"error": 0, "error_pct": 0.0, "elapsed_ms": 100},
        {"error": -4, "error_pct": 2.0, "elapsed_ms": 200},
        {"error": 2, "error_pct": 4.0, "elapsed_ms": 300},
    ]
    summary = evaluation.summarize(rows, cfg)
    assert summary["mae"] == pytest.approx(2.0)
    assert summary["mape"] == pytest.approx(2.0)
    assert summary["within_rate"] == pytest.approx(200.0 / 3)  # two of three within +/-2


def test_merge_close_collapses_slivers(cfg):
    """A bar split into a disc plus a sliver must be counted once."""
    big = detect.Detection(x=100, y=100, r=10, area=314, circularity=0.9, fill_ratio=0.9)
    sliver = detect.Detection(x=103, y=100, r=3, area=28, circularity=0.7, fill_ratio=0.7)
    far = detect.Detection(x=140, y=100, r=10, area=314, circularity=0.9, fill_ratio=0.9)

    merged = detect.merge_close([big, sliver, far], cfg)
    assert len(merged) == 2
    assert sliver not in merged


def test_radius_estimate_ignores_debris(cfg):
    """Many tiny fragments must not drag the scale estimate down; area decides."""
    bars = [
        detect.Detection(x=i, y=0, r=20.0, area=1256.0, circularity=0.9, fill_ratio=0.9)
        for i in range(20)
    ]
    debris = [
        detect.Detection(x=i, y=50, r=2.0, area=12.0, circularity=0.9, fill_ratio=0.9)
        for i in range(60)
    ]
    assert detect.estimate_radius(bars + debris, cfg) == pytest.approx(20.0, abs=1.0)


def test_split_separates_two_touching_discs(cfg):
    """The one step the whole approach depends on, isolated."""
    canvas = np.zeros((120, 200), np.uint8)
    cv2.circle(canvas, (70, 60), 30, 255, -1)
    cv2.circle(canvas, (125, 60), 30, 255, -1)
    assert cv2.connectedComponents(canvas)[0] - 1 == 1  # one merged blob to start with

    separated = segment.split_touching(canvas, canvas, cfg, radius_hint=30.0)
    assert cv2.connectedComponents(separated)[0] - 1 == 2


def test_flatness_mask_survives_neighbours_of_different_brightness(cfg):
    """The failure that sinks a plain adaptive threshold: a dark bar beside a bright one."""
    canvas = np.full((160, 300), 240, np.uint8)
    cv2.circle(canvas, (90, 80), 40, 100, -1)  # dark bar
    cv2.circle(canvas, (200, 80), 40, 190, -1)  # bright bar

    mask = segment.clean_mask(segment.binarize(canvas, cfg), cfg)
    assert mask[80, 90] == 255, "dark bar was lost"
    assert mask[80, 200] == 255, "bright bar was lost"


def test_load_image_handles_non_ascii_path(tmp_path, cfg):
    path = tmp_path / "istif_ölçüm_şantiye.png"
    write_image(path, render(count=40, seed=1).image)
    assert preprocess.load_image(path).shape[2] == 3
