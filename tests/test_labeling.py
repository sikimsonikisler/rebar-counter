"""Phase 2 - pre-labelling and dataset assembly.

The one thing that must never happen is an eval photo sneaking into training, so
that guard gets a test of its own.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import prelabel
from make_synthetic import render, write_image
from ml import dataset
from ml import tile as tiling


def _photo(folder: Path, stem: str, count: int = 60) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    scene = render(count=count, seed=5)
    path = folder / f"{stem}.png"
    write_image(path, scene.image)
    return path


def test_circles_become_boxes_clamped_to_the_image(cfg):
    points = [[50.0, 50.0, 10.0], [2.0, 2.0, 10.0]]
    boxes = prelabel.to_boxes(points, width=100, height=100, cfg=cfg)
    assert boxes[0] == (40.0, 40.0, 60.0, 60.0)
    # A bar at the very corner keeps a valid box instead of a negative one.
    assert boxes[1][0] == 0.0 and boxes[1][1] == 0.0


def test_yolo_labels_are_normalised(tmp_path, cfg):
    path = tmp_path / "labels.txt"
    prelabel.write_yolo(path, [(40.0, 30.0, 60.0, 50.0)], width=200, height=100, cfg=cfg)
    fields = path.read_text().split()
    assert fields[0] == "0"
    assert float(fields[1]) == pytest.approx(0.25)  # cx = 50/200
    assert float(fields[2]) == pytest.approx(0.40)  # cy = 40/100
    assert float(fields[3]) == pytest.approx(0.10)  # w  = 20/200


def test_label_studio_task_carries_a_correctable_prediction(tmp_path, cfg):
    task = prelabel.label_studio_task(
        tmp_path / "istif.jpg", [(0.0, 0.0, 50.0, 25.0)], width=100, height=100, cfg=cfg
    )
    json.dumps(task)  # must be serialisable as-is
    result = task["predictions"][0]["result"][0]
    assert result["value"]["rectanglelabels"] == ["rebar"]
    assert result["value"]["width"] == pytest.approx(50.0)  # percent, not pixels
    assert task["data"]["image"].endswith("istif.jpg")


def test_prelabel_end_to_end(tmp_path, cfg):
    images = tmp_path / "images"
    _photo(images, "20260101_1200_60_12_sentetik")
    out = tmp_path / "labels"
    prelabel.main([str(images), "--out", str(out), "--format", "yolo"])

    label_file = out / "20260101_1200_60_12_sentetik.txt"
    assert label_file.exists()
    assert len(label_file.read_text().strip().splitlines()) > 40


def test_eval_photos_in_the_training_set_are_a_hard_error(tmp_path):
    """SPEC 8: data/eval/ never trains anything. Same filename in both = leak."""
    train = tmp_path / "raw"
    evaluation = tmp_path / "eval"
    _photo(train, "20260101_1200_60_12_ayni", count=30)
    _photo(evaluation, "20260101_1200_60_12_ayni", count=30)

    with pytest.raises(dataset.EvalLeak, match="20260101_1200_60_12_ayni"):
        dataset.assert_disjoint(train, evaluation)


def test_disjoint_sets_pass(tmp_path):
    train = tmp_path / "raw"
    evaluation = tmp_path / "eval"
    _photo(train, "20260101_1200_60_12_egitim", count=30)
    _photo(evaluation, "20260101_1300_60_12_test", count=30)
    assert dataset.assert_disjoint(train, evaluation) == ["20260101_1200_60_12_egitim"]


def test_split_is_deterministic_and_roughly_the_right_size():
    """Re-running must not reshuffle train and val, or two runs are incomparable."""
    stems = [f"photo_{i}" for i in range(400)]
    first = [dataset.is_validation(stem, 0.15, 0) for stem in stems]
    second = [dataset.is_validation(stem, 0.15, 0) for stem in stems]
    assert first == second
    assert 0.10 < sum(first) / len(first) < 0.22


def test_labels_survive_the_round_trip_through_a_tile(cfg):
    boxes = np.array([[100.0, 100.0, 140.0, 140.0], [10.0, 10.0, 50.0, 50.0]])
    window = tiling.Tile(
        left=60, top=60, right=700, bottom=700,
        touches_left=False, touches_top=False, touches_right=False, touches_bottom=False,
    )
    kept = dataset.boxes_for_tile(boxes, window, cfg)
    # The first box is fully inside and shifts by the tile origin; the second is
    # mostly outside the crop and is dropped rather than turned into a stub label.
    assert len(kept) == 1
    assert kept[0].tolist() == [40.0, 40.0, 80.0, 80.0]


def test_partly_visible_labels_are_dropped_not_clipped(cfg):
    cfg["ml"]["tiling"]["train_min_box_visibility"] = 0.6
    boxes = np.array([[-30.0, 10.0, 10.0, 50.0]])  # only a quarter of it is in frame
    window = tiling.Tile(
        left=0, top=0, right=640, bottom=640,
        touches_left=False, touches_top=True, touches_right=False, touches_bottom=False,
    )
    assert len(dataset.boxes_for_tile(boxes, window, cfg)) == 0


def test_dataset_build_produces_a_trainable_layout(tmp_path, cfg):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    for index in range(4):
        _photo(images, f"2026010{index}_1200_120_12_sentetik", count=120)
    prelabel.main([str(images), "--out", str(labels), "--format", "yolo"])

    out = tmp_path / "yolo"
    counts = dataset.build(images, labels, out, tmp_path / "eval", cfg)

    assert counts["photos"] == 4
    assert counts["train"] + counts["val"] > 0
    assert (out / "dataset.yaml").exists()
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            assert (out / kind / split).is_dir()
    # Every tile image must have a label file beside it.
    for split in ("train", "val"):
        for image_path in (out / "images" / split).glob("*.png"):
            assert (out / "labels" / split / f"{image_path.stem}.txt").exists()
