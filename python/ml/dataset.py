#!/usr/bin/env python3
"""Phase 2/3 - assemble the YOLO dataset and prove the eval set stayed out of it.

    python python/ml/dataset.py --check-only          # SPEC 6 phase 2 acceptance
    python python/ml/dataset.py --out data/yolo       # build train/val for training

SPEC 8 makes data/eval/ untouchable. The cheapest way for it to leak into training is
the same photo living in both folders, so that is checked by filename before anything
is built, and a collision is a hard error, never a warning.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from count import REPO_ROOT, load_config  # noqa: E402
from ml import tile as tiling  # noqa: E402
from pipeline import preprocess  # noqa: E402
from prelabel import find_images  # noqa: E402


class EvalLeak(Exception):
    """Raised when a photo appears in both the training set and data/eval/."""


def stems(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    return {p.stem: p for p in find_images(folder)}


def assert_disjoint(train_folder: Path, eval_folder: Path) -> list[str]:
    """SPEC 6 phase 2 acceptance: the two sets share no filename."""
    overlap = sorted(set(stems(train_folder)) & set(stems(eval_folder)))
    if overlap:
        raise EvalLeak(
            f"{len(overlap)} photo(s) are in both {train_folder} and {eval_folder}: "
            + ", ".join(overlap[:5])
            + (" ..." if len(overlap) > 5 else "")
        )
    return sorted(stems(train_folder))


def is_validation(stem: str, fraction: float, seed: int) -> bool:
    """Deterministic split: the same photo always lands in the same half.

    Hash-based rather than random so re-running never quietly reshuffles train and
    val, which would make two training runs incomparable.
    """
    digest = hashlib.sha256(f"{seed}:{stem}".encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") % 10_000) < fraction * 10_000


def read_yolo(path: Path, width: int, height: int) -> np.ndarray:
    """YOLO `class cx cy w h` (normalised) -> absolute (x1, y1, x2, y2)."""
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = (float(v) for v in parts)
        boxes.append(
            [
                (cx - w / 2.0) * width,
                (cy - h / 2.0) * height,
                (cx + w / 2.0) * width,
                (cy + h / 2.0) * height,
            ]
        )
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


def boxes_for_tile(boxes: np.ndarray, window: tiling.Tile, cfg: dict) -> np.ndarray:
    """Clip labels into a tile, dropping the ones a crop cut to pieces."""
    if boxes.size == 0:
        return boxes
    minimum = float(cfg["ml"]["tiling"]["train_min_box_visibility"])
    shifted = boxes.copy()
    shifted[:, [0, 2]] -= window.left
    shifted[:, [1, 3]] -= window.top

    clipped = shifted.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, window.width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, window.height)

    original_area = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
    kept_area = np.maximum(0.0, clipped[:, 2] - clipped[:, 0]) * np.maximum(
        0.0, clipped[:, 3] - clipped[:, 1]
    )
    visible = np.where(original_area > 0, kept_area / original_area, 0.0)
    return clipped[visible >= minimum]


def write_yolo_boxes(path: Path, boxes: np.ndarray, width: int, height: int, class_index: int) -> None:
    lines = []
    for x1, y1, x2, y2 in boxes:
        lines.append(
            f"{class_index} {(x1 + x2) / 2.0 / width:.6f} {(y1 + y2) / 2.0 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def link_or_copy(source: Path, target: Path) -> None:
    """Symlink so a few hundred photos are not duplicated on disk; copy if that fails."""
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        target.write_bytes(source.read_bytes())


def build(
    images_folder: Path,
    labels_folder: Path,
    out_folder: Path,
    eval_folder: Path,
    cfg: dict,
    use_tiles: bool = True,
) -> dict:
    usable = assert_disjoint(images_folder, eval_folder)
    image_paths = stems(images_folder)

    train_cfg = cfg["ml"]["train"]
    fraction = float(train_cfg["val_fraction"])
    seed = int(train_cfg["seed"])
    class_index = int(cfg["prelabel"]["class_index"])
    min_boxes = int(cfg["ml"]["tiling"]["train_min_boxes_per_tile"])

    counts = {"train": 0, "val": 0, "unlabelled": [], "photos": 0}
    for split in ("train", "val"):
        (out_folder / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_folder / "labels" / split).mkdir(parents=True, exist_ok=True)

    for stem in usable:
        label_path = labels_folder / f"{stem}.txt"
        if not label_path.exists():
            counts["unlabelled"].append(stem)
            continue
        # The split is per photo, never per tile: tiles of one photo overlap, so
        # scattering them across train and val would leak the validation set.
        split = "val" if is_validation(stem, fraction, seed) else "train"
        image_path = image_paths[stem]
        counts["photos"] += 1

        if not use_tiles:
            link_or_copy(image_path, out_folder / "images" / split / image_path.name)
            link_or_copy(label_path, out_folder / "labels" / split / label_path.name)
            counts[split] += 1
            continue

        image = preprocess.load_image(image_path)
        height, width = image.shape[:2]
        boxes = read_yolo(label_path, width, height)

        for index, window in enumerate(tiling.plan_tiles(width, height, cfg)):
            tile_boxes = boxes_for_tile(boxes, window, cfg)
            if len(tile_boxes) < min_boxes:
                continue
            tile_stem = f"{stem}__t{index:03d}"
            ok, buffer = cv2.imencode(".png", tiling.crop(image, window))
            if not ok:
                continue
            buffer.tofile(str(out_folder / "images" / split / f"{tile_stem}.png"))
            write_yolo_boxes(
                out_folder / "labels" / split / f"{tile_stem}.txt",
                tile_boxes,
                window.width,
                window.height,
                class_index,
            )
            counts[split] += 1

    dataset_yaml = {
        "path": str(out_folder.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {class_index: cfg["prelabel"]["class_name"]},
    }
    target = out_folder / "dataset.yaml"
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False, allow_unicode=True)
    counts["dataset_yaml"] = target
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the YOLO dataset from labelled photos.")
    parser.add_argument("--images", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--labels", default=str(REPO_ROOT / "data" / "labels"))
    parser.add_argument("--eval", default=str(REPO_ROOT / "data" / "eval"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "yolo"))
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--check-only", action="store_true", help="only verify eval/train separation"
    )
    parser.add_argument(
        "--no-tile", action="store_true", help="use whole photos instead of slicing them"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    images_folder, eval_folder = Path(args.images), Path(args.eval)

    try:
        usable = assert_disjoint(images_folder, eval_folder)
    except EvalLeak as leak:
        print(f"EVAL LEAK: {leak}", file=sys.stderr)
        return 1

    print(f"training photos : {len(usable)} in {images_folder}")
    print(f"eval photos     : {len(stems(eval_folder))} in {eval_folder}")
    print("separation      : OK, no shared filenames")
    if args.check_only:
        return 0

    counts = build(
        images_folder,
        Path(args.labels),
        Path(args.out),
        eval_folder,
        cfg,
        use_tiles=not args.no_tile,
    )
    unit = "photos" if args.no_tile else "tiles"
    print(f"labelled photos : {counts['photos']}")
    print(f"train / val     : {counts['train']} / {counts['val']} {unit}")
    if counts["unlabelled"]:
        print(f"unlabelled      : {len(counts['unlabelled'])} (skipped, run prelabel.py first)")
    print(f"dataset.yaml    : {counts['dataset_yaml']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
