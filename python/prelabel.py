#!/usr/bin/env python3
"""Phase 2 - turn the classical baseline's output into draft labels.

    python python/prelabel.py data/raw --out data/labels --format yolo
    python python/prelabel.py data/raw --out data/labels --format label-studio

The point is speed of labelling, not accuracy: a human opens the pre-labelled image
and corrects the handful the baseline got wrong instead of clicking 400 circles.
Detections are circles, so each one becomes a square box of side
`prelabel.box_side_radius_ratio` x radius, clamped to the image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from count import count_image, load_config  # noqa: E402
from pipeline import preprocess  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES)


def to_boxes(points: list[list[float]], width: int, height: int, cfg: dict) -> list[tuple]:
    """Circle (x, y, r) -> pixel box (left, top, right, bottom), clamped to the image."""
    ratio = float(cfg["prelabel"]["box_side_radius_ratio"])
    boxes = []
    for x, y, r in points:
        half = r * ratio / 2.0
        left = max(0.0, x - half)
        top = max(0.0, y - half)
        right = min(float(width), x + half)
        bottom = min(float(height), y + half)
        if right > left and bottom > top:
            boxes.append((left, top, right, bottom))
    return boxes


def write_yolo(path: Path, boxes: list[tuple], width: int, height: int, cfg: dict) -> None:
    """One .txt beside each image: `class cx cy w h`, all normalised to 0-1."""
    class_index = int(cfg["prelabel"]["class_index"])
    lines = []
    for left, top, right, bottom in boxes:
        cx = (left + right) / 2.0 / width
        cy = (top + bottom) / 2.0 / height
        w = (right - left) / width
        h = (bottom - top) / height
        lines.append(f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def label_studio_task(image: Path, boxes: list[tuple], width: int, height: int, cfg: dict) -> dict:
    """One Label Studio task carrying the baseline result as a prediction to correct."""
    studio_cfg = cfg["prelabel"]["label_studio"]
    class_name = cfg["prelabel"]["class_name"]
    results = []
    for index, (left, top, right, bottom) in enumerate(boxes):
        results.append(
            {
                "id": f"box{index}",
                "type": "rectanglelabels",
                "from_name": studio_cfg["from_name"],
                "to_name": studio_cfg["to_name"],
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                # Label Studio wants percentages of the image, not pixels.
                "value": {
                    "x": 100.0 * left / width,
                    "y": 100.0 * top / height,
                    "width": 100.0 * (right - left) / width,
                    "height": 100.0 * (bottom - top) / height,
                    "rotation": 0,
                    "rectanglelabels": [class_name],
                },
            }
        )
    return {
        "data": {"image": f"{studio_cfg['image_root']}{image.name}"},
        "predictions": [{"model_version": "classical-baseline", "result": results}],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-label photos with the classical baseline.")
    parser.add_argument("folder", help="folder of photos to pre-label")
    parser.add_argument("--out", required=True, help="where the labels are written")
    parser.add_argument("--format", choices=("yolo", "label-studio"), default="yolo")
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", default="classical", help="engine used to draft the labels")
    parser.add_argument("--limit", type=int, default=0, help="stop after N images (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="redo images that already have labels")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(folder)
    if args.limit:
        images = images[: args.limit]

    tasks, total_boxes, skipped = [], 0, 0
    for image_path in images:
        label_path = out_dir / f"{image_path.stem}.txt"
        if args.format == "yolo" and label_path.exists() and not args.overwrite:
            skipped += 1
            continue

        height, width = preprocess.load_image(image_path).shape[:2]
        result = count_image(image_path, cfg, method=args.method)
        boxes = to_boxes(result["points"], width, height, cfg)
        total_boxes += len(boxes)

        if args.format == "yolo":
            write_yolo(label_path, boxes, width, height, cfg)
        else:
            tasks.append(label_studio_task(image_path, boxes, width, height, cfg))
        print(f"{image_path.name}: {len(boxes)} boxes")

    if args.format == "label-studio":
        target = out_dir / "prelabels.json"
        target.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{len(tasks)} tasks -> {target}")
    else:
        print(f"\n{len(images) - skipped} labelled, {skipped} already had labels -> {out_dir}")
    print(f"{total_boxes} draft boxes total; correct them in the labelling tool, do not trust them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
