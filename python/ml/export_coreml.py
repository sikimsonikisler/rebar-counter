#!/usr/bin/env python3
"""Phase 4 - convert the trained detector to Core ML and prove it still counts.

    python python/ml/export_coreml.py --weights runs/rebar/weights/best.pt

Does three things:
  1. exports the .pt to a float16 .mlpackage,
  2. writes counting_config.json next to it, straight from config.yaml, so the phone
     and the Python pipeline cannot drift apart on tile size or thresholds,
  3. re-counts the same photos with both engines and refuses to call the export good
     unless every count matches exactly (SPEC 6 phase 4).

A conversion that silently changes the count is worse than no conversion, hence 3.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from count import REPO_ROOT, load_config  # noqa: E402
from prelabel import find_images  # noqa: E402

DEFAULT_RESOURCES = REPO_ROOT / "ios" / "RebarCounter" / "RebarCounter" / "Resources"
MODEL_NAME = "RebarDetector"


def export(weights: Path, cfg: dict, out_dir: Path) -> Path:
    from ultralytics import YOLO

    export_cfg = cfg["ml"]["export"]
    model = YOLO(str(weights))
    produced = model.export(
        format="coreml",
        imgsz=int(cfg["ml"]["train"]["image_size"]),
        # `quantize=16`, not the old `half=True`: the deprecated flag takes a code
        # path that fails to convert the Detect head outright.
        quantize=16 if export_cfg["quantize_float16"] else None,
        # NMS stays out of the model on purpose: the built-in layer caps its output,
        # and a dense tile can hold several hundred bars. Swift does the merging,
        # with the same rules as python/ml/tile.py.
        nms=False,
    )
    produced = Path(produced)

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{MODEL_NAME}.mlpackage"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(produced), str(target))
    return target


def write_counting_config(cfg: dict, out_dir: Path) -> Path:
    """The only numbers the Swift side needs, taken from the single source of truth."""
    tiling_cfg = cfg["ml"]["tiling"]
    nms_cfg = cfg["ml"]["nms"]
    payload = {
        "tileSize": int(tiling_cfg["tile_size"]),
        "maxLongEdge": int(tiling_cfg["max_long_edge"]),
        "overlap": float(tiling_cfg["overlap"]),
        "edgeMarginRatio": float(tiling_cfg["edge_margin_ratio"]),
        "iouThreshold": float(nms_cfg["iou_threshold"]),
        "scoreThreshold": float(nms_cfg["score_threshold"]),
        "maxDetectionsPerTile": int(nms_cfg["max_detections_per_tile"]),
        "modelName": MODEL_NAME,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "counting_config.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def verify(weights: Path, package: Path, folder: Path, cfg: dict, limit: int) -> list[dict]:
    """Count the same photos with the PyTorch weights and with the converted model."""
    from ml.predict import count_image_ml

    rows = []
    for image_path in find_images(folder)[:limit]:
        torch_result = count_image_ml(image_path, cfg, weights=weights)
        coreml_result = count_image_ml(image_path, cfg, weights=package)
        rows.append(
            {
                "file": image_path.name,
                "pytorch": torch_result["count"],
                "coreml": coreml_result["count"],
                "difference": coreml_result["count"] - torch_result["count"],
                "pytorch_ms": torch_result["elapsed_ms"],
                "coreml_ms": coreml_result["elapsed_ms"],
            }
        )
    return rows


def print_verification(rows: list[dict]) -> bool:
    if not rows:
        print("no images to verify against")
        return True
    width = max(len(row["file"]) for row in rows)
    print(f"\n{'file'.ljust(width)}  {'pytorch':>8} {'coreml':>7} {'diff':>5} {'pt ms':>7} {'cml ms':>7}")
    print("-" * (width + 40))
    for row in rows:
        print(
            f"{row['file'].ljust(width)}  {row['pytorch']:>8} {row['coreml']:>7} "
            f"{row['difference']:>+5} {row['pytorch_ms']:>7} {row['coreml_ms']:>7}"
        )
    mismatches = [row for row in rows if row["difference"] != 0]
    print("-" * (width + 40))
    if mismatches:
        print(f"FAILED: {len(mismatches)} of {len(rows)} images disagree")
        return False
    print(f"OK: all {len(rows)} images agree exactly")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the detector to Core ML.")
    parser.add_argument("--weights", default=None, help="trained .pt; defaults to ml.weights")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=str(DEFAULT_RESOURCES), help="iOS Resources folder")
    parser.add_argument("--verify-folder", default=None, help="photos to compare both engines on")
    parser.add_argument("--verify-count", type=int, default=None)
    parser.add_argument("--skip-verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    weights = Path(args.weights) if args.weights else REPO_ROOT / cfg["ml"]["weights"]
    if not weights.exists():
        print(f"no weights at {weights} - train first", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    package = export(weights, cfg, out_dir)
    config_path = write_counting_config(cfg, out_dir)
    print(f"\nmodel  -> {package}")
    print(f"config -> {config_path}")

    if args.skip_verify:
        print("verification skipped (--skip-verify)")
        return 0

    folder = Path(args.verify_folder) if args.verify_folder else REPO_ROOT / "out" / "synthetic"
    if not folder.is_dir():
        print(f"no folder to verify against at {folder}; pass --verify-folder")
        return 0

    limit = args.verify_count or int(cfg["ml"]["export"]["compare_images"])
    rows = verify(weights, package, folder, cfg, limit)
    return 0 if print_verification(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
