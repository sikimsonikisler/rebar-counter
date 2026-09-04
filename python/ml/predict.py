#!/usr/bin/env python3
"""Phase 3 - the ML counting engine, behind the same contract as the classical one.

    python python/ml/predict.py <image> --weights runs/rebar/weights/best.pt --json

Returns exactly what count.py returns, with "method": "ml". Same CLI, swappable
engine - that is the whole point of keeping the contract identical.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from count import REPO_ROOT, load_config  # noqa: E402
from ml import tile as tiling  # noqa: E402
from pipeline import draw, preprocess  # noqa: E402

# One model per weights file per process: eval.py runs hundreds of images and
# reloading the network each time would dominate the measured time.
_MODEL_CACHE: dict[str, object] = {}


def resolve_weights(cfg: dict, weights: str | Path | None) -> Path:
    path = Path(weights) if weights else REPO_ROOT / cfg["ml"]["weights"]
    if not path.exists():
        raise FileNotFoundError(
            f"no weights at {path} - train first with python/ml/train.py, "
            "or pass --weights"
        )
    return path


def load_model(weights: Path, device: str | None = None):
    key = str(weights.resolve())
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO

        model = YOLO(key)
        # One throwaway inference. Without it the first real photo pays for lazy
        # graph setup - a second or so - and eval.py reports that as counting time.
        model.predict(
            np.zeros((32, 32, 3), dtype=np.uint8), verbose=False, device=device
        )
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def make_tile_predictor(model, cfg: dict, device: str | None = None):
    """Wrap the network in the (boxes, scores) callable tile.run_tiled expects."""
    nms_cfg = cfg["ml"]["nms"]
    image_size = int(cfg["ml"]["train"]["image_size"])

    def predict(tile_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        results = model.predict(
            tile_image,
            imgsz=image_size,
            conf=float(nms_cfg["score_threshold"]),
            iou=float(nms_cfg["iou_threshold"]),
            max_det=int(nms_cfg["max_detections_per_tile"]),
            verbose=False,
            device=device,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        return boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()

    return predict


def count_image_ml(
    image_path: str | Path,
    cfg: dict,
    weights: str | Path | None = None,
    debug_dir: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """The SPEC 6 JSON contract, produced by the detector instead of the thresholds."""
    debug = draw.DebugWriter(debug_dir, stages=draw.ML_STAGES)
    model = load_model(resolve_weights(cfg, weights), device)

    started = time.perf_counter()
    original = preprocess.load_image(image_path)
    # No downscale here: tiling exists precisely so the model sees native pixels.
    boxes, _ = tiling.run_tiled(original, make_tile_predictor(model, cfg, device), cfg)
    points = tiling.boxes_to_points(boxes)
    elapsed_ms = (time.perf_counter() - started - debug.io_seconds) * 1000.0

    if debug:
        height, width = original.shape[:2]
        debug.save("tiles", draw.draw_tiles(original, tiling.plan_tiles(width, height, cfg), cfg))
        detections = [
            draw_detection(x, y, r) for x, y, r in points
        ]
        debug.save("result", draw.render_result(original, detections, cfg))

    return {
        "count": len(points),
        "points": points,
        "method": "ml",
        "elapsed_ms": round(elapsed_ms),
    }


def draw_detection(x: float, y: float, r: float):
    """Adapter so draw.render_result can stay ignorant of which engine produced the points."""
    from pipeline.detect import Detection

    return Detection(x=x, y=y, r=r, area=3.14159 * r * r, circularity=1.0, fill_ratio=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count rebar cross-sections with the ML engine.")
    parser.add_argument("image_path")
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--debug", default=None, metavar="OUT_DIR")
    parser.add_argument("--device", default=None, help="cpu | mps | 0")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    result = count_image_ml(
        args.image_path, cfg, weights=args.weights, debug_dir=args.debug, device=args.device
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"count={result['count']}  method=ml  elapsed={result['elapsed_ms']}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
