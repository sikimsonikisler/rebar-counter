#!/usr/bin/env python3
"""Phase 3 - train the YOLO detector on tiled, labelled photos.

    python python/ml/train.py --data out/synth_yolo/dataset.yaml
    python python/ml/train.py --data data/yolo/dataset.yaml --device mps

Every hyper-parameter comes from config.yaml (ml.train), so a run is reproducible
from the config alone. The base weights are downloaded once at development time;
the shipped app never talks to the network - that constraint is about the phone.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from count import REPO_ROOT, load_config  # noqa: E402

# Keep ultralytics from phoning home during a run.
os.environ.setdefault("YOLO_OFFLINE", "1")


def train(data_yaml: Path, cfg: dict, overrides: dict) -> Path:
    from ultralytics import YOLO, settings

    settings.update({"sync": False})

    train_cfg = cfg["ml"]["train"]
    augment_cfg = train_cfg["augment"]

    model = YOLO(str(overrides.get("model") or train_cfg["model"]))
    results = model.train(
        data=str(data_yaml),
        epochs=int(overrides.get("epochs") or train_cfg["epochs"]),
        imgsz=int(train_cfg["image_size"]),
        batch=int(train_cfg["batch"]),
        patience=int(train_cfg["patience"]),
        seed=int(train_cfg["seed"]),
        project=str(overrides["project"]),
        name=str(overrides["name"]),
        exist_ok=True,
        device=overrides.get("device"),
        workers=int(overrides.get("workers", 4)),
        verbose=True,
        plots=False,
        # Cross-sections do not care which way up they are, so rotation and both
        # flips are free extra data. Erasing is off on purpose: cutting a bar out
        # of the picture without changing the label teaches the model to miscount.
        degrees=float(augment_cfg["degrees"]),
        scale=float(augment_cfg["scale"]),
        translate=float(augment_cfg["translate"]),
        fliplr=float(augment_cfg["fliplr"]),
        flipud=float(augment_cfg["flipud"]),
        hsv_h=float(augment_cfg["hsv_h"]),
        hsv_s=float(augment_cfg["hsv_s"]),
        hsv_v=float(augment_cfg["hsv_v"]),
        mosaic=float(augment_cfg["mosaic"]),
        close_mosaic=int(augment_cfg["close_mosaic"]),
        erasing=float(augment_cfg["erasing"]),
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the rebar detector.")
    parser.add_argument("--data", required=True, help="dataset.yaml from ml/dataset.py")
    parser.add_argument("--config", default=None)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--name", default="rebar")
    parser.add_argument("--model", default=None, help="override ml.train.model")
    parser.add_argument("--epochs", type=int, default=None, help="override ml.train.epochs")
    parser.add_argument("--device", default=None, help="cpu | mps | 0 ; default lets YOLO pick")
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"no dataset.yaml at {data_yaml} - run python/ml/dataset.py first", file=sys.stderr)
        return 2

    weights = train(
        data_yaml,
        cfg,
        {
            "project": args.project,
            "name": args.name,
            "model": args.model,
            "epochs": args.epochs,
            "device": args.device,
            "workers": args.workers,
        },
    )
    print(f"\nbest weights: {weights}")
    print("next: python python/eval.py <folder> --methods classical,ml --weights " + str(weights))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
