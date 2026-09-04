#!/usr/bin/env python3
"""Coordinate-descent tuning of config.yaml thresholds on a DEVELOPMENT folder.

    python python/tune.py out/synthetic --rounds 2 --out config.tuned.yaml

SPEC 8 is explicit that thresholds are never tuned on data/eval/, so this script
refuses to run there. Point it at synthetic images or at a development split.

One key at a time, every candidate value is scored by MAPE (MAE breaks ties) and
the best is kept before moving to the next key - cheap, and it never makes the
headline metric worse than the config it started from.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from count import load_config  # noqa: E402
from eval import evaluate  # noqa: E402

# Dotted config path -> values to try. Keep this short; every entry costs a full pass.
SEARCH_SPACE: dict[str, list] = {
    "preprocess.bilateral.diameter": [5, 7, 9],
    "preprocess.bilateral.sigma_color": [25, 50, 80],
    "preprocess.clahe.clip_limit": [1.5, 2.5, 4.0],
    "segment.flatness.block_size": [7, 9, 13, 17, 23],
    "segment.flatness.keep_percentile": [60, 64, 68, 72, 78],
    "segment.flatness.block_size_radius_ratio": [0.0, 0.25, 0.4, 0.6],
    "segment.morphology.open_kernel": [3, 5],
    "segment.morphology.close_kernel": [3, 5, 7],
    "segment.split.peak_kernel_radius_ratio": [0.7, 0.9, 1.1, 1.4],
    "segment.split.min_peak_dist_radius_ratio": [0.2, 0.35, 0.5, 0.65],
    "segment.split.initial_peak_kernel": [5, 7, 11, 15],
    "detect.shape.min_circularity": [0.3, 0.45, 0.6],
    "detect.shape.min_fill_ratio": [0.4, 0.55, 0.7],
    "detect.relative.min_area_ratio": [0.2, 0.3, 0.45, 0.6],
    "detect.relative.max_area_ratio": [1.6, 2.0, 2.6],
    "detect.merge_distance_ratio": [0.0, 0.35, 0.55, 0.8],
}


def get_path(cfg: dict, dotted: str):
    node = cfg
    for key in dotted.split("."):
        node = node[key]
    return node


def set_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def score(folder: Path, cfg: dict) -> tuple[float, float]:
    """(MAPE, MAE) - lower is better on both, MAPE decides."""
    summary = evaluate(folder, cfg, "classical")["summary"]
    if not summary["images"]:
        return (float("inf"), float("inf"))
    return (summary["mape"], summary["mae"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune config thresholds on a dev folder.")
    parser.add_argument("folder", help="development images (NOT data/eval/)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="config.tuned.yaml")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args(argv)

    folder = Path(args.folder).resolve()
    if "eval" in folder.parts and "data" in folder.parts:
        print("refusing to tune on data/eval/ - see SPEC 8", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    best_score = score(folder, cfg)
    print(f"start  MAPE={best_score[0]:.2f}%  MAE={best_score[1]:.2f}")

    for round_index in range(args.rounds):
        improved = False
        for dotted, values in SEARCH_SPACE.items():
            current = get_path(cfg, dotted)
            for value in values:
                if value == current:
                    continue
                trial = copy.deepcopy(cfg)
                set_path(trial, dotted, value)
                trial_score = score(folder, trial)
                if trial_score < best_score:
                    best_score, cfg, current = trial_score, trial, value
                    improved = True
                    print(f"  {dotted} = {value}  ->  MAPE={best_score[0]:.2f}% MAE={best_score[1]:.2f}")
        print(f"round {round_index + 1} done: MAPE={best_score[0]:.2f}%")
        if not improved:
            break

    with open(args.out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    print(f"\nbest MAPE={best_score[0]:.2f}%  MAE={best_score[1]:.2f}  written to {args.out}")
    print("comments are lost in the dump - copy the changed values into config.yaml by hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
