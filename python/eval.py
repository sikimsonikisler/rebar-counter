#!/usr/bin/env python3
"""Evaluate the counter against a folder of photos with known counts (SPEC 8).

    python python/eval.py data/eval/ --config config.yaml

Ground truth comes from the filename (SPEC 7):
    YYYYMMDD_hhmm_<truecount>_<diameter>_<condition>.jpg

Reports MAE, MAPE (the headline metric) and the "within +/-N pieces" rate.
This must be re-run after every image-processing change.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from count import count_image, load_config  # noqa: E402


def parse_ground_truth(filename: str, cfg: dict) -> dict | None:
    """Pull the hand-counted truth out of the filename; None if it does not conform."""
    match = re.match(cfg["eval"]["filename_pattern"], filename)
    if not match:
        return None
    fields = match.groupdict()
    return {
        "count": int(fields["count"]),
        "diameter": int(fields["diameter"]),
        "condition": fields["condition"],
    }


def collect_images(folder: Path, cfg: dict) -> tuple[list[tuple[Path, dict]], list[str]]:
    """Returns (usable images with their truth, filenames that could not be parsed)."""
    usable: list[tuple[Path, dict]] = []
    skipped: list[str] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        truth = parse_ground_truth(path.name, cfg)
        if truth is None:
            skipped.append(path.name)
        else:
            usable.append((path, truth))
    return usable, skipped


def evaluate(
    folder: Path,
    cfg: dict,
    method: str,
    debug_root: Path | None = None,
    weights: str | None = None,
    device: str | None = None,
) -> dict:
    usable, skipped = collect_images(folder, cfg)
    rows = []
    for path, truth in usable:
        debug_dir = (debug_root / method / path.stem) if debug_root else None
        result = count_image(
            path, cfg, debug_dir=debug_dir, method=method, weights=weights, device=device
        )
        error = result["count"] - truth["count"]
        rows.append(
            {
                "file": path.name,
                "truth": truth["count"],
                "diameter": truth["diameter"],
                "condition": truth["condition"],
                "predicted": result["count"],
                "error": error,
                "error_pct": 100.0 * abs(error) / truth["count"] if truth["count"] else 0.0,
                "elapsed_ms": result["elapsed_ms"],
            }
        )
    return {"method": method, "rows": rows, "skipped": skipped, "summary": summarize(rows, cfg)}


def summarize(rows: list[dict], cfg: dict) -> dict:
    if not rows:
        return {"images": 0}
    tolerance = int(cfg["eval"]["within_tolerance"])
    count = len(rows)
    within = sum(1 for row in rows if abs(row["error"]) <= tolerance)
    return {
        "images": count,
        "mae": sum(abs(row["error"]) for row in rows) / count,
        "mape": sum(row["error_pct"] for row in rows) / count,
        "within_tolerance": tolerance,
        "within_rate": 100.0 * within / count,
        "mean_elapsed_ms": sum(row["elapsed_ms"] for row in rows) / count,
    }


def print_report(report: dict) -> None:
    rows = report["rows"]
    name_width = max([len(row["file"]) for row in rows], default=8)
    header = f"{'file'.ljust(name_width)}  {'truth':>6} {'pred':>6} {'err':>6} {'err%':>7} {'ms':>6}"
    print(f"\nmethod: {report['method']}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['file'].ljust(name_width)}  {row['truth']:>6} {row['predicted']:>6} "
            f"{row['error']:>+6} {row['error_pct']:>6.1f}% {row['elapsed_ms']:>6}"
        )

    summary = report["summary"]
    print("-" * len(header))
    if not summary["images"]:
        print("no images with parseable ground truth")
    else:
        print(f"images            : {summary['images']}")
        print(f"MAE               : {summary['mae']:.2f} pieces")
        print(f"MAPE              : {summary['mape']:.2f} %   <- headline metric")
        print(
            f"within +/-{summary['within_tolerance']}        : "
            f"{summary['within_rate']:.1f} %"
        )
        print(f"mean elapsed      : {summary['mean_elapsed_ms']:.0f} ms")

    if report["skipped"]:
        print(f"\nskipped (filename does not match SPEC 7 pattern): {len(report['skipped'])}")
        for name in report["skipped"]:
            print(f"  - {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate counting accuracy on a folder.")
    parser.add_argument("folder", help="folder of photos named per SPEC 7")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--methods",
        default="classical",
        help="comma-separated engines to compare, e.g. classical,ml",
    )
    parser.add_argument("--debug-dir", default=None, help="write per-image debug stages here")
    parser.add_argument("--weights", default=None, help="ml engine: path to trained weights")
    parser.add_argument("--device", default=None, help="ml engine: cpu | mps | 0")
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    debug_root = Path(args.debug_dir) if args.debug_dir else None
    reports = [
        evaluate(folder, cfg, method.strip(), debug_root, args.weights, args.device)
        for method in args.methods.split(",")
        if method.strip()
    ]

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0

    for report in reports:
        print_report(report)

    if len(reports) > 1:
        print("\ncomparison (MAPE):")
        for report in reports:
            summary = report["summary"]
            mape = f"{summary['mape']:.2f} %" if summary["images"] else "n/a"
            print(f"  {report['method']:>10}: {mape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
