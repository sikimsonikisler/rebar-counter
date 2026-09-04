#!/usr/bin/env python3
"""Rebar cross-section counter - CLI entry point (SPEC 6, phase 1).

    python python/count.py <image_path> [--config config.yaml] [--debug <out_dir>] [--json]

Prints, with --json:
    {"count": 412, "points": [[x, y, r]], "method": "classical", "elapsed_ms": 340}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import detect, draw, preprocess, segment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
METHODS = ("classical", "hough", "ml")


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_method(cfg: dict, requested: str | None) -> str:
    """CLI flag wins; otherwise config decides whether the Hough branch is active."""
    if requested:
        return requested
    return "hough" if cfg["hough"]["enabled"] else "classical"


def _count_classical(gray, cfg: dict, debug) -> list[detect.Detection]:
    """Two-pass counting with automatic scale estimation (SPEC 4).

    Both passes write debug images to the same filenames, so what ends up on disk is
    always the pass that actually produced the count - the extra PNG writes only
    happen under --debug and are excluded from the reported time.
    """
    mask = segment.clean_mask(segment.binarize(gray, cfg, debug), cfg, debug)
    separated = segment.split_touching(gray, mask, cfg, debug=debug, radius_hint=None)
    candidates = detect.filter_by_shape(detect.extract_candidates(separated, cfg), cfg)
    radius = detect.estimate_radius(candidates, cfg)

    # Pass 2: re-binarize and re-seed at the estimated bar size, then re-filter.
    if cfg["segment"]["two_pass"]["enabled"] and radius is not None:
        mask = segment.clean_mask(
            segment.binarize(gray, cfg, debug, radius_hint=radius), cfg, debug
        )
        separated = segment.split_touching(gray, mask, cfg, debug=debug, radius_hint=radius)
        candidates = detect.filter_by_shape(detect.extract_candidates(separated, cfg), cfg)
        candidates = detect.filter_by_relative_area(candidates, cfg, radius)
        refined = detect.estimate_radius(candidates, cfg)
        radius = refined if refined is not None else radius

    if radius is not None:
        candidates = detect.filter_radius_outliers(candidates, cfg, radius)

    if debug:
        debug.save("components", draw.colorize_components(separated))
    return detect.merge_close(candidates, cfg)


def _count_hough(gray, cfg: dict, debug) -> list[detect.Detection]:
    """Alternative branch: classical pass only to estimate scale, then HoughCircles."""
    mask = segment.clean_mask(segment.binarize(gray, cfg, debug), cfg, debug)
    separated = segment.split_touching(gray, mask, cfg, debug=debug, radius_hint=None)
    seed_candidates = detect.filter_by_shape(detect.extract_candidates(separated, cfg), cfg)
    radius = detect.estimate_radius(seed_candidates, cfg)
    if debug:
        debug.save("components", draw.colorize_components(separated))
    return detect.merge_close(detect.detect_hough(gray, cfg, radius), cfg)


def count_image(
    image_path: str | Path,
    cfg: dict,
    debug_dir: str | Path | None = None,
    method: str | None = None,
    weights: str | Path | None = None,
    device: str | None = None,
) -> dict:
    """Count one photo. Returns the JSON contract dict; coordinates are original-image px."""
    method = resolve_method(cfg, method)
    if method == "ml":
        # Imported here on purpose: loading torch costs seconds, and the classical
        # engine must stay usable on a machine that has no ML stack installed at all.
        from ml.predict import count_image_ml

        return count_image_ml(image_path, cfg, weights=weights, debug_dir=debug_dir, device=device)

    debug = draw.DebugWriter(debug_dir)
    started = time.perf_counter()

    original = preprocess.load_image(image_path)
    working, scale_back = preprocess.resize_for_processing(original, cfg)
    gray = preprocess.run(working, cfg, debug)

    if method == "hough":
        detections = _count_hough(gray, cfg, debug)
    else:
        detections = _count_classical(gray, cfg, debug)

    detections = detect.scale_to_original(detections, scale_back)
    # Debug I/O is not part of the measured pipeline cost.
    elapsed_ms = (time.perf_counter() - started - debug.io_seconds) * 1000.0

    if debug:
        debug.save("result", draw.render_result(original, detections, cfg))

    return {
        "count": len(detections),
        "points": [d.as_point() for d in detections],
        "method": method,
        "elapsed_ms": round(elapsed_ms),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count rebar cross-sections in a photo.")
    parser.add_argument("image_path", help="photo of the end of a rebar stack")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--debug", default=None, metavar="OUT_DIR", help="write per-stage images")
    parser.add_argument("--json", action="store_true", help="print the JSON contract on stdout")
    parser.add_argument("--method", choices=METHODS, default=None, help="counting engine")
    parser.add_argument("--weights", default=None, help="ml engine: path to trained weights")
    parser.add_argument("--device", default=None, help="ml engine: cpu | mps | 0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    result = count_image(
        args.image_path,
        cfg,
        debug_dir=args.debug,
        method=args.method,
        weights=args.weights,
        device=args.device,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"count={result['count']}  method={result['method']}  elapsed={result['elapsed_ms']}ms")
        if args.debug:
            print(f"debug images: {args.debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
