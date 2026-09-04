#!/usr/bin/env python3
"""Generate synthetic rebar-stack images with an exactly known count.

Real photos do not exist yet, so this is the first correctness check for the
pipeline: gray circles on a light background, hex-packed so a chosen fraction of
them touch or overlap - which is precisely what the watershed step must undo.

    python tests/make_synthetic.py --out out/synthetic

Files are named per SPEC 7 so eval.py can score the folder directly.
Output never goes into data/eval/ - that folder is reserved for real photos.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Fixed scene defaults; every variant below overrides what it needs.
BACKGROUND_LEVEL = 238
CIRCLE_LEVEL_RANGE = (95, 150)
RIM_LEVEL = 60


def hex_positions(
    width: int, height: int, radius: float, spacing_ratio: float, margin: int
) -> list[tuple[float, float]]:
    """Hexagonal packing, the way bars settle in a bundle."""
    step = 2.0 * radius * spacing_ratio
    row_height = step * math.sqrt(3.0) / 2.0
    positions: list[tuple[float, float]] = []
    row = 0
    y = margin + radius
    while y + radius + margin <= height:
        x = margin + radius + (step / 2.0 if row % 2 else 0.0)
        while x + radius + margin <= width:
            positions.append((x, y))
            x += step
        y += row_height
        row += 1
    return positions


def radius_for_count(width: int, height: int, count: int, spacing_ratio: float) -> float:
    """Pick the bar radius that makes `count` circles fill the frame.

    A real photo is the end of a stack filling the viewfinder, not a handful of
    circles in the corner, so the radius follows from the count - not the reverse.
    """
    radius = math.sqrt(width * height / (0.866 * max(count, 1))) / (2.0 * spacing_ratio)
    for _ in range(40):
        margin = int(radius)
        if len(hex_positions(width, height, radius, spacing_ratio, margin)) >= count:
            return radius
        radius *= 0.97
    return radius


@dataclass
class Scene:
    """A generated stack plus the exact truth behind it."""

    image: np.ndarray
    positions: list[tuple[float, float]]
    radius: float

    @property
    def count(self) -> int:
        return len(self.positions)

    def yolo_lines(self, class_index: int = 0) -> list[str]:
        """Ground-truth boxes in YOLO format - the labels a human would have drawn."""
        height, width = self.image.shape[:2]
        side = 2.0 * self.radius
        lines = []
        for x, y in self.positions:
            lines.append(
                f"{class_index} {x / width:.6f} {y / height:.6f} "
                f"{side / width:.6f} {side / height:.6f}"
            )
        return lines


def render(
    count: int,
    spacing_ratio: float = 1.02,
    jitter_ratio: float = 0.05,
    rim_ratio: float = 0.10,
    noise_sigma: float = 3.0,
    gradient: float = 0.0,
    drop_fraction: float = 0.0,
    size: tuple[int, int] = (1200, 900),
    seed: int = 0,
) -> Scene:
    """Draw `count` circles packed across the whole frame.

    Jitter and rim are expressed relative to the radius so every variant stays
    comparable at any bar size.
    """
    rng = np.random.default_rng(seed)
    width, height = size
    radius = radius_for_count(width, height, count, spacing_ratio)
    margin = int(radius)
    jitter = radius * jitter_ratio

    positions = hex_positions(width, height, radius, spacing_ratio, margin)
    if drop_fraction > 0 and positions:
        keep = rng.random(len(positions)) >= drop_fraction
        positions = [p for p, k in zip(positions, keep) if k]
    positions = positions[:count]

    canvas = np.full((height, width), BACKGROUND_LEVEL, dtype=np.float32)
    if gradient:
        ramp = np.linspace(-gradient, gradient, width, dtype=np.float32)
        canvas += ramp[None, :]

    jittered = [
        (x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)) for x, y in positions
    ]

    # Fill first, then stroke every rim, so touching circles keep a visible border
    # (in a real stack that border is the shadow between two bars).
    for x, y in jittered:
        level = float(rng.integers(*CIRCLE_LEVEL_RANGE))
        cv2.circle(canvas, (int(round(x)), int(round(y))), int(round(radius)), level, -1, cv2.LINE_AA)
    rim_thickness = int(round(radius * rim_ratio))
    if rim_ratio > 0 and rim_thickness > 0:
        for x, y in jittered:
            cv2.circle(
                canvas,
                (int(round(x)), int(round(y))),
                int(round(radius)),
                RIM_LEVEL,
                rim_thickness,
                cv2.LINE_AA,
            )

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    if noise_sigma > 0:
        canvas += rng.normal(0.0, noise_sigma, canvas.shape).astype(np.float32)

    image = np.clip(canvas, 0, 255).astype(np.uint8)
    return Scene(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), jittered, radius)


# name -> render kwargs. Each variant isolates one thing that can break the pipeline.
VARIANTS = {
    "seyrek": dict(count=120, spacing_ratio=1.25),
    "degen": dict(count=180, spacing_ratio=1.02),
    "sikisik": dict(count=240, spacing_ratio=0.98),
    "rimsiz": dict(count=180, spacing_ratio=1.02, rim_ratio=0.0),
    "gurultulu": dict(count=180, spacing_ratio=1.02, noise_sigma=9.0),
    "gradyan": dict(count=180, spacing_ratio=1.02, gradient=55.0),
    "duzensiz": dict(count=300, spacing_ratio=1.02, drop_fraction=0.22, jitter_ratio=0.12),
    "kalabalik": dict(count=420, spacing_ratio=1.02),
}


def write_image(path: Path, image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    buffer.tofile(str(path))


def random_scene(rng: np.random.Generator, seed: int) -> tuple[str, Scene]:
    """A randomly perturbed stack, for generating a training set rather than a test set."""
    name = str(rng.choice(list(VARIANTS)))
    kwargs = dict(VARIANTS[name])
    kwargs["count"] = int(rng.integers(90, 460))
    # Keep the variant's own packing as the centre of the perturbation, so the
    # training set still covers sparse-through-overlapping rather than one average.
    kwargs["spacing_ratio"] = float(kwargs.get("spacing_ratio", 1.02)) * float(
        rng.uniform(0.98, 1.08)
    )
    kwargs["noise_sigma"] = float(rng.uniform(1.0, 11.0))
    kwargs["gradient"] = float(rng.uniform(0.0, 60.0))
    kwargs["jitter_ratio"] = float(rng.uniform(0.02, 0.14))
    kwargs["rim_ratio"] = float(rng.uniform(0.0, 0.16))
    width = int(rng.choice([1000, 1200, 1400, 1600]))
    kwargs["size"] = (width, int(width * rng.uniform(0.68, 0.82)))
    return name, render(seed=seed, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic rebar-stack test images.")
    parser.add_argument("--out", default="out/synthetic", help="output folder")
    parser.add_argument("--labels", default=None, help="also write YOLO ground-truth labels here")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        help="generate N randomly perturbed scenes instead of the fixed variant list",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = Path(args.labels) if args.labels else None
    if label_dir:
        label_dir.mkdir(parents=True, exist_ok=True)

    if args.random:
        rng = np.random.default_rng(args.seed)
        scenes = [random_scene(rng, args.seed + i) for i in range(args.random)]
    else:
        scenes = [(name, render(seed=args.seed + i, **kw)) for i, (name, kw) in enumerate(VARIANTS.items())]

    for index, (name, scene) in enumerate(scenes):
        # SPEC 7 filename: YYYYMMDD_hhmm_<truecount>_<diameter>_<condition>
        stem = f"20260101_{12 * 100 + index:04d}_{scene.count}_12_sentetik-{name}"
        write_image(out_dir / f"{stem}.png", scene.image)
        if label_dir:
            (label_dir / f"{stem}.txt").write_text(
                "\n".join(scene.yolo_lines()) + "\n", encoding="utf-8"
            )
        print(f"{stem}.png  ({scene.count} circles, r={scene.radius:.1f}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
