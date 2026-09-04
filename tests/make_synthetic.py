#!/usr/bin/env python3
"""Generate synthetic rebar-stack images with an exactly known count.

Real photos do not exist yet, so this is the only correctness check the pipeline
has. The variants below walk SPEC 7's diversity matrix as far as a renderer can:
packing density, per-bar brightness (rust next to shiny), noise, lighting gradient,
viewing angle, background texture, glare, and several bundles in one frame.

    python tests/make_synthetic.py --out out/synthetic                      # fixed set
    python tests/make_synthetic.py --out out/train/images \
        --labels out/train/labels --random 80 --seed 500                    # training set

Files are named per SPEC 7 so eval.py can score the folder directly. Output never
goes into data/eval/ - that folder is reserved for real photos.
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


@dataclass
class Scene:
    """A generated stack plus the exact truth behind it."""

    image: np.ndarray
    circles: list[tuple[float, float, float]]  # x, y, radius - radius varies under tilt

    @property
    def count(self) -> int:
        return len(self.circles)

    @property
    def positions(self) -> list[tuple[float, float]]:
        return [(x, y) for x, y, _ in self.circles]

    @property
    def radius(self) -> float:
        """Median bar radius; the per-circle value is what the labels actually use."""
        return float(np.median([r for _, _, r in self.circles])) if self.circles else 0.0

    def yolo_lines(self, class_index: int = 0) -> list[str]:
        """Ground-truth boxes in YOLO format - the labels a human would have drawn."""
        height, width = self.image.shape[:2]
        lines = []
        for x, y, r in self.circles:
            lines.append(
                f"{class_index} {x / width:.6f} {y / height:.6f} "
                f"{2.0 * r / width:.6f} {2.0 * r / height:.6f}"
            )
        return lines


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


def split_into_bundles(
    positions: list[tuple[float, float]], bundles: int, height: int, gap_ratio: float
) -> list[tuple[float, float]]:
    """Drop horizontal strips so the frame holds several bundles side by side.

    SPEC 7 lists "birden fazla demet bir arada" as its own condition: the counter
    must not treat the gap between two bundles as the edge of the photo.
    """
    if bundles <= 1:
        return positions
    band = height / bundles
    gap = band * gap_ratio
    kept = []
    for x, y in positions:
        offset = y % band
        if gap / 2 <= offset <= band - gap / 2:
            kept.append((x, y))
    return kept


def background_texture(
    shape: tuple[int, int], amount: float, rng: np.random.Generator
) -> np.ndarray:
    """Concrete / truck-bed mottling, not a studio backdrop."""
    if amount <= 0:
        return np.zeros(shape, dtype=np.float32)
    height, width = shape
    coarse = rng.normal(0.0, 1.0, (max(2, height // 24), max(2, width // 24))).astype(np.float32)
    coarse = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0.0, 0.35, shape).astype(np.float32)
    return (coarse + fine) * amount


def add_glare(canvas: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """A blown-out patch where the sun or the flash hits the cut faces."""
    if strength <= 0:
        return canvas
    height, width = canvas.shape
    cx = rng.uniform(0.15, 0.85) * width
    cy = rng.uniform(0.15, 0.85) * height
    sigma = rng.uniform(0.15, 0.30) * min(width, height)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    falloff = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma)))
    return canvas + strength * falloff


def tilt_homography(width: int, height: int, degrees: float, rng: np.random.Generator):
    """Homography for looking at the stack end off-axis (SPEC 7: 0, ~10, ~20 degrees)."""
    if degrees <= 0:
        return None
    shrink = math.sin(math.radians(degrees)) * 0.5
    horizontal = rng.random() < 0.5
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    if horizontal:
        inset = shrink * height
        target = np.float32(
            [[0, inset], [width, 0], [width, height], [0, height - inset]]
        )
    else:
        inset = shrink * width
        target = np.float32(
            [[inset, 0], [width - inset, 0], [width, height], [0, height]]
        )
    return cv2.getPerspectiveTransform(source, target)


def apply_tilt(
    canvas: np.ndarray,
    circles: list[tuple[float, float, float]],
    matrix: np.ndarray,
    fill: float,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """Warp the image and the truth together, then drop what left the frame."""
    height, width = canvas.shape
    warped = cv2.warpPerspective(
        canvas, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float(fill),
    )

    centres = np.float32([[[x, y]] for x, y, _ in circles])
    right = np.float32([[[x + r, y]] for x, y, r in circles])
    down = np.float32([[[x, y + r]] for x, y, r in circles])
    centres = cv2.perspectiveTransform(centres, matrix).reshape(-1, 2)
    right = cv2.perspectiveTransform(right, matrix).reshape(-1, 2)
    down = cv2.perspectiveTransform(down, matrix).reshape(-1, 2)

    moved: list[tuple[float, float, float]] = []
    for (cx, cy), (rx, ry), (dx, dy) in zip(centres, right, down):
        # Perspective squashes one axis more than the other; average the two.
        radius = (math.hypot(rx - cx, ry - cy) + math.hypot(dx - cx, dy - cy)) / 2.0
        if 0 <= cx < width and 0 <= cy < height and radius > 2:
            moved.append((float(cx), float(cy), float(radius)))
    return warped, moved


def render(
    count: int,
    spacing_ratio: float = 1.02,
    jitter_ratio: float = 0.05,
    rim_ratio: float = 0.10,
    noise_sigma: float = 3.0,
    gradient: float = 0.0,
    drop_fraction: float = 0.0,
    bundles: int = 1,
    bundle_gap_ratio: float = 0.22,
    tilt_degrees: float = 0.0,
    background_amount: float = 0.0,
    glare: float = 0.0,
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
    positions = split_into_bundles(positions, bundles, height, bundle_gap_ratio)
    if drop_fraction > 0 and positions:
        keep = rng.random(len(positions)) >= drop_fraction
        positions = [p for p, k in zip(positions, keep) if k]
    positions = positions[:count]

    canvas = np.full((height, width), float(BACKGROUND_LEVEL), dtype=np.float32)
    canvas += background_texture((height, width), background_amount, rng)
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

    circles = [(x, y, radius) for x, y in jittered]
    matrix = tilt_homography(width, height, tilt_degrees, rng)
    if matrix is not None:
        canvas, circles = apply_tilt(canvas, circles, matrix, BACKGROUND_LEVEL)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    canvas = add_glare(canvas, glare, rng)
    if noise_sigma > 0:
        canvas += rng.normal(0.0, noise_sigma, canvas.shape).astype(np.float32)

    image = np.clip(canvas, 0, 255).astype(np.uint8)
    return Scene(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), circles)


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
    # SPEC 7 diversity matrix beyond packing and light:
    "egik10": dict(count=240, spacing_ratio=1.02, tilt_degrees=10.0),
    "egik20": dict(count=240, spacing_ratio=1.02, tilt_degrees=20.0),
    "zeminli": dict(count=240, spacing_ratio=1.02, background_amount=14.0, drop_fraction=0.10),
    "parlama": dict(count=240, spacing_ratio=1.02, glare=70.0),
    "cift-demet": dict(count=300, spacing_ratio=1.02, bundles=2),
    "uc-demet": dict(count=300, spacing_ratio=1.02, bundles=3, bundle_gap_ratio=0.26),
    "zor": dict(
        count=360, spacing_ratio=1.00, tilt_degrees=14.0, background_amount=10.0,
        glare=45.0, noise_sigma=7.0, gradient=35.0, rim_ratio=0.06, jitter_ratio=0.09,
    ),
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
    kwargs["tilt_degrees"] = float(rng.choice([0.0, 0.0, 6.0, 10.0, 14.0, 20.0]))
    kwargs["background_amount"] = float(rng.uniform(0.0, 16.0))
    kwargs["glare"] = float(rng.choice([0.0, 0.0, 30.0, 55.0, 80.0]))
    kwargs["bundles"] = int(rng.choice([1, 1, 1, 2, 3]))
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
        scenes = [
            (name, render(seed=args.seed + i, **kw))
            for i, (name, kw) in enumerate(VARIANTS.items())
        ]

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
