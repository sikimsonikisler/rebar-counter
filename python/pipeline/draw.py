"""Step 4 - debug images and the final marked-up result.

The debug images are the only way to tune this pipeline, so they are a first-class
part of it, not an afterthought.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# Fixed order so the files sort the way the pipeline runs: 01_gray.png ... 09_result.png
STAGES = [
    "gray",
    "clahe",
    "bilateral",
    "binary",
    "morph",
    "distance",
    "markers",
    "components",
    "result",
]

# The ML engine has no threshold to eyeball; what matters is where the tiles fell.
ML_STAGES = ["tiles", "result"]


class DebugWriter:
    """Writes one PNG per pipeline stage into out_dir. A no-op when out_dir is None."""

    def __init__(self, out_dir: str | Path | None, stages: list[str] | None = None):
        self.out_dir = Path(out_dir) if out_dir else None
        self.stages = stages or STAGES
        # Time spent writing PNGs, so callers can keep it out of their timings.
        self.io_seconds = 0.0
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def __bool__(self) -> bool:
        return self.out_dir is not None

    def save(self, stage: str, image: np.ndarray) -> None:
        if not self.out_dir:
            return
        if stage not in self.stages:
            raise ValueError(f"unknown debug stage: {stage}")
        index = self.stages.index(stage) + 1
        started = time.perf_counter()
        path = self.out_dir / f"{index:02d}_{stage}.png"
        # imencode + tofile so non-ASCII output paths work everywhere.
        ok, buffer = cv2.imencode(".png", image)
        if ok:
            buffer.tofile(str(path))
        self.io_seconds += time.perf_counter() - started


def colorize_components(separated: np.ndarray) -> np.ndarray:
    """Random colour per blob: the fastest way to see whether watershed split correctly."""
    count, labels = cv2.connectedComponents(separated)
    if count <= 1:
        return cv2.cvtColor(separated, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(seed=0)  # fixed seed: same picture every run
    palette = rng.integers(60, 255, size=(count, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    return palette[labels]


def draw_tiles(bgr: np.ndarray, tiles, cfg: dict) -> np.ndarray:
    """Show where the tile grid fell - the first thing to check when the ML count is off."""
    canvas = bgr.copy()
    colour = tuple(int(v) for v in cfg["draw"]["circle_color_bgr"])
    for window in tiles:
        cv2.rectangle(canvas, (window.left, window.top), (window.right - 1, window.bottom - 1), colour, 2)
    return canvas


def render_result(bgr: np.ndarray, detections, cfg: dict) -> np.ndarray:
    """The shareable picture: every detection marked, big count banner on top."""
    draw_cfg = cfg["draw"]
    canvas = bgr.copy()
    height = canvas.shape[0]

    dot_color = tuple(int(v) for v in draw_cfg["dot_color_bgr"])
    circle_color = tuple(int(v) for v in draw_cfg["circle_color_bgr"])
    dot_ratio = float(draw_cfg["dot_radius_ratio"])
    dot_minimum = int(draw_cfg["dot_min_radius_px"])

    for detection in detections:
        center = (int(round(detection.x)), int(round(detection.y)))
        if draw_cfg["draw_circles"]:
            cv2.circle(
                canvas,
                center,
                max(1, int(round(detection.r))),
                circle_color,
                int(draw_cfg["circle_thickness"]),
                lineType=cv2.LINE_AA,
            )
        cv2.circle(
            canvas,
            center,
            max(dot_minimum, int(round(detection.r * dot_ratio))),
            dot_color,
            -1,
            lineType=cv2.LINE_AA,
        )

    banner_height = max(24, int(height * float(draw_cfg["banner_height_ratio"])))
    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], banner_height),
        tuple(int(v) for v in draw_cfg["banner_color_bgr"]),
        -1,
    )
    font_scale = max(0.4, height * float(draw_cfg["font_scale_ratio"]))
    thickness = max(1, int(round(height * float(draw_cfg["font_thickness_ratio"]))))
    text = f"{len(detections)} adet"
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.putText(
        canvas,
        text,
        (max(8, (canvas.shape[1] - text_width) // 2), (banner_height + text_height) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        tuple(int(v) for v in draw_cfg["text_color_bgr"]),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return canvas
