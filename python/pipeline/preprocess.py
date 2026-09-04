"""Step 1 - get from a raw photo to a clean grayscale image.

Every function here does exactly one thing and takes its numbers from config.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_INTERPOLATIONS = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


def load_image(path: str | Path) -> np.ndarray:
    """Read a BGR image. Uses imdecode so non-ASCII paths work on every platform."""
    path = Path(path)
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError(f"empty or unreadable file: {path}")
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"not a decodable image: {path}")
    return image


def resize_for_processing(bgr: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    """Downscale so the long edge matches config. Returns (image, scale_back_factor).

    scale_back_factor converts a processing-image coordinate to an original-image one.
    """
    max_long_edge = int(cfg["io"]["max_long_edge"])
    height, width = bgr.shape[:2]
    long_edge = max(height, width)
    if max_long_edge <= 0 or long_edge <= max_long_edge:
        return bgr, 1.0

    ratio = max_long_edge / float(long_edge)
    interpolation = _INTERPOLATIONS[cfg["io"]["resize_interpolation"]]
    resized = cv2.resize(
        bgr, (int(round(width * ratio)), int(round(height * ratio))), interpolation=interpolation
    )
    return resized, 1.0 / ratio


def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """Local contrast equalization: rescues detail in both shadow and glare."""
    clahe_cfg = cfg["preprocess"]["clahe"]
    if not clahe_cfg["enabled"]:
        return gray
    clahe = cv2.createCLAHE(
        clipLimit=float(clahe_cfg["clip_limit"]),
        tileGridSize=tuple(int(v) for v in clahe_cfg["tile_grid"]),
    )
    return clahe.apply(gray)


def denoise(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """Edge-preserving smoothing: kills rust speckle, keeps the gap lines sharp."""
    bilateral_cfg = cfg["preprocess"]["bilateral"]
    if not bilateral_cfg["enabled"]:
        return gray
    return cv2.bilateralFilter(
        gray,
        d=int(bilateral_cfg["diameter"]),
        sigmaColor=float(bilateral_cfg["sigma_color"]),
        sigmaSpace=float(bilateral_cfg["sigma_space"]),
    )


def run(bgr: np.ndarray, cfg: dict, debug=None) -> np.ndarray:
    """Full preprocessing chain on an already-resized BGR image."""
    gray = to_grayscale(bgr)
    if debug:
        debug.save("gray", gray)

    equalized = apply_clahe(gray, cfg)
    if debug:
        debug.save("clahe", equalized)

    smoothed = denoise(equalized, cfg)
    if debug:
        debug.save("bilateral", smoothed)

    return smoothed
