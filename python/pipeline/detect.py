"""Step 3 - turn separated blobs into filtered, deduplicated detections."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """One cross-section. x, y and r are in processing-image pixels."""

    x: float
    y: float
    r: float
    area: float
    circularity: float
    fill_ratio: float

    def as_point(self) -> list[float]:
        return [round(self.x, 1), round(self.y, 1), round(self.r, 1)]


def extract_candidates(separated: np.ndarray, cfg: dict) -> list[Detection]:
    """One Detection per connected blob, with the shape metrics needed for filtering.

    Blobs outside the absolute area guards are dropped here so the (more expensive)
    contour work never runs on obvious noise.
    """
    absolute_cfg = cfg["detect"]["absolute"]
    min_area = float(absolute_cfg["min_area_px"])
    max_area = float(absolute_cfg["max_area_px"])

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(separated, connectivity=8)
    detections: list[Detection] = []

    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        left = int(stats[index, cv2.CC_STAT_LEFT])
        top = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])

        blob = (labels[top : top + height, left : left + width] == index).astype(np.uint8) * 255
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
        _, enclosing_radius = cv2.minEnclosingCircle(contour)
        fill_ratio = area / (math.pi * enclosing_radius**2) if enclosing_radius > 0 else 0.0

        detections.append(
            Detection(
                x=float(centroids[index][0]),
                y=float(centroids[index][1]),
                r=math.sqrt(area / math.pi),  # equivalent-area radius, stable on jagged cuts
                area=area,
                circularity=circularity,
                fill_ratio=min(fill_ratio, 1.0),
            )
        )
    return detections


def filter_by_shape(detections: list[Detection], cfg: dict) -> list[Detection]:
    """Drop blobs that are not round enough to be a bar end."""
    shape_cfg = cfg["detect"]["shape"]
    min_circularity = float(shape_cfg["min_circularity"])
    min_fill_ratio = float(shape_cfg["min_fill_ratio"])
    return [
        d
        for d in detections
        if d.circularity >= min_circularity and d.fill_ratio >= min_fill_ratio
    ]


def estimate_radius(detections: list[Detection], cfg: dict) -> float | None:
    """The automatic scale estimate (SPEC 4): how big is one bar in this photo?

    The default is the area-weighted mode rather than a plain median. Bars carry
    almost all of the foreground area while noise fragments are numerous but tiny,
    so the mode survives a messy first pass where a median gets dragged down to the
    size of the debris. Returns None when there is not enough to estimate from.
    """
    estimate_cfg = cfg["detect"]["radius_estimate"]
    if len(detections) < int(estimate_cfg["min_detections"]):
        return None

    radii = np.array([d.r for d in detections], dtype=np.float64)
    if estimate_cfg["method"] == "median":
        return float(np.median(radii))

    areas = np.array([d.area for d in detections], dtype=np.float64)
    histogram, edges = np.histogram(
        radii, bins=int(estimate_cfg["histogram_bins"]), weights=areas
    )
    peak = int(np.argmax(histogram))
    in_peak = radii[(radii >= edges[peak]) & (radii <= edges[peak + 1])]
    return float(np.median(in_peak)) if in_peak.size else float(np.median(radii))


def filter_by_relative_area(
    detections: list[Detection], cfg: dict, radius: float
) -> list[Detection]:
    """Area guards expressed against the estimated bar size instead of raw pixels."""
    relative_cfg = cfg["detect"]["relative"]
    nominal_area = math.pi * radius * radius
    min_area = float(relative_cfg["min_area_ratio"]) * nominal_area
    max_area = float(relative_cfg["max_area_ratio"]) * nominal_area
    return [d for d in detections if min_area <= d.area <= max_area]


def filter_radius_outliers(
    detections: list[Detection], cfg: dict, radius: float
) -> list[Detection]:
    """SPEC 4: keep only candidates within +/-40% of the median radius."""
    outlier_cfg = cfg["detect"]["radius_outlier"]
    if not outlier_cfg["enabled"]:
        return detections
    low = float(outlier_cfg["low_ratio"]) * radius
    high = float(outlier_cfg["high_ratio"]) * radius
    return [d for d in detections if low <= d.r <= high]


def merge_close(detections: list[Detection], cfg: dict) -> list[Detection]:
    """Collapse duplicates: watershed sometimes shaves a sliver off one bar."""
    ratio = float(cfg["detect"]["merge_distance_ratio"])
    if ratio <= 0 or not detections:
        return detections

    ordered = sorted(detections, key=lambda d: d.area, reverse=True)
    kept: list[Detection] = []
    kept_xy = np.zeros((0, 2), dtype=np.float64)
    kept_r = np.zeros((0,), dtype=np.float64)

    for candidate in ordered:
        if len(kept):
            distances = np.hypot(kept_xy[:, 0] - candidate.x, kept_xy[:, 1] - candidate.y)
            thresholds = ratio * np.maximum(kept_r, candidate.r)
            if np.any(distances < thresholds):
                continue
        kept.append(candidate)
        kept_xy = np.vstack([kept_xy, [candidate.x, candidate.y]])
        kept_r = np.append(kept_r, candidate.r)

    return kept


def detect_hough(gray: np.ndarray, cfg: dict, radius_hint: float | None = None) -> list[Detection]:
    """Alternative branch kept for comparison; misses bars that touch."""
    hough_cfg = cfg["hough"]
    radius = float(radius_hint or hough_cfg["fallback_radius_px"])
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(hough_cfg["dp"]),
        minDist=max(1.0, radius * float(hough_cfg["min_dist_radius_ratio"])),
        param1=float(hough_cfg["param1"]),
        param2=float(hough_cfg["param2"]),
        minRadius=int(radius * float(hough_cfg["min_radius_ratio"])),
        maxRadius=int(radius * float(hough_cfg["max_radius_ratio"])),
    )
    if circles is None:
        return []
    return [
        Detection(
            x=float(x),
            y=float(y),
            r=float(r),
            area=math.pi * float(r) ** 2,
            circularity=1.0,
            fill_ratio=1.0,
        )
        for x, y, r in np.round(circles[0]).astype(np.float64)
    ]


def scale_to_original(detections: list[Detection], factor: float) -> list[Detection]:
    """Map processing-image coordinates back onto the original photo."""
    if factor == 1.0:
        return detections
    return [
        Detection(
            x=d.x * factor,
            y=d.y * factor,
            r=d.r * factor,
            area=d.area * factor * factor,
            circularity=d.circularity,
            fill_ratio=d.fill_ratio,
        )
        for d in detections
    ]
