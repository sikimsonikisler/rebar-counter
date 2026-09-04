"""Phase 3 - overlapping tiling and merge-back (SPEC 4).

A photo holds 300-600 cross-sections. Resized to 640x640 each one is a handful of
pixels and the detector simply cannot see it, so the image is cut into overlapping
tiles, detected on tile by tile, and the boxes are mapped back to global coordinates
and merged with NMS. That is the SAHI idea; it is written out here rather than pulled
in as a dependency, because the same logic has to be reimplemented in Swift for the
phone and a short readable version is worth more than a library call.

The module knows nothing about YOLO: `run_tiled` takes a callable, so it can be
tested against a fake detector on a synthetic image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Tile:
    """A crop window in image coordinates, plus which of its sides touch the image edge."""

    left: int
    top: int
    right: int
    bottom: int
    touches_left: bool
    touches_top: bool
    touches_right: bool
    touches_bottom: bool

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def _starts(length: int, tile_size: int, step: int, minimum: int) -> list[int]:
    """Tile origins along one axis. The last tile is pulled back to end on the edge."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, step))
    if not positions or positions[-1] + tile_size < length:
        positions.append(length - tile_size)
    while len(positions) < minimum:
        positions.append(positions[-1])
    return sorted(set(positions))


def plan_tiles(width: int, height: int, cfg: dict) -> list[Tile]:
    """Cut a width x height image into overlapping tiles."""
    tiling_cfg = cfg["ml"]["tiling"]
    tile_size = int(tiling_cfg["tile_size"])
    overlap = float(tiling_cfg["overlap"])
    if not 0.0 <= overlap < 1.0:
        raise ValueError("ml.tiling.overlap must be in [0, 1)")
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    minimum = int(tiling_cfg["min_tiles"])

    tiles = []
    for top in _starts(height, tile_size, step, minimum):
        for left in _starts(width, tile_size, step, minimum):
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            tiles.append(
                Tile(
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    touches_left=left == 0,
                    touches_top=top == 0,
                    touches_right=right >= width,
                    touches_bottom=bottom >= height,
                )
            )
    return tiles


def crop(image: np.ndarray, tile: Tile) -> np.ndarray:
    return image[tile.top : tile.bottom, tile.left : tile.right]


def _edge_margin(cfg: dict) -> float:
    """How far from a tile edge a detection is still trusted.

    Must not exceed half the overlap: a bar rejected near one tile's edge has to sit
    comfortably inside its neighbour, otherwise tiling would lose bars instead of
    duplicating them - and a duplicate is cheap (NMS eats it) while a loss is not.
    """
    tiling_cfg = cfg["ml"]["tiling"]
    ratio = float(tiling_cfg["edge_margin_ratio"])
    overlap = float(tiling_cfg["overlap"])
    if ratio > overlap / 2.0 + 1e-9:
        raise ValueError(
            f"ml.tiling.edge_margin_ratio ({ratio}) must be <= overlap/2 ({overlap / 2.0}), "
            "otherwise bars fall between tiles"
        )
    return ratio * float(tiling_cfg["tile_size"])


def keep_detection(box: Sequence[float], tile: Tile, margin: float) -> bool:
    """Drop boxes whose centre hugs an interior tile edge - they are cut in half there.

    A tile edge that is also the image edge is real, not an artefact, so detections
    there are kept.
    """
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    if not tile.touches_left and cx < margin:
        return False
    if not tile.touches_top and cy < margin:
        return False
    if not tile.touches_right and cx > tile.width - margin:
        return False
    if not tile.touches_bottom and cy > tile.height - margin:
        return False
    return True


def to_global(boxes: np.ndarray, tile: Tile) -> np.ndarray:
    """Tile-local (x1, y1, x2, y2) -> image coordinates."""
    if boxes.size == 0:
        return boxes
    shifted = boxes.astype(np.float64).copy()
    shifted[:, [0, 2]] += tile.left
    shifted[:, [1, 3]] += tile.top
    return shifted


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Plain greedy NMS. Kept in numpy so the module has no torch dependency."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        overlap_w = np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]))
        overlap_h = np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]))
        intersection = overlap_w * overlap_h
        union = areas[best] + areas[rest] - intersection
        iou = np.where(union > 0, intersection / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def run_tiled(
    image: np.ndarray,
    predict: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect over the whole image tile by tile.

    `predict` receives one tile image and returns (boxes Nx4 in tile coordinates,
    scores N). Returns the merged (boxes, scores) in image coordinates.
    """
    nms_cfg = cfg["ml"]["nms"]
    score_threshold = float(nms_cfg["score_threshold"])
    margin = _edge_margin(cfg)

    height, width = image.shape[:2]
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for tile in plan_tiles(width, height, cfg):
        boxes, scores = predict(crop(image, tile))
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if boxes.size == 0:
            continue

        wanted = scores >= score_threshold
        boxes, scores = boxes[wanted], scores[wanted]
        if boxes.size == 0:
            continue

        inside = np.array([keep_detection(box, tile, margin) for box in boxes], dtype=bool)
        boxes, scores = boxes[inside], scores[inside]
        if boxes.size == 0:
            continue

        all_boxes.append(to_global(boxes, tile))
        all_scores.append(scores)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros((0,))

    merged_boxes = np.vstack(all_boxes)
    merged_scores = np.concatenate(all_scores)
    keep = nms(merged_boxes, merged_scores, float(nms_cfg["iou_threshold"]))
    return merged_boxes[keep], merged_scores[keep]


def boxes_to_points(boxes: np.ndarray) -> list[list[float]]:
    """(x1, y1, x2, y2) -> the [x, y, r] contract count.py speaks."""
    points = []
    for x1, y1, x2, y2 in boxes:
        points.append(
            [
                round((x1 + x2) / 2.0, 1),
                round((y1 + y2) / 2.0, 1),
                round(((x2 - x1) + (y2 - y1)) / 4.0, 1),  # mean half-side
            ]
        )
    return points
