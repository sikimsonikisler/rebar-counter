"""Phase 3 - tile.py on its own, against a synthetic image (SPEC 6).

The module is deliberately model-free: `run_tiled` takes a callable, so the tiling
and merge logic can be checked against an oracle detector that cannot be wrong. If
these tests pass and the count is still off, the fault is in the network, not here.
"""

import numpy as np
import pytest

from make_synthetic import render
from ml import tile as tiling


def oracle(scene, window):
    """A detector that never misses: returns the true boxes, in tile coordinates."""
    side = 2.0 * scene.radius
    boxes = []
    for x, y in scene.positions:
        x1, y1 = x - side / 2 - window.left, y - side / 2 - window.top
        x2, y2 = x + side / 2 - window.left, y + side / 2 - window.top
        # A real detector only reports what it can see inside its own crop.
        if x2 > 0 and y2 > 0 and x1 < window.width and y1 < window.height:
            boxes.append([x1, y1, x2, y2])
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,))
    return np.array(boxes), np.ones(len(boxes))


def test_tiles_cover_every_pixel(cfg):
    tiles = tiling.plan_tiles(1600, 1200, cfg)
    covered = np.zeros((1200, 1600), dtype=bool)
    for window in tiles:
        covered[window.top : window.bottom, window.left : window.right] = True
    assert covered.all(), "tiling left a hole in the image"


def test_tiles_overlap_as_configured(cfg):
    cfg["ml"]["tiling"]["tile_size"] = 640
    cfg["ml"]["tiling"]["overlap"] = 0.25
    tiles = tiling.plan_tiles(2000, 640, cfg)
    lefts = sorted({t.left for t in tiles})
    steps = {b - a for a, b in zip(lefts, lefts[1:])}
    # The last tile is pulled back onto the edge, so it may step less - never more.
    assert max(steps) == 480  # 640 * (1 - 0.25)


def test_small_image_becomes_one_tile(cfg):
    tiles = tiling.plan_tiles(300, 200, cfg)
    assert len(tiles) == 1
    assert (tiles[0].width, tiles[0].height) == (300, 200)
    assert all([tiles[0].touches_left, tiles[0].touches_top, tiles[0].touches_right])


def test_edge_margin_wider_than_half_the_overlap_is_rejected(cfg):
    """Otherwise a bar can be dropped by both tiles that see it."""
    cfg["ml"]["tiling"]["overlap"] = 0.10
    cfg["ml"]["tiling"]["edge_margin_ratio"] = 0.20
    with pytest.raises(ValueError, match="edge_margin_ratio"):
        tiling.run_tiled(np.zeros((800, 800, 3), np.uint8), lambda _: ([], []), cfg)


def test_nms_collapses_duplicates_but_keeps_neighbours():
    boxes = np.array(
        [
            [0, 0, 10, 10],  # kept
            [1, 1, 11, 11],  # near-duplicate of the first
            [40, 0, 50, 10],  # a different bar
        ],
        dtype=np.float64,
    )
    scores = np.array([0.9, 0.8, 0.7])
    assert sorted(tiling.nms(boxes, scores, 0.35)) == [0, 2]


def test_nms_keeps_touching_bars_apart():
    """Cross-sections touch; NMS must not eat the neighbour of a detected bar."""
    boxes = np.array([[0, 0, 20, 20], [19, 0, 39, 20]], dtype=np.float64)
    keep = tiling.nms(boxes, np.array([0.9, 0.9]), 0.35)
    assert len(keep) == 2


def test_run_tiled_recovers_the_exact_count(cfg):
    """The whole point: many tiles in, one correct global count out."""
    scene = render(count=300, size=(1800, 1300), seed=11)

    def predict(tile_image):
        window = predict.window
        return oracle(scene, window)

    # run_tiled hands over crops, not windows, so walk the plan alongside it.
    windows = iter(tiling.plan_tiles(1800, 1300, cfg))

    def predict_with_window(tile_image):
        predict.window = next(windows)
        return predict(tile_image)

    boxes, scores = tiling.run_tiled(scene.image, predict_with_window, cfg)
    assert len(boxes) == scene.count
    assert len(scores) == scene.count


def test_run_tiled_maps_back_to_global_coordinates(cfg):
    """A detection in the bottom-right tile must not come back at the origin."""
    image = np.zeros((1400, 1800, 3), np.uint8)
    windows = iter(tiling.plan_tiles(1800, 1400, cfg))
    state = {}

    def predict(tile_image):
        state["window"] = next(windows)
        # One box dead centre of every tile, well away from any edge margin.
        w, h = state["window"].width, state["window"].height
        return np.array([[w / 2 - 15, h / 2 - 15, w / 2 + 15, h / 2 + 15]]), np.array([0.9])

    boxes, _ = tiling.run_tiled(image, predict, cfg)
    points = tiling.boxes_to_points(boxes)
    assert max(x for x, _, _ in points) > 900, "boxes never left the first tile's frame"
    assert all(0 <= x <= 1800 and 0 <= y <= 1400 for x, y, _ in points)


def test_oversized_photos_are_capped_before_tiling(cfg):
    """A 48 MP frame must not turn into 187 tiles when 48 already see the bars."""
    from ml.predict import _fit_for_tiling

    cfg["ml"]["tiling"]["max_long_edge"] = 4032
    huge = np.zeros((6048, 8064, 3), np.uint8)
    fitted, scale_back = _fit_for_tiling(huge, cfg)

    assert max(fitted.shape[:2]) == 4032
    assert scale_back == pytest.approx(2.0)
    # And the tile count drops with it.
    assert len(tiling.plan_tiles(fitted.shape[1], fitted.shape[0], cfg)) < len(
        tiling.plan_tiles(8064, 6048, cfg)
    )


def test_photos_within_the_cap_are_left_alone(cfg):
    from ml.predict import _fit_for_tiling

    cfg["ml"]["tiling"]["max_long_edge"] = 4032
    normal = np.zeros((3024, 4032, 3), np.uint8)
    fitted, scale_back = _fit_for_tiling(normal, cfg)
    assert fitted is normal
    assert scale_back == 1.0


def test_boxes_to_points_matches_the_count_contract():
    boxes = np.array([[10, 20, 30, 40]], dtype=np.float64)
    assert tiling.boxes_to_points(boxes) == [[20.0, 30.0, 10.0]]
