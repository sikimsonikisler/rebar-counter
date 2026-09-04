"""Step 2 - turn a grayscale image into one connected blob per cross-section.

The hard part is the last function: in a tight stack the bars touch, so a plain
threshold yields one huge blob. distance transform + watershed cuts it apart.
"""

from __future__ import annotations

import cv2
import numpy as np


def _odd(value: float, minimum: int = 3) -> int:
    """Structuring elements and adaptive windows must be odd and at least `minimum`."""
    size = int(round(value))
    if size % 2 == 0:
        size += 1
    return max(minimum, size)


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(size), _odd(size)))


def _normalize_u8(image: np.ndarray) -> np.ndarray:
    """Scale a float map to 0-255 so it can be written as a debug PNG."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _flatness(gray: np.ndarray, cfg: dict, radius_hint: float | None) -> np.ndarray:
    """Keep the locally flat pixels; drop everything sitting on an edge.

    A cut face is flat, and whatever separates two bars is an edge - dark shadow in a
    real stack, a bright sliver of background elsewhere. Working on |I - local mean|
    excludes both kinds of separator, and because the comparison is against the pixel's
    own neighbourhood it does not matter how bright that particular bar is.
    """
    flatness_cfg = cfg["segment"]["flatness"]
    ratio = float(flatness_cfg["block_size_radius_ratio"])
    if radius_hint is not None and ratio > 0:
        block = _odd(radius_hint * ratio)
    else:
        block = _odd(flatness_cfg["block_size"])

    local_mean = cv2.GaussianBlur(gray, (block, block), 0)
    deviation = cv2.absdiff(gray, local_mean)
    # A percentile, not a fixed grey level, so the same setting holds across
    # exposures and noise levels.
    threshold = np.percentile(deviation, float(flatness_cfg["keep_percentile"]))
    return np.where(deviation <= threshold, np.uint8(255), np.uint8(0))


def _adaptive(gray: np.ndarray, cfg: dict, polarity: str) -> np.ndarray:
    adaptive_cfg = cfg["segment"]["adaptive"]
    threshold_type = cv2.THRESH_BINARY if polarity == "bright" else cv2.THRESH_BINARY_INV
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        _odd(adaptive_cfg["block_size"]),
        float(adaptive_cfg["c"]),
    )


def _canny(gray: np.ndarray, cfg: dict) -> np.ndarray:
    canny_cfg = cfg["segment"]["canny"]
    edges = cv2.Canny(gray, float(canny_cfg["low"]), float(canny_cfg["high"]))
    if int(canny_cfg["close_iterations"]) > 0:
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            _kernel(canny_cfg["close_kernel"]),
            iterations=int(canny_cfg["close_iterations"]),
        )
    # Edges are the gaps between bars; the interiors are what we want to keep.
    return cv2.bitwise_not(edges) if canny_cfg["invert_edges"] else edges


def foreground_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def resolve_polarity(gray: np.ndarray, cfg: dict) -> str:
    """Decide whether cut faces are brighter or darker than the gaps.

    In 'auto' mode both are tried and the one whose foreground fraction is closest
    to the expected one wins; a fraction outside the sane band is penalised.
    """
    segment_cfg = cfg["segment"]
    if segment_cfg["polarity"] in ("bright", "dark"):
        return segment_cfg["polarity"]

    auto_cfg = segment_cfg["auto_polarity"]
    expected = float(auto_cfg["expected_foreground_ratio"])
    low = float(auto_cfg["min_foreground_ratio"])
    high = float(auto_cfg["max_foreground_ratio"])

    best_polarity, best_score = "bright", None
    for polarity in ("bright", "dark"):
        ratio = foreground_ratio(_adaptive(gray, cfg, polarity))
        score = abs(ratio - expected)
        if not (low <= ratio <= high):
            score += 1000.0  # out of band: only chosen if the other one is worse too
        if best_score is None or score < best_score:
            best_polarity, best_score = polarity, score
    return best_polarity


def binarize(
    gray: np.ndarray, cfg: dict, debug=None, radius_hint: float | None = None
) -> np.ndarray:
    """Grayscale -> binary mask where 255 marks candidate cross-section pixels."""
    method = cfg["segment"]["method"]
    if method == "flatness":
        mask = _flatness(gray, cfg, radius_hint)
    elif method == "adaptive":
        mask = _adaptive(gray, cfg, resolve_polarity(gray, cfg))
    elif method == "canny":
        mask = _canny(gray, cfg)
    else:
        raise ValueError(f"unknown segment.method: {method}")
    if debug:
        debug.save("binary", mask)
    return mask


def clean_mask(mask: np.ndarray, cfg: dict, debug=None) -> np.ndarray:
    """Opening removes speckle, closing fills pinholes inside a cross-section."""
    morphology_cfg = cfg["segment"]["morphology"]
    cleaned = mask
    if int(morphology_cfg["open_iterations"]) > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            _kernel(morphology_cfg["open_kernel"]),
            iterations=int(morphology_cfg["open_iterations"]),
        )
    if int(morphology_cfg["close_iterations"]) > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            _kernel(morphology_cfg["close_kernel"]),
            iterations=int(morphology_cfg["close_iterations"]),
        )
    if debug:
        debug.save("morph", cleaned)
    return cleaned


def _seeds_local_max(distance: np.ndarray, window: int, min_distance: float) -> np.ndarray:
    """One seed per local peak of the distance transform.

    A pixel is a peak if nothing in its `window` neighbourhood is taller. Works on
    merged blobs, where a single global threshold would keep them merged.
    """
    dilated = cv2.dilate(distance, np.ones((_odd(window), _odd(window)), np.uint8))
    peaks = (distance >= dilated) & (distance > min_distance)
    return peaks.astype(np.uint8) * 255


def _seeds_global_ratio(distance: np.ndarray, ratio: float) -> np.ndarray:
    """Classic sure-foreground: distance > ratio * global max. Simple, undersegments."""
    _, seeds = cv2.threshold(distance, ratio * float(distance.max()), 255, cv2.THRESH_BINARY)
    return seeds.astype(np.uint8)


def build_seeds(distance: np.ndarray, cfg: dict, radius_hint: float | None) -> np.ndarray:
    """Seed markers for the watershed, sized from the radius estimate when we have one."""
    split_cfg = cfg["segment"]["split"]
    if split_cfg["seed_method"] == "global_ratio":
        return _seeds_global_ratio(distance, float(split_cfg["global_ratio"]))

    if radius_hint is None:
        window = int(split_cfg["initial_peak_kernel"])
        min_distance = float(split_cfg["min_peak_distance_px"])
    else:
        window = _odd(radius_hint * float(split_cfg["peak_kernel_radius_ratio"]))
        min_distance = max(
            float(split_cfg["min_peak_distance_px"]),
            radius_hint * float(split_cfg["min_peak_dist_radius_ratio"]),
        )
    return _seeds_local_max(distance, window, min_distance)


def _split_nearest_seed(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Cut the mask along the boundaries of each seed's nearest-neighbour region.

    Every pixel is labelled with its closest seed; where two labels meet, the mask is
    cut. This is pure geometry - unlike a grayscale watershed, one bar cannot swallow
    its neighbour just because the two are different shades of rust.
    """
    _, labels = cv2.distanceTransformWithLabels(
        cv2.bitwise_not(seeds), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP
    )
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    separated = mask.copy()
    separated[boundary] = 0
    return separated


def _split_watershed(
    gray: np.ndarray, mask: np.ndarray, seeds: np.ndarray, cfg: dict
) -> np.ndarray:
    """Classic marker watershed flooded on the grayscale image."""
    split_cfg = cfg["segment"]["split"]
    marker_count, markers = cv2.connectedComponents(seeds)
    if marker_count <= 1:
        return mask.copy()

    sure_background = cv2.dilate(
        mask,
        _kernel(split_cfg["background_dilate_kernel"]),
        iterations=int(split_cfg["background_dilate_iterations"]),
    )
    unknown = cv2.subtract(sure_background, seeds)

    markers = markers + 1  # reserve 1 for background, seeds become 2..n
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)

    separated = np.zeros(mask.shape, dtype=np.uint8)
    separated[(markers > 1) & (mask > 0)] = 255
    return separated


def split_touching(
    gray: np.ndarray,
    mask: np.ndarray,
    cfg: dict,
    debug=None,
    radius_hint: float | None = None,
) -> np.ndarray:
    """Cut touching cross-sections apart. Returns a mask with one blob per bar.

    radius_hint (the radius estimated by a previous pass) sizes the seed search
    window; without it a scale-agnostic default from config is used.
    """
    split_cfg = cfg["segment"]["split"]
    if not split_cfg["enabled"]:
        return mask.copy()

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if debug:
        debug.save("distance", _normalize_u8(distance))

    seeds = build_seeds(distance, cfg, radius_hint)
    if debug:
        debug.save("markers", cv2.dilate(seeds, _kernel(3)))
    if not seeds.any():
        return mask.copy()

    if split_cfg["method"] == "nearest_seed":
        return _split_nearest_seed(mask, seeds)
    return _split_watershed(gray, mask, seeds, cfg)
