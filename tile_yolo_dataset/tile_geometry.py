"""Pure geometry helpers shared by YOLO dataset tiling and tiled inference."""

from __future__ import annotations

import math
import random
from collections import defaultdict


def iter_tile_windows(
    width: int, height: int, tile: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Return overlapping tile windows, with final windows flush to image edges."""
    if width <= 0 or height <= 0 or tile <= 0:
        raise ValueError("width, height, and tile must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if width <= tile and height <= tile:
        return [(0, 0, width, height)]

    stride = max(1, int(tile * (1 - overlap)))

    def starts(length: int) -> list[int]:
        if length <= tile:
            return [0]
        positions = list(range(0, length - tile + 1, stride))
        final = length - tile
        if positions[-1] != final:
            positions.append(final)
        return positions

    return [
        (x1, y1, min(x1 + tile, width), min(y1 + tile, height))
        for y1 in starts(height)
        for x1 in starts(width)
    ]


def yolo_line_to_xyxy(
    parts: list[float], img_w: int, img_h: int
) -> tuple[int, float, float, float, float]:
    """Convert a YOLO class/center/size record to pixel-space xyxy coordinates."""
    cls_id, cx, cy, box_w, box_h = parts
    cx *= img_w
    cy *= img_h
    box_w *= img_w
    box_h *= img_h
    return (
        int(cls_id),
        cx - box_w / 2,
        cy - box_h / 2,
        cx + box_w / 2,
        cy + box_h / 2,
    )


def xyxy_to_yolo_line(
    cls_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    tile_w: int,
    tile_h: int,
) -> str:
    """Serialize a pixel-space xyxy box as a normalized YOLO annotation line."""
    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) / 2 / tile_w
    cy = (y1 + y2) / 2 / tile_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {box_w / tile_w:.6f} {box_h / tile_h:.6f}"


def clip_box_to_tile(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    tx1: int,
    ty1: int,
    tx2: int,
    ty2: int,
) -> tuple[float, float, float, float] | None:
    """Clip an xyxy box to a tile and return coordinates relative to its origin."""
    clipped_x1 = max(x1, tx1)
    clipped_y1 = max(y1, ty1)
    clipped_x2 = min(x2, tx2)
    clipped_y2 = min(y2, ty2)
    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        return None
    return (
        clipped_x1 - tx1,
        clipped_y1 - ty1,
        clipped_x2 - tx1,
        clipped_y2 - ty1,
    )


def keep_clipped_box(
    orig_area: float,
    clipped: tuple[float, float, float, float],
    min_frac: float = 0.2,
) -> bool:
    """Return whether a clipped box retains the required original-area fraction."""
    if orig_area <= 0:
        return False
    x1, y1, x2, y2 = clipped
    return (x2 - x1) * (y2 - y1) / orig_area >= min_frac


def select_empty_tiles(
    labelled_count: int,
    empty_indices: list[int],
    empty_frac: float = 0.10,
    seed: int = 42,
) -> list[int]:
    """Deterministically sample empty tiles without exceeding their output share."""
    if labelled_count <= 0 or empty_frac <= 0:
        return []
    if empty_frac >= 1:
        max_empty = len(empty_indices)
    else:
        max_empty = math.floor(labelled_count * empty_frac / (1 - empty_frac))
    chosen = list(empty_indices)
    random.Random(seed).shuffle(chosen)
    return chosen[:max_empty]


def nms_xyxy(dets: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Apply greedy confidence-ordered non-maximum suppression per class."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for detection in dets:
        by_class[detection["cls"]].append(detection)

    kept: list[dict] = []
    for class_dets in by_class.values():
        candidates = sorted(class_dets, key=lambda det: det["conf"], reverse=True)
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [
                candidate
                for candidate in candidates
                if _iou_xyxy(best, candidate) <= iou_thresh
            ]
    return kept


def _iou_xyxy(first: dict, second: dict) -> float:
    intersection_w = max(0.0, min(first["x2"], second["x2"]) - max(first["x1"], second["x1"]))
    intersection_h = max(0.0, min(first["y2"], second["y2"]) - max(first["y1"], second["y1"]))
    intersection = intersection_w * intersection_h
    first_area = max(0.0, first["x2"] - first["x1"]) * max(
        0.0, first["y2"] - first["y1"]
    )
    second_area = max(0.0, second["x2"] - second["x1"]) * max(
        0.0, second["y2"] - second["y1"]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
