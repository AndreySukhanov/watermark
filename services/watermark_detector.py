from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionMatch:
    x: int
    y: int
    w: int
    h: int
    score: float

    def to_region(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "score": round(self.score, 4)}


def _clamp_region(region: dict, width: int, height: int) -> dict | None:
    x = max(0, int(region.get("x", 0)))
    y = max(0, int(region.get("y", 0)))
    w = max(0, int(region.get("w", 0)))
    h = max(0, int(region.get("h", 0)))
    if w <= 4 or h <= 4:
        return None
    if x >= width or y >= height:
        return None
    w = min(w, width - x)
    h = min(h, height - y)
    if w <= 4 or h <= 4:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _iou(a: dict, b: dict) -> float:
    ax2 = a["x"] + a["w"]
    ay2 = a["y"] + a["h"]
    bx2 = b["x"] + b["w"]
    by2 = b["y"] + b["h"]
    inter_x1 = max(a["x"], b["x"])
    inter_y1 = max(a["y"], b["y"])
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def dedupe_regions(regions: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    kept: list[dict] = []
    for region in regions:
        if any(_iou(region, existing) >= iou_threshold for existing in kept):
            continue
        kept.append(region)
    return kept


def _prepare_match_surface(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    detail = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    edges = cv2.Canny(detail, 40, 120)
    return cv2.addWeighted(detail, 0.75, edges, 0.25, 0)


def _template_matches(
    prepared_frame: np.ndarray,
    template_rect: dict,
    threshold: float,
    max_matches: int,
) -> list[RegionMatch]:
    x, y, w, h = template_rect["x"], template_rect["y"], template_rect["w"], template_rect["h"]
    template = prepared_frame[y : y + h, x : x + w]
    if template.size == 0 or template.shape[0] < 8 or template.shape[1] < 8:
        return []

    result = cv2.matchTemplate(prepared_frame, template, cv2.TM_CCOEFF_NORMED)
    matches: list[RegionMatch] = []
    suppression_pad = max(6, min(w, h) // 3)

    for _ in range(max_matches):
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            break
        mx, my = max_loc
        matches.append(RegionMatch(mx, my, w, h, float(max_val)))

        x0 = max(0, mx - suppression_pad)
        y0 = max(0, my - suppression_pad)
        x1 = min(result.shape[1], mx + w + suppression_pad)
        y1 = min(result.shape[0], my + h + suppression_pad)
        result[y0:y1, x0:x1] = -1.0
    return matches


def detect_repeated_regions(
    reference_frame_path: str | Path,
    seed_regions: list[dict],
    threshold: float = 0.58,
    max_matches_per_region: int = 12,
) -> list[dict]:
    frame = cv2.imread(str(reference_frame_path))
    if frame is None:
        raise RuntimeError(f"Не удалось открыть reference frame: {reference_frame_path}")

    height, width = frame.shape[:2]
    prepared = _prepare_match_surface(frame)
    candidates: list[dict] = []

    normalized = []
    for region in seed_regions:
        if clamped := _clamp_region(region, width, height):
            normalized.append(clamped)
            candidates.append(dict(clamped))

    for region in normalized:
        matches = _template_matches(prepared, region, threshold, max_matches_per_region)
        candidates.extend(match.to_region() for match in matches)

    candidates.sort(key=lambda item: float(item.get("score", 1.0)), reverse=True)
    deduped = dedupe_regions(
        [{"x": item["x"], "y": item["y"], "w": item["w"], "h": item["h"]} for item in candidates],
        iou_threshold=0.38,
    )
    deduped.sort(key=lambda item: (item["y"], item["x"]))
    return deduped
