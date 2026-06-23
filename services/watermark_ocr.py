"""OCR-based watermark mask generation.

Pipeline:
  1. Sample N frames evenly from the video, compute median → watermark is
     sharp while content is ghosted.
  2. Run EasyOCR on the median + each sample, fuzzy-filter text against
     known watermark fragments.
  3. Merge overlapping detections → initial tile positions.
  4. Pick a top-left clean tile as template, run cv2.matchTemplate on the
     median high-pass to catch any remaining tile positions the OCR missed
     (including ones always obscured by content).
  5. For every tile, binarize the median high-pass stamp → glyph-tight mask.
  6. Dilate by a small radius, save as PNG.

Results are cached under `assets/ocr_mask_cache/<video_sha8>_<paramhash>.png`
so the ~2-minute OCR cost is paid once per video.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE / "assets" / "ocr_mask_cache"
DEFAULT_MODEL_DIR = os.environ.get("OCR_MODEL_DIR", str(BASE / "models" / "easyocr"))

# Substrings (lowercased, no spaces) used to identify watermark text in OCR
# output. Override via `WATERMARK_TOKENS` env var (comma-separated). Default
# catches the most common watermark patterns: any TLD-like fragment.
_DEFAULT_TOKENS = (".com", ".net", ".io", ".tv", "www.", "http")
WATERMARK_TOKENS = tuple(
    s.strip().lower()
    for s in os.environ.get("WATERMARK_TOKENS", ",".join(_DEFAULT_TOKENS)).split(",")
    if s.strip()
)


def _cv_imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def _cv_imwrite(path: Path, img: np.ndarray, ext: str = ".png") -> None:
    ok, enc = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    enc.tofile(str(path))


def _extract_frame(video: Path, t: float, out: Path, register_process=None) -> None:
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-ss", f"{max(0.0, t):.2f}", "-i", str(video),
            "-frames:v", "1", "-q:v", "2", str(out),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if register_process:
        register_process(proc)
    proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting {t}s from {video}")


def _sha8(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Hash first 4 MB — watermark tile is static and determined by the
        # opening content, so this is fast and sufficient for cache keying.
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:12]


def _is_watermark_text(text: str, tokens: Iterable[str] = WATERMARK_TOKENS) -> bool:
    t = text.lower().replace(" ", "").replace("|", "i").replace("1", "i")
    return any(tok in t for tok in tokens)


def _merge_overlapping(rects, dist: int = 20):
    merged = []
    for r in sorted(rects, key=lambda r: (r["y"], r["x"])):
        placed = False
        for m in merged:
            ox0 = m["x"] - dist
            oy0 = m["y"] - dist
            ox1 = m["x"] + m["w"] + dist
            oy1 = m["y"] + m["h"] + dist
            if (r["x"] < ox1 and r["x"] + r["w"] > ox0
                    and r["y"] < oy1 and r["y"] + r["h"] > oy0):
                nx0 = min(m["x"], r["x"])
                ny0 = min(m["y"], r["y"])
                nx1 = max(m["x"] + m["w"], r["x"] + r["w"])
                ny1 = max(m["y"] + m["h"], r["y"] + r["h"])
                m["x"], m["y"] = nx0, ny0
                m["w"], m["h"] = nx1 - nx0, ny1 - ny0
                placed = True
                break
        if not placed:
            merged.append(dict(r))
    return merged


def _bbox_to_rect(bbox) -> dict:
    xs = [float(p[0]) for p in bbox]
    ys = [float(p[1]) for p in bbox]
    x0 = int(max(0, min(xs)))
    y0 = int(max(0, min(ys)))
    x1 = int(max(xs))
    y1 = int(max(ys))
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _run_ocr(reader, img_bgr: np.ndarray, W: int, H: int) -> list[dict]:
    results = reader.readtext(img_bgr, detail=1, paragraph=False)
    kept = []
    for bbox, text, conf in results:
        if float(conf) < 0.15:
            continue
        if not _is_watermark_text(text):
            continue
        rect = _bbox_to_rect(bbox)
        if rect["w"] < 60 or rect["h"] < 30:
            continue
        rect["x"] = max(0, rect["x"])
        rect["y"] = max(0, rect["y"])
        rect["w"] = min(W - rect["x"], rect["w"])
        rect["h"] = min(H - rect["y"], rect["h"])
        kept.append(rect)
    return kept


def _template_match_additional(median_hp: np.ndarray, seed: dict, existing: list[dict],
                                match_thresh: float) -> list[dict]:
    x, y, w, h = seed["x"], seed["y"], seed["w"], seed["h"]
    template = median_hp[y:y + h, x:x + w]
    if template.size == 0:
        return existing
    result = cv2.matchTemplate(median_hp, template, cv2.TM_CCOEFF_NORMED)
    tH, tW = template.shape
    nms_h = max(40, tH // 2)
    nms_w = max(40, tW // 2)
    picks: list[tuple[int, int, float]] = []
    work = result.copy()
    while work.max() >= match_thresh and len(picks) < 30:
        idx = np.unravel_index(np.argmax(work), work.shape)
        py, px = int(idx[0]), int(idx[1])
        picks.append((px, py, float(work[py, px])))
        work[max(0, py - nms_h):py + nms_h + 1,
             max(0, px - nms_w):px + nms_w + 1] = -1
    out = list(existing)
    for px, py, _score in picks:
        out.append({"x": px, "y": py, "w": int(tW), "h": int(tH)})
    return _merge_overlapping(out, dist=20)


def _build_glyph_mask(median_gray: np.ndarray, tiles: list[dict],
                       blur_radius: int, stroke_thresh: int, dilate_px: int) -> np.ndarray:
    H, W = median_gray.shape
    blur = cv2.GaussianBlur(median_gray, (0, 0), sigmaX=blur_radius)
    hp = cv2.absdiff(median_gray, blur)
    mask = np.zeros((H, W), dtype=np.uint8)
    for t in tiles:
        x, y = int(t["x"]), int(t["y"])
        w, h = int(t["w"]), int(t["h"])
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        if x2 <= x or y2 <= y:
            continue
        tile_hp = hp[y:y2, x:x2]
        _, stamp = cv2.threshold(tile_hp, stroke_thresh, 255, cv2.THRESH_BINARY)
        mask[y:y2, x:x2] = np.maximum(mask[y:y2, x:x2], stamp)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _subtract_face_zones(
    mask: np.ndarray,
    frames_gray: list[np.ndarray],
    W: int, H: int,
    *,
    margin: float = 0.22,
    emit_log=None,
) -> np.ndarray:
    """Detect faces across sampled frames and zero-out mask pixels inside face zones."""
    log = emit_log or (lambda *_a, **_k: None)
    from services.face_safe import create_detector
    detector = create_detector(W, H, emit_log=log)
    all_faces: list[tuple[int, int, int, int]] = []
    for gray in frames_gray:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        all_faces.extend(detector.detect(bgr))
    if not all_faces:
        log("OCR mask: face-safe — no faces detected, mask unchanged")
        return mask
    face_zone = np.zeros((H, W), dtype=np.uint8)
    for (x, y, w, h) in all_faces:
        mx = int(round(w * margin))
        my = int(round(h * margin))
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(W, x + w + mx)
        y1 = min(H, y + h + my)
        face_zone[y0:y1, x0:x1] = 255
    before = int((mask > 0).sum())
    mask = mask.copy()
    mask[face_zone > 0] = 0
    after = int((mask > 0).sum())
    erased_pct = (before - after) / max(1, before) * 100
    log(f"OCR mask: face-safe — {len(all_faces)} face detections across {len(frames_gray)} frames, "
        f"erased {erased_pct:.1f}% of mask pixels")
    return mask


def generate_ocr_mask(
    input_video: str | Path,
    mask_path: str | Path,
    *,
    width: int,
    height: int,
    duration: float,
    work_dir: str | Path | None = None,
    samples: int = 20,
    blur_radius: int = 11,
    stroke_thresh: int = 14,
    dilate_px: int = 5,
    match_thresh: float = 0.30,
    face_safe: bool = False,
    face_margin: float = 0.10,
    model_dir: str | None = None,
    register_process=None,
    emit_log=None,
) -> Path:
    """Build a glyph-tight mask by OCR-detecting the tiled watermark.

    Returns the path to the saved mask PNG.
    """
    input_video = Path(input_video)
    mask_path = Path(mask_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    log = emit_log or (lambda *_args, **_kwargs: None)

    # --- cache lookup ---
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params_blob = json.dumps({
        "w": width, "h": height, "s": samples, "b": blur_radius,
        "t": stroke_thresh, "d": dilate_px, "m": match_thresh,
        "fs": face_safe, "fm": face_margin,
    }, sort_keys=True).encode()
    video_key = _sha8(input_video)
    params_key = hashlib.sha1(params_blob).hexdigest()[:8]
    cache_file = CACHE_DIR / f"{video_key}_{params_key}.png"
    if cache_file.exists():
        log(f"OCR mask: cache hit {cache_file.name}")
        data = np.fromfile(str(cache_file), dtype=np.uint8)
        cached = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        _cv_imwrite(mask_path, cached)
        return mask_path

    log(f"OCR mask: sampling {samples} frames, computing median…")

    # --- sample frames and compute median ---
    tmp_parent = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="ocrmask_", dir=str(tmp_parent)))
    try:
        duration = max(1.0, float(duration or 0) or 1.0)
        times = np.linspace(duration * 0.02, duration * 0.98, samples)
        frames_gray: list[np.ndarray] = []
        for idx, t in enumerate(times):
            fp = tmp_dir / f"f_{idx:02d}.jpg"
            _extract_frame(input_video, float(t), fp, register_process=register_process)
            img = _cv_imread(fp, cv2.IMREAD_COLOR)
            if img is None:
                continue
            frames_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if not frames_gray:
            raise RuntimeError("OCR mask: no frames sampled")
        median_gray = np.median(np.stack(frames_gray, axis=0), axis=0).astype(np.uint8)
        median_bgr = cv2.cvtColor(median_gray, cv2.COLOR_GRAY2BGR)

        # --- run OCR ---
        log("OCR mask: loading EasyOCR reader…")
        import easyocr
        reader = easyocr.Reader(
            ["en"],
            gpu=False,
            model_storage_directory=model_dir or DEFAULT_MODEL_DIR,
            download_enabled=True,
            verbose=False,
        )

        detections: list[dict] = []
        detections += _run_ocr(reader, median_bgr, width, height)
        # Sample 4 direct frames too — interior tiles may be visible in some
        # individual frames even if ghosted in the median.
        for idx in np.linspace(0, len(frames_gray) - 1, 4).astype(int):
            ref = cv2.cvtColor(frames_gray[idx], cv2.COLOR_GRAY2BGR)
            detections += _run_ocr(reader, ref, width, height)

        merged = _merge_overlapping(detections, dist=20)
        log(f"OCR mask: kept {len(merged)} OCR-detected tile(s)")

        if not merged:
            raise RuntimeError("OCR mask: no watermark text detected")

        # --- template matching to catch obscured tiles ---
        blur = cv2.GaussianBlur(median_gray, (0, 0), sigmaX=blur_radius)
        median_hp = cv2.absdiff(median_gray, blur)
        seed = sorted(merged, key=lambda r: (r["y"], r["x"]))[0]
        tiles = _template_match_additional(median_hp, seed, merged, match_thresh)
        log(f"OCR mask: after template match → {len(tiles)} tile(s)")

        mask = _build_glyph_mask(
            median_gray, tiles,
            blur_radius=blur_radius,
            stroke_thresh=stroke_thresh,
            dilate_px=dilate_px,
        )

        if face_safe:
            mask = _subtract_face_zones(
                mask, frames_gray, width, height,
                margin=face_margin, emit_log=log,
            )

        coverage = (mask > 0).mean() * 100
        log(f"OCR mask: glyph coverage={coverage:.2f}%")

        _cv_imwrite(mask_path, mask)
        _cv_imwrite(cache_file, mask)
        return mask_path
    finally:
        try:
            for p in tmp_dir.iterdir():
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except OSError:
            pass
