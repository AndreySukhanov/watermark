"""Face restoration post-processing with GFPGAN.

Pipeline:
  1. GFPGAN restores faces damaged by LaMa inpainting
  2. Only faces overlapping the watermark mask are processed
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import torchvision.transforms.functional as _F
sys.modules.setdefault("torchvision.transforms.functional_tensor", _F)

BASE = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE / "models" / "gfpgan"
GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
GFPGAN_MODEL_PATH = MODELS_DIR / "GFPGANv1.4.pth"


def _cv_imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def _cv_imwrite(path: Path, img: np.ndarray, ext: str = ".png") -> None:
    ok, enc = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    enc.tofile(str(path))


def _ensure_model(emit_log: Callable | None = None) -> Path:
    if GFPGAN_MODEL_PATH.exists():
        return GFPGAN_MODEL_PATH
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log = emit_log or (lambda *a: None)
    log(f"face_restore: downloading GFPGANv1.4 → {GFPGAN_MODEL_PATH.name}")
    urllib.request.urlretrieve(GFPGAN_MODEL_URL, str(GFPGAN_MODEL_PATH))
    return GFPGAN_MODEL_PATH


def create_restorer(device: str = "cuda", emit_log: Callable | None = None):
    model_path = _ensure_model(emit_log)
    from gfpgan import GFPGANer
    return GFPGANer(
        model_path=str(model_path),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=device,
    )


def _face_overlaps_mask(face_bbox, mask: np.ndarray, min_overlap_ratio: float = 0.05) -> bool:
    x, y, w, h = face_bbox
    H, W = mask.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(W, int(x + w)), min(H, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return False
    face_area = max(1, int(w) * int(h))
    overlap_px = int((mask[y0:y1, x0:x1] > 0).sum())
    return (overlap_px / face_area) >= min_overlap_ratio


def restore_faces_in_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    mask_path: str | Path | None = None,
    emit_log: Callable[[str], None] | None = None,
) -> int:
    """Restore faces with GFPGAN. Only faces overlapping the mask are modified."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = emit_log or (lambda *a: None)

    frames = sorted(input_dir.glob("*.png"))
    if not frames:
        log("face_restore: no frames found")
        return 0

    mask = None
    if mask_path is not None:
        mask = _cv_imread(Path(mask_path), cv2.IMREAD_GRAYSCALE)

    log(f"face_restore: initializing GFPGAN on {device}…")
    restorer = create_restorer(device=device, emit_log=log)

    from services.face_safe import create_detector
    first_img = _cv_imread(frames[0])
    fH, fW = first_img.shape[:2]
    face_det = create_detector(fW, fH, emit_log=log)

    total = len(frames)
    restored_count = 0
    skipped = 0
    written = 0

    for idx, fp in enumerate(frames):
        img = _cv_imread(fp)
        if img is None:
            continue

        faces = face_det.detect(img)
        overlapping = [f for f in faces
                       if mask is None or _face_overlaps_mask(f, mask)]

        if overlapping:
            try:
                _, _, out = restorer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True,
                )
            except Exception:
                out = None

            if out is not None:
                _cv_imwrite(output_dir / fp.name, out)
                restored_count += 1
            else:
                if output_dir != input_dir:
                    _cv_imwrite(output_dir / fp.name, img)
                skipped += 1
        else:
            if output_dir != input_dir:
                _cv_imwrite(output_dir / fp.name, img)
            skipped += 1

        written += 1
        if (idx + 1) % 200 == 0 or idx + 1 == total:
            log(f"  face_restore: {idx + 1}/{total} (restored={restored_count}, skipped={skipped})")

    log(f"face_restore: done ({written} frames, {restored_count} restored, {skipped} skipped)")
    return written
