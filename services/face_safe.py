"""Face-aware post-processing to protect facial features from inpainting.

Pipeline:
  1. For each inpainted frame, load its matching *original* frame.
  2. Run YuNet on the original (small, fast, accurate frontal+profile).
  3. Build a feathered alpha from the union of face bboxes.
  4. Blend: out = inpainted * (1 - alpha) + original * alpha
  5. Where alpha is 1 we get the original face; where alpha is 0 we get the
     inpainted (watermark-removed) pixels. The feather avoids hard seams.

Falls back to OpenCV Haar frontal-face if YuNet model is unavailable.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
YUNET_DIR = BASE / "models" / "face_detection"
YUNET_PATH = YUNET_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _cv_imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def _cv_imwrite(path: Path, img: np.ndarray, ext: str = ".png") -> None:
    ok, enc = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    enc.tofile(str(path))


def ensure_yunet_model(emit_log: Callable[[str], None] | None = None) -> Path | None:
    """Download YuNet ONNX if missing. Return path or None on failure."""
    if YUNET_PATH.exists():
        return YUNET_PATH
    YUNET_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if emit_log:
            emit_log(f"face_safe: downloading YuNet model → {YUNET_PATH.name}")
        urllib.request.urlretrieve(YUNET_URL, str(YUNET_PATH))
        return YUNET_PATH
    except Exception as exc:
        if emit_log:
            emit_log(f"face_safe: YuNet download failed ({exc}); using Haar fallback")
        return None


class _YuNetDetector:
    def __init__(self, model_path: Path, width: int, height: int,
                 score_thresh: float = 0.6, nms_thresh: float = 0.3, top_k: int = 50):
        self._det = cv2.FaceDetectorYN_create(
            str(model_path), "", (width, height), score_thresh, nms_thresh, top_k
        )
        self._size = (width, height)

    def detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        H, W = bgr.shape[:2]
        if (W, H) != self._size:
            self._det.setInputSize((W, H))
            self._size = (W, H)
        _, faces = self._det.detect(bgr)
        if faces is None:
            return []
        out: list[tuple[int, int, int, int]] = []
        for f in faces:
            x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            if w <= 0 or h <= 0:
                continue
            out.append((x, y, w, h))
        return out


class _HaarDetector:
    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._frontal = cv2.CascadeClassifier(path)
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self._profile = cv2.CascadeClassifier(profile_path)

    def detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        out: list[tuple[int, int, int, int]] = []
        for det in (self._frontal, self._profile):
            faces = det.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
            for (x, y, w, h) in faces:
                out.append((int(x), int(y), int(w), int(h)))
        return out


def create_detector(width: int, height: int,
                    emit_log: Callable[[str], None] | None = None):
    model = ensure_yunet_model(emit_log)
    if model is not None:
        try:
            return _YuNetDetector(model, width, height)
        except Exception as exc:
            if emit_log:
                emit_log(f"face_safe: YuNet init failed ({exc}); using Haar")
    return _HaarDetector()


def build_face_alpha(
    H: int, W: int,
    faces: list[tuple[int, int, int, int]],
    *,
    margin_ratio: float = 0.18,
    feather: int = 25,
) -> np.ndarray | None:
    """Feathered float32 alpha in [0,1] where faces (expanded by margin) are 1."""
    if not faces:
        return None
    alpha = np.zeros((H, W), dtype=np.uint8)
    for (x, y, w, h) in faces:
        mx = int(round(w * margin_ratio))
        my = int(round(h * margin_ratio))
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(W, x + w + mx)
        y1 = min(H, y + h + my)
        if x1 <= x0 or y1 <= y0:
            continue
        alpha[y0:y1, x0:x1] = 255
    if feather > 0:
        k = max(1, feather) | 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return alpha.astype(np.float32) / 255.0


def blend_frame(original: np.ndarray, inpainted: np.ndarray,
                 alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return inpainted
    a = alpha[..., None]
    return (original.astype(np.float32) * a + inpainted.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def restore_faces_in_frames(
    original_frames_dir: str | Path,
    inpainted_dir: str | Path,
    output_dir: str | Path,
    *,
    margin_ratio: float = 0.18,
    feather: int = 25,
    emit_log: Callable[[str], None] | None = None,
) -> int:
    """Blend original face pixels back over inpainted frames.

    Returns the number of frames written.
    """
    original_frames_dir = Path(original_frames_dir)
    inpainted_dir = Path(inpainted_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = emit_log or (lambda *_a, **_k: None)

    inp_frames = sorted(inpainted_dir.glob("*.png"))
    if not inp_frames:
        log("face_safe: no inpainted frames found")
        return 0

    first = _cv_imread(inp_frames[0])
    if first is None:
        raise RuntimeError(f"face_safe: failed to read {inp_frames[0]}")
    H, W = first.shape[:2]
    detector = create_detector(W, H, emit_log=log)

    total = len(inp_frames)
    written = 0
    faces_seen = 0
    for idx, inp_path in enumerate(inp_frames):
        orig_path = original_frames_dir / inp_path.name
        if not orig_path.exists():
            inpainted = _cv_imread(inp_path)
            _cv_imwrite(output_dir / inp_path.name, inpainted)
            written += 1
            continue
        original = _cv_imread(orig_path)
        inpainted = _cv_imread(inp_path)
        if original is None or inpainted is None:
            continue
        faces = detector.detect(original)
        if faces:
            faces_seen += 1
        alpha = build_face_alpha(H, W, faces, margin_ratio=margin_ratio, feather=feather)
        blended = blend_frame(original, inpainted, alpha)
        _cv_imwrite(output_dir / inp_path.name, blended)
        written += 1
        if orig_path != inp_path:
            orig_path.unlink(missing_ok=True)
        if (idx + 1) % 200 == 0 or idx + 1 == total:
            log(f"face_safe: {idx + 1}/{total} frames ({faces_seen} with faces)")
    log(f"face_safe: done ({written}/{total}, {faces_seen} frames with detected faces)")
    return written
