import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SOURCE_VIDEO = Path(os.environ.get("SOURCE_VIDEO", str(ROOT / "temp_web" / "arab_full_source.mp4")))
REGIONS_FILE = Path(os.environ.get("REGIONS_FILE", str(ROOT / "assets" / "arab_watermark_regions_wide.json")))
OUTPUT_DIR = Path(os.environ.get("SUPPRESS_OUTPUT_DIR", str(ROOT / "output" / "watermark_suppressor")))
CLIP_OFFSET = float(os.environ.get("SUPPRESS_OFFSET", "0"))
CLIP_DURATION = float(os.environ.get("SUPPRESS_DURATION", "15"))
ALPHA = float(os.environ.get("SUPPRESS_ALPHA", "0.34"))
VALUE_THRESHOLD = int(os.environ.get("SUPPRESS_VALUE_THRESHOLD", "135"))
SAT_THRESHOLD = int(os.environ.get("SUPPRESS_SAT_THRESHOLD", "70"))
MASK_BLUR = float(os.environ.get("SUPPRESS_MASK_BLUR", "1.2"))


def run(command: list[str]):
    subprocess.run(command, check=True)


def create_clip(work_dir: Path) -> Path:
    clip_path = work_dir / "source_15s.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(CLIP_OFFSET),
            "-i",
            str(SOURCE_VIDEO),
            "-t",
            str(CLIP_DURATION),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(clip_path),
        ]
    )
    return clip_path


def expand_region(region: dict, width: int, height: int):
    x0 = max(0, int(region["x"]))
    y0 = max(0, int(region["y"]))
    x1 = min(width, x0 + int(region["w"]))
    y1 = min(height, y0 + int(region["h"]))
    return x0, y0, x1, y1


def suppress_white_overlay(frame_path: Path, out_path: Path, regions: list[dict]):
    image = Image.open(frame_path).convert("RGB")
    arr = np.asarray(image).astype(np.float32)
    height, width = arr.shape[:2]

    for region in regions:
        x0, y0, x1, y1 = expand_region(region, width, height)
        if x1 <= x0 or y1 <= y0:
            continue

        crop = arr[y0:y1, x0:x1]
        maxc = crop.max(axis=2)
        minc = crop.min(axis=2)
        sat = np.zeros_like(maxc)
        nonzero = maxc > 1
        sat[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero] * 255.0
        mask = ((maxc >= VALUE_THRESHOLD) & (sat <= SAT_THRESHOLD)).astype(np.uint8) * 255

        if mask.max() == 0:
            continue

        mask_img = Image.fromarray(mask, mode="L").filter(ImageFilter.MaxFilter(5))
        if MASK_BLUR > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(MASK_BLUR))
        matte = np.asarray(mask_img).astype(np.float32) / 255.0
        alpha = np.clip(matte * ALPHA, 0.0, 0.85)[..., None]
        restored = (crop - alpha * 255.0) / np.maximum(1.0 - alpha, 0.05)
        crop[:] = crop * (1.0 - matte[..., None]) + restored * matte[..., None]

    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(out_path, quality=96)


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / stamp
    frames_dir = run_dir / "frames"
    out_frames_dir = run_dir / "suppressed"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_frames_dir.mkdir(parents=True, exist_ok=True)

    regions = json.loads(REGIONS_FILE.read_text(encoding="utf-8"))
    clip_path = create_clip(run_dir)

    run(["ffmpeg", "-y", "-i", str(clip_path), str(frames_dir / "%06d.jpg")])
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    for idx, frame_path in enumerate(frame_paths, start=1):
        suppress_white_overlay(frame_path, out_frames_dir / frame_path.name, regions)
        if idx % 50 == 0:
            print(f"processed {idx}/{len(frame_paths)}", flush=True)

    output_video = run_dir / "watermark_suppressed.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "25",
            "-i",
            str(out_frames_dir / "%06d.jpg"),
            "-i",
            str(clip_path),
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output_video),
        ]
    )
    preview_path = run_dir / "watermark_suppressed_5s.jpg"
    run(["ffmpeg", "-y", "-ss", "5", "-i", str(output_video), "-frames:v", "1", "-q:v", "2", str(preview_path)])
    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(out_frames_dir, ignore_errors=True)
    print(json.dumps({"run_dir": str(run_dir), "video": str(output_video), "preview": str(preview_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
