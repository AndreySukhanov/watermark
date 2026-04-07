import json
import math
import os
import shutil
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

MASK_PADDING = int(os.environ.get("MASK_PADDING", "8"))
MASK_DILATE = int(os.environ.get("MASK_DILATE", "4"))
MASK_MIN_RATIO = float(os.environ.get("MASK_MIN_RATIO", "0.002"))
MASK_MAX_RATIO = float(os.environ.get("MASK_MAX_RATIO", "0.18"))
IOPAINT_HD_STRATEGY = os.environ.get("IOPAINT_HD_STRATEGY", "Resize")
IOPAINT_RESIZE_LIMIT = int(os.environ.get("IOPAINT_RESIZE_LIMIT", "1024"))
FINAL_FRAME_SUFFIX = os.environ.get("FINAL_FRAME_SUFFIX", ".jpg").lower()
if FINAL_FRAME_SUFFIX not in {".jpg", ".png"}:
    FINAL_FRAME_SUFFIX = ".jpg"
FINAL_FRAME_QUALITY = int(os.environ.get("FINAL_FRAME_QUALITY", "99"))
COMPOSE_FEATHER_RADIUS = int(os.environ.get("COMPOSE_FEATHER_RADIUS", "2"))
COMPOSE_BLEND_SKIPPED = os.environ.get("COMPOSE_BLEND_SKIPPED", "1").lower() not in {
    "0",
    "false",
    "no",
}


# Number of parallel IOPaint workers (LAMA uses ~1.5 GB VRAM each)
def get_worker_count(device="cpu"):
    if device == "cuda":
        # Resize strategy lowered per-worker pressure enough to run eight workers on A5000.
        return 8
    return 1


def _mask_pixel_ratio(mask: Image.Image) -> float:
    hist = mask.histogram()
    total = mask.size[0] * mask.size[1]
    if total <= 0:
        return 0.0
    return hist[255] / float(total)


def _build_band_mask(width: int, height: int) -> Image.Image:
    band = Image.new("L", (width, height), 0)
    left = max(0, int(width * 0.04))
    right = min(width, int(width * 0.96))
    top = max(0, int(height * 0.30))
    bottom = min(height, int(height * 0.72))
    ImageDraw.Draw(band).rectangle((left, top, right, bottom), fill=255)
    return band.filter(ImageFilter.GaussianBlur(1)).point(lambda p: 255 if p >= 18 else 0)


def _refine_region_mask(frame_crop: Image.Image) -> Image.Image:
    gray = frame_crop.convert("L")
    blur = gray.filter(ImageFilter.GaussianBlur(10))
    detail = ImageChops.subtract(gray, blur)
    bright = gray.point(lambda p: 255 if p >= 132 else 0)
    detail = detail.point(lambda p: 255 if p >= 16 else 0)
    candidate = ImageChops.multiply(bright, detail).point(lambda p: 255 if p >= 60 else 0)
    candidate = candidate.filter(ImageFilter.MaxFilter(5))
    candidate = candidate.filter(ImageFilter.GaussianBlur(1))
    candidate = candidate.point(lambda p: 255 if p >= 22 else 0)

    ratio = _mask_pixel_ratio(candidate)
    if MASK_MIN_RATIO <= ratio <= MASK_MAX_RATIO:
        return candidate
    return _build_band_mask(*frame_crop.size)


def generate_mask(
    width,
    height,
    regions,
    out_path,
    padding=MASK_PADDING,
    dilate=MASK_DILATE,
    reference_frame_path=None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)

    reference_frame = None
    if reference_frame_path:
        ref_path = Path(reference_frame_path)
        if ref_path.exists():
            reference_frame = Image.open(ref_path).convert("RGB")

    try:
        for r in regions:
            x0 = max(0, int(r["x"]) - padding)
            y0 = max(0, int(r["y"]) - padding)
            x1 = min(width, int(r["x"]) + int(r["w"]) + padding)
            y1 = min(height, int(r["y"]) + int(r["h"]) + padding)
            if x1 <= x0 or y1 <= y0:
                continue
            if reference_frame is None:
                draw.rectangle([x0, y0, x1, y1], fill=255)
                continue
            frame_crop = reference_frame.crop((x0, y0, x1, y1))
            region_mask = _refine_region_mask(frame_crop)
            img.paste(region_mask, (x0, y0))
    finally:
        if reference_frame is not None:
            reference_frame.close()

    if dilate > 0:
        kernel = max(3, dilate * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        img = img.filter(ImageFilter.MaxFilter(kernel))
        img = img.point(lambda p: 255 if p >= 18 else 0)
    img.save(str(out_path))


def _temporal_sample_times(duration: float, sample_count: int) -> list[float]:
    sample_count = max(1, int(sample_count))
    if duration <= 0:
        return [0.0]
    if sample_count == 1:
        return [min(5.0, max(0.0, duration * 0.25))]
    start = min(max(0.3, duration * 0.05), max(0.0, duration - 0.2))
    end = max(start, min(max(0.6, duration - 0.3), duration * 0.92))
    return [round(float(t), 3) for t in np.linspace(start, end, sample_count)]


def _temporal_region_candidate(crop: Image.Image) -> np.ndarray:
    rgb = np.asarray(crop.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 6).astype(np.float32)
    detail = gray.astype(np.float32) - blur
    bright_low_sat = (value >= 120) & (saturation <= 95)
    bright_detail = detail >= 7
    candidate = (bright_low_sat & bright_detail).astype(np.uint8) * 255
    if candidate.max() == 0:
        candidate = ((value >= 145) & (saturation <= 115)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    return candidate


def generate_temporal_mask(
    input_video,
    width,
    height,
    duration,
    regions,
    out_path,
    *,
    work_dir,
    padding=MASK_PADDING,
    dilate=MASK_DILATE,
    samples=6,
    min_hits=2,
    register_process=None,
) -> list[Path]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    refs_dir = Path(work_dir) / "temporal_refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    reference_paths: list[Path] = []
    for idx, sample_time in enumerate(_temporal_sample_times(float(duration), int(samples)), start=1):
        ref_path = refs_dir / f"ref_{idx:02d}.jpg"
        extract_reference_frame(input_video, ref_path, time_sec=sample_time, register_process=register_process)
        reference_paths.append(ref_path)

    if not reference_paths:
        generate_mask(width, height, regions, out_path, padding=padding, dilate=dilate)
        return []

    frames = [Image.open(path).convert("RGB") for path in reference_paths]
    mask = Image.new("L", (width, height), 0)

    try:
        for region in regions:
            x0 = max(0, int(region["x"]) - padding)
            y0 = max(0, int(region["y"]) - padding)
            x1 = min(width, int(region["x"]) + int(region["w"]) + padding)
            y1 = min(height, int(region["y"]) + int(region["h"]) + padding)
            if x1 <= x0 or y1 <= y0:
                continue

            hits = None
            for frame in frames:
                candidate = _temporal_region_candidate(frame.crop((x0, y0, x1, y1)))
                if hits is None:
                    hits = np.zeros(candidate.shape, dtype=np.uint16)
                hits += (candidate > 0).astype(np.uint16)

            if hits is None:
                continue
            threshold = max(1, min(int(min_hits), len(frames)))
            region_mask = (hits >= threshold).astype(np.uint8) * 255
            coverage = float(region_mask.mean() / 255.0)
            if coverage < MASK_MIN_RATIO:
                continue
            if coverage > MASK_MAX_RATIO:
                threshold = min(len(frames), threshold + 1)
                region_mask = (hits >= threshold).astype(np.uint8) * 255
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            mask.paste(Image.fromarray(region_mask, mode="L"), (x0, y0))
    finally:
        for frame in frames:
            frame.close()

    if dilate > 0:
        kernel = max(3, dilate * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        mask = mask.filter(ImageFilter.MaxFilter(kernel))
        mask = mask.point(lambda p: 255 if p >= 18 else 0)
    mask.save(str(out_path))
    return reference_paths


def _terminate_process(proc):
    try:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
    except Exception:
        pass


def _run_managed_command(command, register_process=None):
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if register_process:
        register_process(proc)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return stdout, stderr


def get_total_frames(duration: float, fps: float) -> int:
    if duration <= 0 or fps <= 0:
        return 0
    return int(math.ceil(duration * fps))


def extract_reference_frame(input_path, out_path, time_sec=5.0, register_process=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_managed_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(max(0.0, time_sec)),
            "-i",
            str(input_path),
            "-frames:v",
            "1",
            str(out_path),
        ],
        register_process=register_process,
    )
    return out_path


def build_mask_preview(reference_frame_path, mask_path, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(reference_frame_path) as frame_image, Image.open(mask_path) as mask_image:
        frame = frame_image.convert("RGB")
        mask = mask_image.convert("L")
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        overlay.putalpha(mask.point(lambda p: 180 if p > 0 else 0))
        preview = Image.alpha_composite(frame.convert("RGBA"), overlay)
        preview.save(out_path)
    return out_path


def get_mask_stats(mask_path):
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        bbox = mask.getbbox()
        coverage = _mask_pixel_ratio(mask)
    return {
        "coverage": round(coverage * 100, 3),
        "bbox": bbox,
    }


def extract_frames(input_path, frames_dir, register_process=None):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    _run_managed_command(
        ["ffmpeg", "-y", "-i", str(input_path), str(frames_dir / "%06d.png")],
        register_process=register_process,
    )
    return sorted(frames_dir.glob("*.png"))


def _nearest_number(sorted_numbers, target):
    if not sorted_numbers:
        return None
    idx = bisect_left(sorted_numbers, target)
    if idx == 0:
        return sorted_numbers[0]
    if idx >= len(sorted_numbers):
        return sorted_numbers[-1]
    before = sorted_numbers[idx - 1]
    after = sorted_numbers[idx]
    return before if abs(target - before) <= abs(after - target) else after


def _neighbor_numbers(sorted_numbers, target):
    if not sorted_numbers:
        return None, None
    idx = bisect_left(sorted_numbers, target)
    prev_num = sorted_numbers[idx - 1] if idx > 0 else None
    next_num = sorted_numbers[idx] if idx < len(sorted_numbers) else None
    return prev_num, next_num


def extract_frames_range(input_path, frames_dir, start_frame, count, fps, register_process=None):
    """Extract a range of frames starting at start_frame (0-based).

    Uses -ss after -i for frame-accurate seeking.
    Frames are numbered starting from start_frame+1 (1-based, matching ffmpeg convention).
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    start_time = start_frame / fps
    _run_managed_command([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ss", str(start_time),
        "-frames:v", str(count),
        "-start_number", str(start_frame + 1),
        str(frames_dir / "%06d.png"),
    ], register_process=register_process)
    return sorted(frames_dir.glob("*.png"))


def thin_frames(frames_dir, skip=2, kept_dir=None):
    """Select frames on skip boundaries while preserving original numbering.

    If kept_dir is provided, the selected frames are copied there and the original
    directory remains intact. Otherwise, non-selected frames are deleted in-place.
    Returns count of selected frames.
    """
    frames_dir = Path(frames_dir)
    all_frames = sorted(frames_dir.glob("*.png"))
    if not all_frames or skip <= 1:
        if kept_dir:
            kept_dir = Path(kept_dir)
            kept_dir.mkdir(parents=True, exist_ok=True)
            for f in all_frames:
                shutil.copy2(str(f), str(kept_dir / f.name))
        return len(all_frames)

    keep = set()
    for i, num in enumerate(sorted(int(f.stem) for f in all_frames)):
        if i % skip == 0:
            keep.add(num)

    def _link_or_copy(src: Path, dst: Path):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(str(src), str(dst))

    if kept_dir:
        kept_dir = Path(kept_dir)
        kept_dir.mkdir(parents=True, exist_ok=True)
        for f in all_frames:
            if int(f.stem) in keep:
                _link_or_copy(f, kept_dir / f.name)
        return len(keep)

    for f in all_frames:
        if int(f.stem) not in keep:
            f.unlink()

    return len(keep)


def write_iopaint_config(
    config_path,
    hd_strategy=IOPAINT_HD_STRATEGY,
    resize_limit=IOPAINT_RESIZE_LIMIT,
):
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "hd_strategy": hd_strategy,
                "hd_strategy_resize_limit": resize_limit,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def start_iopaint(frames_dir, mask_path, output_dir, device="cpu", config_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "iopaint", "run", "--model", "lama", "--device", device,
        "--image", str(frames_dir), "--mask", str(mask_path),
        "--output", str(output_dir),
    ]
    if config_path:
        command.extend(["--config", str(config_path)])
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )


def run_iopaint_sync(
    frames_dir,
    mask_path,
    output_dir,
    device="cpu",
    register_process=None,
    config_path=None,
):
    """Run IOPaint synchronously, return (returncode, output_text)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = start_iopaint(
        frames_dir,
        mask_path,
        output_dir,
        device=device,
        config_path=config_path,
    )
    if register_process:
        register_process(proc)
    stdout, _ = proc.communicate()
    return proc.returncode, stdout or ""


def run_iopaint_parallel(frames_dir, mask_path, output_dir, device="cpu",
                         workers=None, register_process=None, is_cancelled=None,
                         config_path=None):
    """Split frames into sub-batches, run IOPaint in parallel.

    Returns (success: bool, error_message: str | None).
    All inpainted frames end up in output_dir with original names.
    """
    if workers is None:
        workers = get_worker_count(device)

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(frames_dir.glob("*.png"))
    if not all_frames:
        return True, None
    if is_cancelled and is_cancelled():
        return False, "Отменено пользователем"

    if len(all_frames) <= workers:
        rc, out = run_iopaint_sync(
            frames_dir,
            mask_path,
            output_dir,
            device,
            register_process=register_process,
            config_path=config_path,
        )
        if rc != 0:
            return False, f"IOPaint error (code {rc}): {out[-500:]}"
        return True, None

    chunk_size = math.ceil(len(all_frames) / workers)
    sub_dirs = []
    sub_outs = []
    for i in range(workers):
        chunk = all_frames[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            break
        sub_in = frames_dir / f"_sub{i}"
        sub_out = output_dir / f"_sub{i}"
        sub_in.mkdir(exist_ok=True)
        sub_out.mkdir(exist_ok=True)
        for f in chunk:
            shutil.move(str(f), str(sub_in / f.name))
        sub_dirs.append(sub_in)
        sub_outs.append(sub_out)

    started_processes = []
    started_lock = threading.Lock()

    def _register_subprocess(proc):
        with started_lock:
            started_processes.append(proc)
        if register_process:
            register_process(proc)

    def _terminate_started():
        with started_lock:
            for proc in started_processes:
                _terminate_process(proc)

    with ThreadPoolExecutor(max_workers=len(sub_dirs)) as pool:
        futures = {
            pool.submit(
                run_iopaint_sync,
                sd,
                mask_path,
                so,
                device,
                _register_subprocess,
                config_path,
            ): (sd, so)
            for sd, so in zip(sub_dirs, sub_outs)
        }
        for fut in as_completed(futures):
            if is_cancelled and is_cancelled():
                _terminate_started()
                return False, "Отменено пользователем"
            rc, out = fut.result()
            if rc != 0:
                _terminate_started()
                return False, f"IOPaint error (code {rc}): {out[-500:]}"
    if is_cancelled and is_cancelled():
        _terminate_started()
        return False, "Отменено пользователем"

    for so in sub_outs:
        for f in so.glob("*.png"):
            shutil.move(str(f), str(output_dir / f.name))
        shutil.rmtree(so, ignore_errors=True)

    for sd in sub_dirs:
        shutil.rmtree(sd, ignore_errors=True)

    return True, None


def compose_inpainted_frames(
    original_frames_dir,
    inpainted_dir,
    output_dir,
    mask_path,
    feather_radius=COMPOSE_FEATHER_RADIUS,
    output_suffix=FINAL_FRAME_SUFFIX,
    output_quality=FINAL_FRAME_QUALITY,
    blend_skipped=COMPOSE_BLEND_SKIPPED,
):
    """Compose final frames as original + masked region from nearest inpainted frame.

    This avoids full-frame duplication artifacts when FRAME_SKIP > 1.
    """
    original_frames_dir = Path(original_frames_dir)
    inpainted_dir = Path(inpainted_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_frames = sorted(original_frames_dir.glob("*.png"))
    inpainted_map = {int(f.stem): f for f in inpainted_dir.glob("*.png")}
    inpainted_nums = sorted(inpainted_map)
    if not original_frames or not inpainted_nums:
        return 0

    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
    mask_bbox = mask.getbbox()
    if not mask_bbox:
        for original_path in original_frames:
            with Image.open(original_path) as original_image:
                original_image.convert("RGB").save(
                    output_dir / f"{original_path.stem}{output_suffix}",
                    quality=output_quality,
                    subsampling=0,
                )
        return len(original_frames)

    mask_crop = mask.crop(mask_bbox)
    blend_mask = (
        mask_crop.filter(ImageFilter.GaussianBlur(feather_radius))
        if feather_radius > 0 else mask_crop
    )

    compose_workers = min(len(original_frames), max(1, min(8, (os.cpu_count() or 4))))

    @lru_cache(maxsize=32)
    def _load_source_crop(frame_num: int):
        with Image.open(inpainted_map[frame_num]) as source_image:
            return source_image.convert("RGB").crop(mask_bbox).copy()

    def _compose_one(original_path):
        output_path = output_dir / f"{original_path.stem}{output_suffix}"
        frame_num = int(original_path.stem)
        with Image.open(original_path) as original_image:
            original_img = original_image.convert("RGB")
            original_crop = original_img.crop(mask_bbox)

            if frame_num in inpainted_map:
                source_crop = _load_source_crop(frame_num)
            elif blend_skipped:
                prev_num, next_num = _neighbor_numbers(inpainted_nums, frame_num)
                if prev_num is None and next_num is None:
                    return 0
                if prev_num is None:
                    source_crop = _load_source_crop(next_num)
                elif next_num is None or prev_num == next_num:
                    source_crop = _load_source_crop(prev_num)
                else:
                    alpha = (frame_num - prev_num) / max(1, next_num - prev_num)
                    source_crop = Image.blend(
                        _load_source_crop(prev_num),
                        _load_source_crop(next_num),
                        alpha,
                    )
            else:
                source_num = _nearest_number(inpainted_nums, frame_num)
                if source_num is None:
                    return 0
                source_crop = _load_source_crop(source_num)

            composed_crop = Image.composite(source_crop, original_crop, blend_mask)
            original_img.paste(composed_crop, mask_bbox)
            if output_suffix == ".png":
                original_img.save(output_path)
            else:
                original_img.save(
                    output_path,
                    quality=output_quality,
                    subsampling=0,
                )
        return 1

    with ThreadPoolExecutor(max_workers=compose_workers) as pool:
        return sum(pool.map(_compose_one, original_frames))


def fill_skipped_frames(all_inpainted_dir, total_frames, skip=2):
    """Fill ALL missing frames 1..total_frames by copying nearest existing frame.

    Works regardless of which frames were processed — finds the nearest
    available frame for each gap.
    """
    if skip <= 1:
        return
    all_dir = Path(all_inpainted_dir)
    existing_nums = sorted(int(f.stem) for f in all_dir.glob("*.png"))
    if not existing_nums:
        return

    existing_set = set(existing_nums)
    prev_existing = existing_nums[0]
    for num in range(1, total_frames + 1):
        if num in existing_set:
            prev_existing = num
            continue
        src = all_dir / f"{prev_existing:06d}.png"
        dst = all_dir / f"{num:06d}.png"
        shutil.copy2(str(src), str(dst))


def reassemble_video(inpainted_dir, input_video, output_path, fps, register_process=None):
    inpainted_dir = Path(inpainted_dir)
    first_frame = next((f for f in sorted(inpainted_dir.iterdir()) if f.is_file()), None)
    if first_frame is None:
        raise RuntimeError("Не найдено кадров для сборки видео")
    frame_pattern = f"%06d{first_frame.suffix}"

    nvenc_command = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(inpainted_dir / frame_pattern),
        "-i", str(input_video),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "h264_nvenc", "-cq", "18", "-preset", "p4", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output_path),
    ]
    try:
        _run_managed_command(nvenc_command, register_process=register_process)
        return
    except subprocess.CalledProcessError:
        pass

    _run_managed_command([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(inpainted_dir / frame_pattern),
        "-i", str(input_video),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output_path),
    ], register_process=register_process)
