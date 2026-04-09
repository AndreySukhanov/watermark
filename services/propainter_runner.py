import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from services.ai_engines import AIEngineConfig
from services.iopaint_runner import (
    extract_reference_frame,
    generate_mask,
    generate_temporal_mask,
    reassemble_video,
)
from services.video_info import get_video_info
from services.watermark_segmenter import generate_hf_segmenter_mask, generate_hybrid_segmenter_mask

PROPAINTER_DIR = Path(os.environ.get("PROPAINTER_DIR", "/workspace/ProPainter"))
PROPAINTER_PYTHON = os.environ.get("PROPAINTER_PYTHON", "python3")


def _run_streaming_command(
    command,
    cwd=None,
    register_process=None,
    is_cancelled=None,
    emit_log=None,
    emit_progress=None,
    progress_span=None,
):
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if register_process:
        register_process(proc)

    progress_re = re.compile(r"(\d+)%\|")
    start_pct, end_pct = progress_span or (None, None)

    try:
        for line in proc.stdout:
            if is_cancelled and is_cancelled():
                proc.terminate()
                raise RuntimeError("Отменено пользователем")
            text = line.rstrip()
            if emit_log and text:
                emit_log(text)
            if emit_progress and start_pct is not None and end_pct is not None:
                if match := progress_re.search(text):
                    pct = int(match.group(1))
                    mapped = start_pct + int((end_pct - start_pct) * (pct / 100.0))
                    emit_progress(mapped)
        proc.wait()
    finally:
        if proc.stdout:
            proc.stdout.close()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)


def ensure_propainter_available(propainter_dir: Path | None = None):
    repo_dir = Path(propainter_dir or PROPAINTER_DIR)
    if not (repo_dir / "inference_propainter.py").exists():
        raise RuntimeError(
            "ProPainter не установлен на сервере. Ожидался репозиторий в "
            f"{repo_dir}"
        )
    return repo_dir


def _clamp_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        max(0, min(width, int(x0))),
        max(0, min(height, int(y0))),
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
    )


def _box_size(box) -> tuple[int, int]:
    return max(0, int(box[2] - box[0])), max(0, int(box[3] - box[1]))


def _box_area(box) -> int:
    width, height = _box_size(box)
    return width * height


def _union_box(left, right) -> tuple[int, int, int, int]:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _boxes_close(left, right, gap: int) -> bool:
    return not (
        left[2] + gap < right[0]
        or right[2] + gap < left[0]
        or left[3] + gap < right[1]
        or right[3] + gap < left[1]
    )


def _region_to_crop_box(region: dict, width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    return _clamp_box(
        (
            int(region["x"]) - padding,
            int(region["y"]) - padding,
            int(region["x"]) + int(region["w"]) + padding,
            int(region["y"]) + int(region["h"]) + padding,
        ),
        width,
        height,
    )


def plan_propainter_crop_groups(width: int, height: int, regions: list[dict], engine_config: AIEngineConfig) -> list[dict]:
    if not regions:
        return []

    max_width = max(64, int(engine_config.propainter_crop_max_width))
    max_height = max(64, int(engine_config.propainter_crop_max_height))
    padding = max(0, int(engine_config.propainter_crop_padding))
    merge_gap = max(0, int(engine_config.propainter_crop_merge_gap))
    ordered_regions = sorted(regions, key=lambda item: (int(item["y"]), int(item["x"])))

    groups: list[dict] = []
    for region in ordered_regions:
        region_box = _region_to_crop_box(region, width, height, padding)
        best_idx = None
        best_union = None
        best_added_area = None

        for idx, group in enumerate(groups):
            if not _boxes_close(group["box"], region_box, merge_gap):
                continue
            union_box = _union_box(group["box"], region_box)
            union_width, union_height = _box_size(union_box)
            if union_width > max_width or union_height > max_height:
                continue
            added_area = _box_area(union_box) - _box_area(group["box"])
            if best_added_area is None or added_area < best_added_area:
                best_idx = idx
                best_union = union_box
                best_added_area = added_area

        if best_idx is None:
            groups.append({"box": region_box, "regions": [dict(region)]})
        else:
            groups[best_idx]["box"] = best_union
            groups[best_idx]["regions"].append(dict(region))

    result = []
    for idx, group in enumerate(sorted(groups, key=lambda item: (item["box"][1], item["box"][0])), start=1):
        x0, y0, x1, y1 = group["box"]
        result.append(
            {
                "index": idx,
                "box": (x0, y0, x1, y1),
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                "regions": group["regions"],
                "region_count": len(group["regions"]),
            }
        )
    return result


def build_propainter_crop_preview(
    reference_frame_path: Path,
    out_path: Path,
    crop_groups: list[dict],
    merged_regions: list[dict] | None = None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_regions = merged_regions or []

    with Image.open(reference_frame_path) as reference_image:
        preview = reference_image.convert("RGBA")
        overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        crop_palette = [
            ((224, 144, 32, 38), (236, 160, 48, 220)),
            ((184, 48, 32, 42), (220, 96, 80, 220)),
            ((74, 128, 80, 38), (126, 180, 118, 220)),
            ((96, 120, 184, 34), (144, 170, 236, 220)),
        ]

        for idx, group in enumerate(crop_groups):
            fill_rgba, stroke_rgba = crop_palette[idx % len(crop_palette)]
            x0 = int(group["x"])
            y0 = int(group["y"])
            x1 = x0 + int(group["w"])
            y1 = y0 + int(group["h"])
            draw.rectangle((x0, y0, x1, y1), fill=fill_rgba, outline=stroke_rgba, width=3)
            label = f"#{group['index']} · {group['region_count']}"
            label_x = x0 + 8
            label_y = max(4, y0 + 6)
            draw.rectangle((label_x - 4, label_y - 2, label_x + 66, label_y + 12), fill=(8, 7, 6, 180))
            draw.text((label_x, label_y), label, fill=stroke_rgba[:3] + (255,))

        for region in merged_regions:
            x0 = int(region["x"])
            y0 = int(region["y"])
            x1 = x0 + int(region["w"])
            y1 = y0 + int(region["h"])
            draw.rectangle((x0, y0, x1, y1), outline=(255, 245, 214, 180), width=1)

        preview = Image.alpha_composite(preview, overlay)
        preview.save(out_path)
    return out_path


def _watermark_signal_map(patch: np.ndarray) -> np.ndarray:
    max_rgb = patch.max(axis=2)
    min_rgb = patch.min(axis=2)
    saturation = np.where(max_rgb > 1e-6, (max_rgb - min_rgb) / max_rgb, 0.0)
    luminance = patch.mean(axis=2)
    return np.clip((luminance - 0.42) * 2.4 - saturation * 1.5, 0.0, 1.0)


def _integral_image(image: np.ndarray) -> np.ndarray:
    return np.pad(image.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _box_sum(integral: np.ndarray, x: int, y: int, width: int, height: int) -> float:
    return (
        integral[y + height, x + width]
        - integral[y, x + width]
        - integral[y + height, x]
        + integral[y, x]
    )


def tighten_propainter_regions(
    reference_frame_path: Path,
    regions: list[dict],
    width: int,
    height: int,
) -> list[dict]:
    if not regions:
        return []

    tightened = []
    with Image.open(reference_frame_path) as reference_image:
        frame = np.asarray(reference_image.convert("RGB"), dtype=np.float32) / 255.0

    for region in regions:
        x = max(0, int(region["x"]))
        y = max(0, int(region["y"]))
        w = max(1, int(region["w"]))
        h = max(1, int(region["h"]))
        x1 = min(width, x + w)
        y1 = min(height, y + h)
        if x1 <= x or y1 <= y:
            tightened.append(dict(region))
            continue

        patch = frame[y:y1, x:x1]
        if patch.size == 0:
            tightened.append(dict(region))
            continue

        signal = _watermark_signal_map(patch)
        if float(signal.max()) < 0.12 or patch.shape[1] < 120 or patch.shape[0] < 48:
            tightened.append({"x": x, "y": y, "w": x1 - x, "h": y1 - y})
            continue

        patch_w = patch.shape[1]
        patch_h = patch.shape[0]
        win_w = min(patch_w, max(96, int(round(patch_w * 0.72))))
        win_h = min(patch_h, max(42, int(round(patch_h * 0.62))))
        if win_w >= patch_w or win_h >= patch_h:
            tightened.append({"x": x, "y": y, "w": patch_w, "h": patch_h})
            continue

        integral = _integral_image(signal)
        best_score = None
        best_xy = (0, 0)
        for yy in range(0, max(1, patch_h - win_h + 1), 2):
            for xx in range(0, max(1, patch_w - win_w + 1), 2):
                score = _box_sum(integral, xx, yy, win_w, win_h)
                cx = xx + win_w / 2.0
                cy = yy + win_h / 2.0
                score -= abs(cx - patch_w / 2.0) * 0.03
                score -= abs(cy - patch_h / 2.0) * 0.05
                if best_score is None or score > best_score:
                    best_score = score
                    best_xy = (xx, yy)

        pad_x = max(4, int(round(patch_w * 0.04)))
        pad_y = max(4, int(round(patch_h * 0.06)))
        xx, yy = best_xy
        tx0 = max(0, xx - pad_x)
        ty0 = max(0, yy - pad_y)
        tx1 = min(patch_w, xx + win_w + pad_x)
        ty1 = min(patch_h, yy + win_h + pad_y)
        tightened.append(
            {
                "x": x + tx0,
                "y": y + ty0,
                "w": tx1 - tx0,
                "h": ty1 - ty0,
            }
        )

    return tightened


def _translate_regions(regions: list[dict], box) -> list[dict]:
    x0, y0, _, _ = box
    return [
        {
            "x": int(region["x"]) - x0,
            "y": int(region["y"]) - y0,
            "w": int(region["w"]),
            "h": int(region["h"]),
        }
        for region in regions
    ]


def _ensure_clean_dir(path: Path):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _build_mask(
    *,
    reference_frame_path: Path,
    mask_path: Path,
    width: int,
    height: int,
    regions: list[dict],
    engine_config: AIEngineConfig,
    emit_log,
    input_video: Path | None = None,
    duration: float | None = None,
    work_dir: Path | None = None,
    register_process=None,
):
    if engine_config.mask_shape == "hybrid_segmenter":
        emit_log("ProPainter: hybrid segmenter mask")
        generate_hybrid_segmenter_mask(
            reference_frame_path,
            mask_path,
            width=width,
            height=height,
            regions=regions,
            padding=engine_config.mask_padding,
            dilate=engine_config.mask_dilate,
            threshold=engine_config.segmenter_threshold,
            weights_name=engine_config.segmenter_weights,
        )
        return

    if engine_config.mask_shape == "hf_segmenter":
        emit_log("ProPainter: HF segmenter mask")
        generate_hf_segmenter_mask(
            reference_frame_path,
            mask_path,
            width=width,
            height=height,
            regions=regions,
            padding=engine_config.mask_padding,
            dilate=engine_config.mask_dilate,
            threshold=engine_config.segmenter_threshold,
            weights_name=engine_config.segmenter_weights,
        )
        return

    if (
        engine_config.temporal_mask_samples > 1
        and input_video is not None
        and duration is not None
        and work_dir is not None
    ):
        emit_log(
            "ProPainter: temporal mask "
            f"samples={engine_config.temporal_mask_samples} min_hits={engine_config.temporal_mask_min_hits}"
        )
        generate_temporal_mask(
            input_video,
            width,
            height,
            duration,
            regions,
            mask_path,
            work_dir=work_dir,
            padding=engine_config.mask_padding,
            dilate=engine_config.mask_dilate,
            samples=engine_config.temporal_mask_samples,
            min_hits=engine_config.temporal_mask_min_hits,
            register_process=register_process,
        )
        return

    emit_log("ProPainter: glyph/region mask")
    generate_mask(
        width,
        height,
        regions,
        mask_path,
        padding=engine_config.mask_padding,
        dilate=engine_config.mask_dilate,
        reference_frame_path=reference_frame_path if engine_config.refine_mask else None,
    )


def _fit_propainter_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    width = max(32, int(width))
    height = max(32, int(height))
    max_width = max(64, int(max_width))
    max_height = max(64, int(max_height))

    scale = min(max_width / width, max_height / height)
    target_width = max(64, int(round(width * scale)))
    target_height = max(64, int(round(height * scale)))
    target_width = min(max_width, max(64, (target_width // 16) * 16 or 64))
    target_height = min(max_height, max(64, (target_height // 16) * 16 or 64))
    return target_width, target_height


def _extract_full_frames(input_video: Path, frames_dir: Path, register_process=None, is_cancelled=None):
    _ensure_clean_dir(frames_dir)
    _run_streaming_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            str(frames_dir / "%06d.png"),
        ],
        register_process=register_process,
        is_cancelled=is_cancelled,
    )


def _crop_frame_sequence(source_frames_dir: Path, crop_box, out_dir: Path):
    _ensure_clean_dir(out_dir)
    x0, y0, x1, y1 = crop_box
    frame_paths = sorted(source_frames_dir.glob("*.png"))

    def _crop_one(frame_path: Path):
        with Image.open(frame_path) as frame_image:
            crop = frame_image.convert("RGB").crop((x0, y0, x1, y1))
            crop.save(out_dir / frame_path.name)

    worker_count = min(8, max(1, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        list(pool.map(_crop_one, frame_paths))


def _extract_video_frames(input_video: Path, frames_dir: Path, register_process=None, is_cancelled=None):
    _ensure_clean_dir(frames_dir)
    _run_streaming_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            str(frames_dir / "%06d.png"),
        ],
        register_process=register_process,
        is_cancelled=is_cancelled,
    )


def _run_propainter_inference(
    *,
    propainter_dir: Path,
    frames_dir: Path,
    mask_path: Path,
    output_root: Path,
    save_fps: int,
    target_width: int,
    target_height: int,
    engine_config: AIEngineConfig,
    register_process=None,
    is_cancelled=None,
    emit_log=None,
    emit_progress=None,
    progress_span=None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    infer_cmd = [
        PROPAINTER_PYTHON,
        "inference_propainter.py",
        "--video",
        str(frames_dir),
        "--mask",
        str(mask_path),
        "--output",
        str(output_root),
        "--save_fps",
        str(save_fps),
        "--width",
        str(target_width),
        "--height",
        str(target_height),
        "--subvideo_length",
        str(engine_config.propainter_subvideo_length),
        "--neighbor_length",
        str(engine_config.propainter_neighbor_length),
        "--ref_stride",
        str(engine_config.propainter_ref_stride),
        "--mask_dilation",
        str(engine_config.propainter_mask_dilation),
    ]
    if engine_config.propainter_fp16:
        infer_cmd.append("--fp16")

    emit_log(
        "ProPainter: inference "
        f"{target_width}x{target_height} subvideo={engine_config.propainter_subvideo_length}"
    )
    _run_streaming_command(
        infer_cmd,
        cwd=propainter_dir,
        register_process=register_process,
        is_cancelled=is_cancelled,
        emit_log=emit_log,
        emit_progress=emit_progress,
        progress_span=progress_span,
    )
    output_video = output_root / frames_dir.name / "inpaint_out.mp4"
    if not output_video.exists():
        raise RuntimeError(f"Не найден результат ProPainter: {output_video}")
    return output_video


def _compose_crop_group(
    *,
    source_frames_dir: Path,
    composed_frames_dir: Path,
    processed_frames_dir: Path,
    mask_path: Path,
    crop_box,
    feather_radius: int,
):
    x0, y0, x1, y1 = crop_box
    crop_width = x1 - x0
    crop_height = y1 - y0
    if crop_width <= 0 or crop_height <= 0:
        return

    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        blend_mask = (
            mask.filter(ImageFilter.GaussianBlur(feather_radius))
            if feather_radius > 0
            else mask
        )
        blend_mask = blend_mask.copy()

    processed_frames = sorted(processed_frames_dir.glob("*.png"))

    def _compose_one(processed_path: Path):
        target_path = composed_frames_dir / processed_path.name
        base_path = target_path if target_path.exists() else source_frames_dir / processed_path.name
        with Image.open(base_path) as base_image, Image.open(processed_path) as processed_image:
            base = base_image.convert("RGB")
            processed = processed_image.convert("RGB")
            if processed.size != (crop_width, crop_height):
                processed = processed.resize((crop_width, crop_height), Image.Resampling.BICUBIC)
            original_crop = base.crop((x0, y0, x1, y1))
            composed_crop = Image.composite(processed, original_crop, blend_mask)
            base.paste(composed_crop, (x0, y0))
            target_path.parent.mkdir(parents=True, exist_ok=True)
            base.save(target_path)

    worker_count = min(8, max(1, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        list(pool.map(_compose_one, processed_frames))


def _run_propainter_full_frame_pipeline(
    *,
    input_video: Path,
    output_path: Path,
    work_dir: Path,
    info,
    regions: list[dict],
    engine_config: AIEngineConfig,
    emit_log,
    emit_progress,
    register_process=None,
    is_cancelled=None,
):
    propainter_dir = ensure_propainter_available()
    reference_time = min(5.0, max(0.0, info.duration * 0.25))
    reference_frame_path = work_dir / "reference.jpg"
    mask_path = work_dir / "mask.png"
    frames_dir = work_dir / "source_frames"
    output_root = work_dir / "propainter_output"
    frames_dir.mkdir(parents=True, exist_ok=True)

    emit_log(f"ProPainter: reference frame {reference_time:.1f}s")
    emit_progress(4)
    extract_reference_frame(
        input_video,
        reference_frame_path,
        time_sec=reference_time,
        register_process=register_process,
    )
    effective_regions = regions
    if engine_config.propainter_tighten_regions:
        effective_regions = tighten_propainter_regions(reference_frame_path, regions, info.width, info.height)
        emit_log(f"ProPainter: tightened regions {len(effective_regions)}")
    _build_mask(
        reference_frame_path=reference_frame_path,
        mask_path=mask_path,
        width=info.width,
        height=info.height,
        regions=effective_regions,
        engine_config=engine_config,
        emit_log=emit_log,
        input_video=input_video,
        duration=info.duration,
        work_dir=work_dir,
        register_process=register_process,
    )

    emit_log("ProPainter: извлечение исходных кадров...")
    extract_started = time.perf_counter()
    _extract_full_frames(
        input_video,
        frames_dir,
        register_process=register_process,
        is_cancelled=is_cancelled,
    )
    emit_log(f"  Extract frames: {time.perf_counter() - extract_started:.1f}s")
    emit_progress(16)

    save_fps = max(1, int(round(info.fps or 25.0)))
    infer_started = time.perf_counter()
    propainter_video = _run_propainter_inference(
        propainter_dir=propainter_dir,
        frames_dir=frames_dir,
        mask_path=mask_path,
        output_root=output_root,
        save_fps=save_fps,
        target_width=engine_config.propainter_width,
        target_height=engine_config.propainter_height,
        engine_config=engine_config,
        register_process=register_process,
        is_cancelled=is_cancelled,
        emit_log=emit_log,
        emit_progress=emit_progress,
        progress_span=(20, 88),
    )
    emit_log(f"  Inference: {time.perf_counter() - infer_started:.1f}s")

    emit_log("ProPainter: upscale + audio mux...")
    emit_progress(92)
    mux_started = time.perf_counter()
    _run_streaming_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(propainter_video),
            "-i",
            str(input_video),
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-vf",
            f"scale={info.width}:{info.height}",
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
            str(output_path),
        ],
        register_process=register_process,
        is_cancelled=is_cancelled,
    )
    emit_log(f"  Mux/upscale: {time.perf_counter() - mux_started:.1f}s")
    emit_progress(99)


def _run_propainter_crop_pipeline(
    *,
    input_video: Path,
    output_path: Path,
    work_dir: Path,
    info,
    regions: list[dict],
    engine_config: AIEngineConfig,
    emit_log,
    emit_progress,
    register_process=None,
    is_cancelled=None,
):
    propainter_dir = ensure_propainter_available()
    reference_time = min(5.0, max(0.0, info.duration * 0.25))
    reference_frame_path = work_dir / "reference.jpg"
    source_frames_dir = work_dir / "source_frames"
    composed_frames_dir = work_dir / "composed_frames"

    emit_log(f"ProPainter crop: reference frame {reference_time:.1f}s")
    emit_progress(4)
    extract_reference_frame(
        input_video,
        reference_frame_path,
        time_sec=reference_time,
        register_process=register_process,
    )
    effective_regions = regions
    if engine_config.propainter_tighten_regions:
        effective_regions = tighten_propainter_regions(reference_frame_path, regions, info.width, info.height)
        emit_log(f"ProPainter: tightened regions {len(effective_regions)}")

    crop_groups = plan_propainter_crop_groups(info.width, info.height, effective_regions, engine_config)
    if not crop_groups:
        raise RuntimeError("Не удалось построить crop-группы для ProPainter")
    emit_log(f"ProPainter crop groups: {len(crop_groups)}")
    for group in crop_groups:
        emit_log(
            f"  Crop {group['index']}: {group['region_count']} regions, "
            f"{group['w']}x{group['h']}"
        )

    emit_log("ProPainter: извлечение исходных кадров...")
    extract_started = time.perf_counter()
    _extract_full_frames(
        input_video,
        source_frames_dir,
        register_process=register_process,
        is_cancelled=is_cancelled,
    )
    emit_log(f"  Extract frames: {time.perf_counter() - extract_started:.1f}s")
    emit_progress(16)

    save_fps = max(1, int(round(info.fps or 25.0)))
    for index, group in enumerate(crop_groups, start=1):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Отменено пользователем")

        x0, y0, x1, y1 = group["box"]
        crop_width = x1 - x0
        crop_height = y1 - y0
        group_dir = work_dir / f"crop_{index:02d}"
        crop_reference_path = group_dir / "reference.jpg"
        crop_mask_path = group_dir / "mask.png"
        crop_frames_dir = group_dir / "frames"
        crop_output_root = group_dir / "propainter_output"
        crop_processed_dir = group_dir / "processed_frames"

        group_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(reference_frame_path) as reference_image:
            reference_image.convert("RGB").crop((x0, y0, x1, y1)).save(crop_reference_path)

        translated_regions = _translate_regions(group["regions"], group["box"])
        emit_log(
            f"ProPainter crop {index}/{len(crop_groups)}: mask "
            f"{crop_width}x{crop_height}"
        )
        _build_mask(
            reference_frame_path=crop_reference_path,
            mask_path=crop_mask_path,
            width=crop_width,
            height=crop_height,
            regions=translated_regions,
            engine_config=engine_config,
            emit_log=emit_log,
        )

        emit_log(f"ProPainter crop {index}/{len(crop_groups)}: кадры...")
        crop_extract_started = time.perf_counter()
        _crop_frame_sequence(source_frames_dir, group["box"], crop_frames_dir)
        emit_log(f"  Crop extract: {time.perf_counter() - crop_extract_started:.1f}s")

        target_width, target_height = _fit_propainter_size(
            crop_width,
            crop_height,
            engine_config.propainter_width,
            engine_config.propainter_height,
        )
        group_start = 20 + int((68 * (index - 1)) / len(crop_groups))
        group_end = 20 + int((68 * index) / len(crop_groups))
        infer_started = time.perf_counter()
        propainter_video = _run_propainter_inference(
            propainter_dir=propainter_dir,
            frames_dir=crop_frames_dir,
            mask_path=crop_mask_path,
            output_root=crop_output_root,
            save_fps=save_fps,
            target_width=target_width,
            target_height=target_height,
            engine_config=engine_config,
            register_process=register_process,
            is_cancelled=is_cancelled,
            emit_log=emit_log,
            emit_progress=emit_progress,
            progress_span=(group_start, max(group_start, group_end - 2)),
        )
        emit_log(f"  Crop inference: {time.perf_counter() - infer_started:.1f}s")

        emit_log(f"ProPainter crop {index}/{len(crop_groups)}: извлечение результата...")
        _extract_video_frames(
            propainter_video,
            crop_processed_dir,
            register_process=register_process,
            is_cancelled=is_cancelled,
        )

        emit_log(f"ProPainter crop {index}/{len(crop_groups)}: композиция...")
        compose_started = time.perf_counter()
        _compose_crop_group(
            source_frames_dir=source_frames_dir,
            composed_frames_dir=composed_frames_dir,
            processed_frames_dir=crop_processed_dir,
            mask_path=crop_mask_path,
            crop_box=group["box"],
            feather_radius=engine_config.feather_radius,
        )
        emit_log(f"  Crop compose: {time.perf_counter() - compose_started:.1f}s")
        emit_progress(group_end)

    emit_log("ProPainter crop: сборка видео...")
    emit_progress(92)
    reassemble_started = time.perf_counter()
    reassemble_video(
        composed_frames_dir,
        input_video,
        output_path,
        info.fps,
        register_process=register_process,
    )
    emit_log(f"  Reassemble: {time.perf_counter() - reassemble_started:.1f}s")
    emit_progress(99)


def run_propainter_pipeline(
    input_video,
    regions,
    output_path,
    work_dir,
    engine_config: AIEngineConfig,
    emit_log,
    emit_progress,
    register_process=None,
    is_cancelled=None,
):
    input_video = Path(input_video)
    output_path = Path(output_path)
    work_dir = Path(work_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = get_video_info(str(input_video))
    if info is None:
        raise RuntimeError(f"Не удалось прочитать видео: {input_video}")

    if engine_config.propainter_use_crops:
        return _run_propainter_crop_pipeline(
            input_video=input_video,
            output_path=output_path,
            work_dir=work_dir,
            info=info,
            regions=regions,
            engine_config=engine_config,
            emit_log=emit_log,
            emit_progress=emit_progress,
            register_process=register_process,
            is_cancelled=is_cancelled,
        )

    return _run_propainter_full_frame_pipeline(
        input_video=input_video,
        output_path=output_path,
        work_dir=work_dir,
        info=info,
        regions=regions,
        engine_config=engine_config,
        emit_log=emit_log,
        emit_progress=emit_progress,
        register_process=register_process,
        is_cancelled=is_cancelled,
    )
