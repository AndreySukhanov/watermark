import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from services.ai_engines import DEFAULT_AI_ENGINE, resolve_engine_config
from services.iopaint_runner import (
    build_mask_preview,
    extract_reference_frame,
    generate_mask,
    generate_temporal_mask,
    get_mask_stats,
)
from services.propainter_runner import (
    build_propainter_group_debug_image,
    build_propainter_crop_preview,
    plan_propainter_crop_groups,
    relax_propainter_crop_groups,
    summarize_propainter_crop_groups,
    tighten_regions_to_mask,
    tighten_propainter_regions,
)
from services.video_info import get_video_info
from services.watermark_detector import dedupe_regions, detect_repeated_regions
from services.watermark_segmenter import (
    generate_hf_segmenter_mask,
    generate_hybrid_segmenter_mask,
    generate_temporal_hf_segmenter_mask,
)


VIDEO_FILE = Path(os.environ.get("VIDEO_FILE", str(BASE / "Араб.mp4")))
REGIONS_FILE = Path(
    os.environ.get("REGIONS_FILE", str(BASE / "assets" / "arab_watermark_regions.json"))
)
ENGINE = os.environ.get("ENGINE", DEFAULT_AI_ENGINE)
ENGINE_OPTIONS_FILE = os.environ.get("ENGINE_OPTIONS_FILE")


def load_engine_options() -> dict:
    if ENGINE_OPTIONS_FILE:
        return json.loads(Path(ENGINE_OPTIONS_FILE).read_text(encoding="utf-8"))
    return json.loads(os.environ.get("ENGINE_OPTIONS_JSON", "{}") or "{}")


ENGINE_OPTIONS = load_engine_options()
OUTPUT_ROOT = Path(
    os.environ.get("OUTPUT_DIR", str(BASE / "test001" / "local_quality_probe"))
)
CLIP_DURATION = float(os.environ.get("CLIP_DURATION", "5"))
CLIP_OFFSET = float(os.environ.get("CLIP_OFFSET", "0"))
AUTODETECT = os.environ.get("AUTODETECT", "0").lower() not in {"0", "false", "no"}
ALLOW_SEGMENTER_FALLBACK = os.environ.get("ALLOW_SEGMENTER_FALLBACK", "1").lower() not in {"0", "false", "no"}
CLIP_CACHE_ROOT = BASE / "test001" / "_clip_cache"


def log(line: str):
    print(line, flush=True)


def run(command: list[str]):
    subprocess.run(command, check=True)


def run_maybe(command: list[str]) -> bool:
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def create_clip(source_video: Path, output_root: Path) -> Path:
    clip_key = hashlib.md5(str(source_video.resolve()).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    clip_name = f"probe_v2_{clip_key}_o{int(CLIP_OFFSET * 1000):06d}_d{int(CLIP_DURATION * 1000):06d}.mp4"
    clip_path = CLIP_CACHE_ROOT / clip_name
    if clip_path.exists():
        return clip_path
    CLIP_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"[clip] {source_video.name} -> {clip_path.name}")
    if run_maybe(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(CLIP_OFFSET),
            "-i",
            str(source_video),
            "-t",
            str(CLIP_DURATION),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            "-threads",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(clip_path),
        ]
    ):
        return clip_path
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(CLIP_OFFSET),
            "-i",
            str(source_video),
            "-t",
            str(CLIP_DURATION),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "1",
            str(clip_path),
        ]
    )
    return clip_path


def load_regions() -> list[dict]:
    return json.loads(REGIONS_FILE.read_text(encoding="utf-8"))


def reference_time(duration: float) -> float:
    return min(5.0, max(0.0, duration * 0.25))


def build_probe_mask(
    *,
    config,
    input_path: Path,
    reference_path: Path,
    work_dir: Path,
    mask_path: Path,
    info,
    merged_regions: list[dict],
) -> tuple[str, str | None]:
    try:
        if config.mask_shape == "temporal_hf_segmenter":
            generate_temporal_hf_segmenter_mask(
                str(input_path),
                reference_path,
                mask_path,
                width=info.width,
                height=info.height,
                duration=info.duration,
                regions=merged_regions,
                work_dir=work_dir,
                padding=config.mask_padding,
                dilate=config.mask_dilate,
                threshold=config.segmenter_threshold,
                weights_name=config.segmenter_weights,
                samples=max(4, int(config.temporal_mask_samples or 6)),
                min_hits=max(1, int(config.temporal_mask_min_hits or 2)),
            )
            return config.mask_shape, None
        if config.mask_shape == "hybrid_segmenter":
            generate_hybrid_segmenter_mask(
                reference_path,
                mask_path,
                width=info.width,
                height=info.height,
                regions=merged_regions,
                padding=config.mask_padding,
                dilate=config.mask_dilate,
                threshold=config.segmenter_threshold,
                weights_name=config.segmenter_weights,
            )
            return config.mask_shape, None
        if config.mask_shape == "hf_segmenter":
            generate_hf_segmenter_mask(
                reference_path,
                mask_path,
                width=info.width,
                height=info.height,
                regions=merged_regions,
                padding=config.mask_padding,
                dilate=config.mask_dilate,
                threshold=config.segmenter_threshold,
                weights_name=config.segmenter_weights,
            )
            return config.mask_shape, None
    except RuntimeError as exc:
        if not ALLOW_SEGMENTER_FALLBACK:
            raise
        warning = str(exc)
        log(f"[mask] fallback to glyph/auto refine: {warning}")
    else:
        warning = None

    if config.temporal_mask_samples > 1:
        generate_temporal_mask(
            str(input_path),
            info.width,
            info.height,
            info.duration,
            merged_regions,
            mask_path,
            work_dir=work_dir,
            padding=config.mask_padding,
            dilate=config.mask_dilate,
            samples=config.temporal_mask_samples,
            min_hits=config.temporal_mask_min_hits,
        )
        return "temporal_auto", warning

    generate_mask(
        info.width,
        info.height,
        merged_regions,
        mask_path,
        padding=config.mask_padding,
        dilate=config.mask_dilate,
        reference_frame_path=reference_path if config.refine_mask else None,
    )
    fallback_shape = "auto_refine" if config.refine_mask else "auto"
    return fallback_shape, warning


def build_quality_analysis(
    *,
    input_path: Path,
    regions: list[dict],
    engine_key: str,
    engine_options: dict,
    autodetect: bool,
    run_dir: Path,
) -> dict:
    info = get_video_info(str(input_path))
    if info is None:
        raise RuntimeError(f"Не удалось прочитать видео: {input_path}")

    config = resolve_engine_config(engine_key, engine_options)
    work_dir = run_dir / f"analysis_{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True, exist_ok=True)
    reference_path = run_dir / "reference.jpg"
    mask_path = work_dir / "mask.png"
    mask_preview_path = run_dir / "mask_preview.png"
    crop_preview_path = run_dir / "crop_preview.png"
    ref_time = reference_time(float(info.duration or 0.0))
    base_regions = dedupe_regions(
        [
            {
                "x": int(item["x"]),
                "y": int(item["y"]),
                "w": int(item["w"]),
                "h": int(item["h"]),
            }
            for item in regions
        ]
    )
    suggested_regions: list[dict] = []

    try:
        extract_reference_frame(str(input_path), reference_path, time_sec=ref_time)

        if autodetect and base_regions:
            current = list(base_regions)
            for _ in range(3 if len(base_regions) >= 3 else 1):
                detected = detect_repeated_regions(reference_path, current)
                merged = dedupe_regions(detected)
                if len(merged) <= len(current):
                    break
                current = merged[:12]
            for candidate in current:
                merged = dedupe_regions(base_regions + suggested_regions + [candidate])
                if len(merged) > len(base_regions) + len(suggested_regions):
                    suggested_regions.append(candidate)

        merged_regions = dedupe_regions(base_regions + suggested_regions)
        if config.family == "propainter" and config.propainter_tighten_regions:
            merged_regions = dedupe_regions(
                tighten_propainter_regions(reference_path, merged_regions, info.width, info.height)
            )

        mask_runtime_shape, mask_runtime_warning = build_probe_mask(
            config=config,
            input_path=input_path,
            reference_path=reference_path,
            work_dir=work_dir,
            mask_path=mask_path,
            info=info,
            merged_regions=merged_regions,
        )

        crop_groups = []
        crop_status_counts = {}
        if config.family == "propainter" and config.propainter_use_crops:
            crop_regions = tighten_regions_to_mask(mask_path, merged_regions, info.width, info.height)
            planned_groups = plan_propainter_crop_groups(info.width, info.height, crop_regions, config)
            if planned_groups:
                planned_groups = relax_propainter_crop_groups(
                    info.width,
                    info.height,
                    planned_groups,
                    summarize_propainter_crop_groups(mask_path, planned_groups),
                )
            crop_groups = [
                {
                    "index": item["index"],
                    "x": item["x"],
                    "y": item["y"],
                    "w": item["w"],
                    "h": item["h"],
                    "region_count": item["region_count"],
                }
                for item in planned_groups
            ]
            if crop_groups:
                group_summaries = {
                    item["index"]: item for item in summarize_propainter_crop_groups(mask_path, crop_groups)
                }
                group_debug_dir = run_dir / "crop_group_debug"
                for group in crop_groups:
                    summary = group_summaries.get(group["index"], {})
                    group.update(summary)
                    debug_path = group_debug_dir / f"group_{int(group['index']):02d}_overlay.png"
                    build_propainter_group_debug_image(
                        reference_path,
                        mask_path,
                        group,
                        debug_path,
                        summary=summary,
                    )
                    group["debug_path"] = str(debug_path)
                crop_status_counts = {
                    status: sum(1 for item in crop_groups if item.get("status") == status)
                    for status in ("safe", "tight", "risky", "empty")
                }
                build_propainter_crop_preview(
                    reference_path,
                    crop_preview_path,
                    crop_groups,
                    merged_regions=merged_regions,
                )

        build_mask_preview(reference_path, mask_path, mask_preview_path)
        stats = get_mask_stats(mask_path)
        crop_area_total = sum(int(item["w"]) * int(item["h"]) for item in crop_groups)
        frame_area = max(1, int(info.width) * int(info.height))
        crop_area_pct = round(crop_area_total / frame_area * 100, 3)

        return {
            "input_path": str(input_path),
            "engine": config.to_metadata(),
            "reference_time": round(ref_time, 2),
            "reference_path": str(reference_path),
            "mask_preview_path": str(mask_preview_path),
            "crop_preview_path": str(crop_preview_path) if crop_groups else None,
            "mask_runtime_shape": mask_runtime_shape,
            "mask_runtime_warning": mask_runtime_warning,
            "base_region_count": len(base_regions),
            "suggested_region_count": len(suggested_regions),
            "merged_region_count": len(merged_regions),
            "suggested_regions": suggested_regions,
            "merged_regions": merged_regions,
            "crop_groups": crop_groups,
            "crop_status_counts": crop_status_counts,
            "crop_area_total": crop_area_total,
            "crop_area_pct": crop_area_pct,
            "mask_coverage": stats["coverage"],
            "mask_bbox": stats["bbox"],
            "video_info": {
                "width": info.width,
                "height": info.height,
                "duration": info.duration,
                "fps": info.fps,
            },
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(f"Видео не найдено: {VIDEO_FILE}")
    if not REGIONS_FILE.exists():
        raise FileNotFoundError(f"Файл регионов не найден: {REGIONS_FILE}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    clip_path = create_clip(VIDEO_FILE, OUTPUT_ROOT)
    analysis = build_quality_analysis(
        input_path=clip_path,
        regions=load_regions(),
        engine_key=ENGINE,
        engine_options=ENGINE_OPTIONS,
        autodetect=AUTODETECT,
        run_dir=run_dir,
    )
    (run_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(json.dumps(
        {
            "run_dir": str(run_dir),
            "engine": analysis["engine"]["key"],
            "mask_coverage": analysis["mask_coverage"],
            "crop_groups": len(analysis["crop_groups"] or []),
            "crop_area_pct": analysis["crop_area_pct"],
            "risky_groups": (analysis.get("crop_status_counts") or {}).get("risky", 0),
            "empty_groups": (analysis.get("crop_status_counts") or {}).get("empty", 0),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
