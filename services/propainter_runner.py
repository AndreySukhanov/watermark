import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from services.ai_engines import AIEngineConfig
from services.iopaint_runner import extract_reference_frame, generate_mask
from services.video_info import get_video_info

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

    propainter_dir = ensure_propainter_available()
    info = get_video_info(str(input_video))
    if info is None:
        raise RuntimeError(f"Не удалось прочитать видео: {input_video}")

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
    generate_mask(
        info.width,
        info.height,
        regions,
        mask_path,
        padding=engine_config.mask_padding,
        dilate=engine_config.mask_dilate,
        reference_frame_path=reference_frame_path,
    )

    emit_log("ProPainter: извлечение исходных кадров...")
    extract_started = time.perf_counter()
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
    emit_log(f"  Extract frames: {time.perf_counter() - extract_started:.1f}s")
    emit_progress(16)

    save_fps = max(1, int(round(info.fps or 25.0)))
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
        str(engine_config.propainter_width),
        "--height",
        str(engine_config.propainter_height),
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
        f"{engine_config.propainter_width}x{engine_config.propainter_height} "
        f"subvideo={engine_config.propainter_subvideo_length}"
    )
    infer_started = time.perf_counter()
    _run_streaming_command(
        infer_cmd,
        cwd=propainter_dir,
        register_process=register_process,
        is_cancelled=is_cancelled,
        emit_log=emit_log,
        emit_progress=emit_progress,
        progress_span=(20, 88),
    )
    emit_log(f"  Inference: {time.perf_counter() - infer_started:.1f}s")

    propainter_video = output_root / frames_dir.name / "inpaint_out.mp4"
    if not propainter_video.exists():
        raise RuntimeError(f"Не найден результат ProPainter: {propainter_video}")

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
