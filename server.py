import asyncio
import concurrent.futures
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))
from services.video_info import get_video_info
from services.ai_engines import DEFAULT_AI_ENGINE, get_engine_config, list_engine_metadata
from services.iopaint_runner import (
    generate_mask, extract_frames_range,
    get_total_frames, reassemble_video,
    run_iopaint_parallel, compose_inpainted_frames, thin_frames,
    write_iopaint_config, get_worker_count, extract_reference_frame,
    build_mask_preview, get_mask_stats,
)
from services.propainter_runner import ensure_propainter_available, run_propainter_pipeline
from services.watermark_detector import dedupe_regions, detect_repeated_regions

BASE = Path(__file__).parent
TEMP = BASE / "temp_web"          # use project dir to avoid 8.3-path issues
TEMP.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_active: dict[str, subprocess.Popen] = {}

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "500"))
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "4"))

# ── Auto-stop on inactivity ─────────────────────────────────────────────────
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "30")) * 60  # seconds
_last_activity = time.monotonic()


def _touch_activity():
    global _last_activity
    _last_activity = time.monotonic()


async def _idle_watchdog():
    """Stop the RunPod pod if no activity for IDLE_TIMEOUT seconds."""
    while True:
        await asyncio.sleep(60)
        idle = time.monotonic() - _last_activity
        if idle >= IDLE_TIMEOUT:
            pod_id = os.environ.get("RUNPOD_POD_ID", "")
            if not pod_id:
                # Try to get pod ID from runpodctl
                try:
                    r = subprocess.run(["runpodctl", "get", "pod"], capture_output=True, text=True, timeout=5)
                    # Skip if we can't determine pod ID
                except Exception:
                    pass
                continue
            print(f"[auto-stop] No activity for {IDLE_TIMEOUT//60} min — stopping pod {pod_id}")
            try:
                subprocess.run(
                    ["runpodctl", "stop", "pod", pod_id],
                    capture_output=True, timeout=10,
                )
            except Exception:
                pass
            break


def _cleanup_work_dir(work_dir: Path):
    """Remove AI work directory (frames, inpainted, mask)."""
    shutil.rmtree(work_dir, ignore_errors=True)


def _cleanup_stale_temp():
    """Remove stale temp files and abandoned AI work dirs older than 1 hour."""
    cutoff = time.time() - 3600
    for path in TEMP.iterdir():
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            if path.is_dir() and path.name.startswith("ai_"):
                shutil.rmtree(path, ignore_errors=True)
                continue
            if not path.is_file():
                continue

            is_known_output = (
                path.match("output_*.mp4")
                or path.match("frame_*.jpg")
                or path.match("reference_*.jpg")
                or path.match("mask_preview_*.png")
            )
            is_uploaded_source = re.fullmatch(r"[0-9a-fA-F-]{36}", path.stem) is not None
            if is_known_output or is_uploaded_source:
                path.unlink()
        except Exception:
            pass


# ── Batch queue ──────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"
    CANCELLED  = "cancelled"


@dataclass
class QueueJob:
    job_id:       str
    params:       dict
    status:       JobStatus      = JobStatus.QUEUED
    progress:     int            = 0
    log:          list[str]      = field(default_factory=list)
    download_url: Optional[str]  = None
    error:        Optional[str]  = None


_queue: list[QueueJob] = []
_queue_worker_task: Optional[asyncio.Task] = None
_runtimes: dict[str, "JobRuntime"] = {}


@dataclass
class JobRuntime:
    job_id: str
    work_dir: Optional[Path] = None
    cancel_requested: bool = False
    processes: list[subprocess.Popen] = field(default_factory=list)


class JobCancelled(RuntimeError):
    pass


def _ensure_runtime(job_id: str) -> JobRuntime:
    runtime = _runtimes.get(job_id)
    if runtime is None:
        runtime = JobRuntime(job_id=job_id)
        _runtimes[job_id] = runtime
    return runtime


def _register_runtime_process(job_id: str, proc: subprocess.Popen):
    _ensure_runtime(job_id).processes.append(proc)


def _terminate_process(proc: subprocess.Popen):
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


def _terminate_runtime_processes(job_id: str):
    runtime = _runtimes.get(job_id)
    if not runtime:
        return
    for proc in runtime.processes:
        _terminate_process(proc)


def _clear_runtime(job_id: str):
    _terminate_runtime_processes(job_id)
    _runtimes.pop(job_id, None)


def _cancel_job_runtime(job_id: str):
    if proc := _active.pop(job_id, None):
        _terminate_process(proc)
    if runtime := _runtimes.get(job_id):
        runtime.cancel_requested = True
    _terminate_runtime_processes(job_id)


def _raise_if_cancelled(job_id: str):
    runtime = _runtimes.get(job_id)
    if runtime and runtime.cancel_requested:
        raise JobCancelled("Отменено пользователем")


def _reference_time(duration: float) -> float:
    return min(5.0, max(0.0, duration * 0.25))


def _resolve_worker_count(engine_key: str, device: str) -> int:
    config = get_engine_config(engine_key)
    if config.worker_count_override > 0 and device == "cuda":
        return config.worker_count_override
    return get_worker_count(device)


def _normalize_regions(raw_regions: list[dict] | None) -> list[dict]:
    normalized = []
    for region in raw_regions or []:
        try:
            normalized.append(
                {
                    "x": int(region["x"]),
                    "y": int(region["y"]),
                    "w": int(region["w"]),
                    "h": int(region["h"]),
                }
            )
        except Exception:
            continue
    return dedupe_regions(normalized)


def _build_quality_analysis(body: dict) -> dict:
    input_path = body.get("path")
    if not input_path:
        raise RuntimeError("Не передан путь к видео")

    info = get_video_info(input_path)
    width = int(body.get("width") or (info.width if info else 0) or 0)
    height = int(body.get("height") or (info.height if info else 0) or 0)
    duration = float(body.get("duration") or (info.duration if info else 0) or 0.0)
    if width <= 0 or height <= 0:
        raise RuntimeError("Не удалось определить размеры видео")

    base_regions = _normalize_regions(body.get("regions"))
    if not base_regions:
        raise RuntimeError("Для quality analysis нужен хотя бы один регион")

    engine_key = body.get("engine") or DEFAULT_AI_ENGINE
    config = get_engine_config(engine_key)
    analysis_id = str(uuid.uuid4())
    work_dir = TEMP / f"ai_preview_{analysis_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    reference_path = TEMP / f"reference_{analysis_id}.jpg"
    preview_path = TEMP / f"mask_preview_{analysis_id}.png"
    mask_path = work_dir / "mask.png"
    reference_time = _reference_time(duration)
    suggested_regions: list[dict] = []

    try:
        extract_reference_frame(input_path, reference_path, time_sec=reference_time)

        if body.get("autodetect"):
            autodetect_passes = max(1, int(body.get("autodetect_passes") or (3 if len(base_regions) >= 3 else 1)))
            autodetect_limit = max(
                len(base_regions),
                int(body.get("autodetect_limit") or (12 if len(base_regions) >= 3 else min(18, len(base_regions) + 12))),
            )
            current_regions = list(base_regions)
            for _ in range(autodetect_passes):
                detected = detect_repeated_regions(str(reference_path), current_regions)
                merged = dedupe_regions(detected)
                if len(merged) <= len(current_regions):
                    break
                current_regions = merged[:autodetect_limit]
                if len(current_regions) >= autodetect_limit:
                    break

            for candidate in current_regions:
                merged = dedupe_regions(base_regions + suggested_regions + [candidate])
                if len(merged) > len(base_regions) + len(suggested_regions):
                    suggested_regions.append(candidate)

        merged_regions = dedupe_regions(base_regions + suggested_regions)
        generate_mask(
            width,
            height,
            merged_regions,
            mask_path,
            padding=config.mask_padding,
            dilate=config.mask_dilate,
            reference_frame_path=reference_path if config.refine_mask else None,
        )
        build_mask_preview(reference_path, mask_path, preview_path)
        stats = get_mask_stats(mask_path)
        return {
            "engine": config.to_metadata(),
            "reference_time": round(reference_time, 2),
            "reference_url": f"/api/file/{reference_path.name}",
            "mask_preview_url": f"/api/file/{preview_path.name}",
            "base_region_count": len(base_regions),
            "suggested_region_count": len(suggested_regions),
            "merged_region_count": len(merged_regions),
            "suggested_regions": suggested_regions,
            "merged_regions": merged_regions,
            "mask_coverage": stats["coverage"],
            "mask_bbox": stats["bbox"],
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.on_event("startup")
async def start_queue_worker():
    global _queue_worker_task
    # Clean up leftovers from previous runs
    for d in TEMP.glob("ai_*"):
        shutil.rmtree(d, ignore_errors=True)
    _cleanup_stale_temp()
    _queue_worker_task = asyncio.create_task(queue_worker())
    asyncio.create_task(_idle_watchdog())


async def queue_worker():
    while True:
        job = next((j for j in _queue if j.status == JobStatus.QUEUED), None)
        if job:
            job.status = JobStatus.PROCESSING
            out_path = TEMP / f"output_{job.job_id}.mp4"
            try:
                await _run_job(job, out_path)
                if job.status != JobStatus.CANCELLED:
                    job.status = JobStatus.DONE
                    job.download_url = f"/api/download/{out_path.name}"
                    job.progress = 100
            except JobCancelled:
                job.status = JobStatus.CANCELLED
            except Exception as e:
                if job.status != JobStatus.CANCELLED:
                    job.status = JobStatus.ERROR
                    job.error = str(e)
            finally:
                _cleanup_stale_temp()
        await asyncio.sleep(1)


async def _run_job(job: QueueJob, out_path: Path):
    params = job.params
    regions = params.get("regions", [])
    mode = params.get("mode", "delogo")

    def _log(line: str):
        job.log.append(line)
        print(f"[job {job.job_id}] {line}", flush=True)

    loop = asyncio.get_event_loop()

    if mode == "ai":
        queue = _start_ai_pipeline(loop, params, out_path, job.job_id)

        while True:
            kind, value = await queue.get()
            if kind == "log":
                _log(value)
            elif kind == "progress":
                job.progress = value
            elif kind == "done":
                break
            elif kind == "cancelled":
                raise JobCancelled("Отменено пользователем")
            elif kind == "error":
                raise RuntimeError(value)

    else:  # delogo
        vf = ",".join(
            f"delogo=x={int(r['x'])}:y={int(r['y'])}:w={int(r['w'])}:h={int(r['h'])}"
            for r in regions
        )
        cmd = [
            "ffmpeg", "-y", "-i", params["path"],
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            str(out_path),
        ]
        _log(" ".join(cmd))
        duration = float(params.get("duration", 0))
        time_re = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")

        queue: asyncio.Queue = asyncio.Queue()

        def run_ffmpeg():
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            _active[job.job_id] = proc
            loop.call_soon_threadsafe(queue.put_nowait, ("started", None))
            for line in proc.stdout:
                loop.call_soon_threadsafe(queue.put_nowait, ("line", line.rstrip()))
            proc.wait()
            _active.pop(job.job_id, None)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", proc.returncode))

        _executor.submit(run_ffmpeg)
        returncode = -1
        while True:
            kind, value = await queue.get()
            if kind == "line":
                if value:
                    _log(value)
                if duration > 0 and (m := time_re.search(value or "")):
                    elapsed = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
                    job.progress = min(int(elapsed / duration * 100), 99)
            elif kind == "done":
                returncode = value
                break

        if returncode != 0:
            raise RuntimeError(f"FFmpeg завершился с кодом {returncode}")


def _prepare_mask_assets(
    *,
    input_path: str,
    width: int,
    height: int,
    duration: float,
    regions: list[dict],
    work_dir: Path,
    engine_key: str,
    register_process,
    emit_log,
):
    config = get_engine_config(engine_key)
    mask_path = work_dir / "mask.png"
    reference_frame_path = None
    reference_time = _reference_time(duration)

    if config.refine_mask:
        reference_frame_path = work_dir / "reference.jpg"
        extract_reference_frame(
            input_path,
            reference_frame_path,
            time_sec=reference_time,
            register_process=register_process,
        )
        emit_log(f"Mask reference frame: {reference_time:.1f}s")

    generate_mask(
        width,
        height,
        regions,
        mask_path,
        padding=config.mask_padding,
        dilate=config.mask_dilate,
        reference_frame_path=reference_frame_path,
    )
    return mask_path, reference_frame_path


def _run_lama_pipeline(
    params: dict,
    out_path: Path,
    job_id: str,
    emit_log,
    emit_progress,
    engine_key: str,
):
    fps = params.get("fps", 25) or 25
    width = params.get("width", 1920) or 1920
    height = params.get("height", 1080) or 1080
    duration = float(params.get("duration", 0))
    device = params.get("device", "cpu")
    input_path = params["path"]
    work_dir = TEMP / f"ai_{job_id}"
    runtime = _ensure_runtime(job_id)
    runtime.work_dir = work_dir
    runtime.cancel_requested = False

    total_frames = get_total_frames(duration, fps)
    if total_frames <= 0:
        raise RuntimeError("Не удалось определить количество кадров")

    config = get_engine_config(engine_key)
    skip = max(1, config.skip)
    worker_count = _resolve_worker_count(engine_key, device)
    actual_total_frames = 0
    pipeline_started_at = time.perf_counter()

    try:
        _raise_if_cancelled(job_id)
        to_process = math.ceil(total_frames / skip)
        emit_log(
            f"Engine: {config.label}. Кадров: {total_frames}. "
            f"Skip={skip} → обработка {to_process}. Параллельность: x{worker_count}."
        )

        mask_path, reference_frame_path = _prepare_mask_assets(
            input_path=input_path,
            width=width,
            height=height,
            duration=duration,
            regions=params.get("regions", []),
            work_dir=work_dir,
            engine_key=engine_key,
            register_process=lambda proc: _register_runtime_process(job_id, proc),
            emit_log=emit_log,
        )
        mask_stats = get_mask_stats(mask_path)
        emit_log(f"Mask coverage: {mask_stats['coverage']:.3f}%")
        emit_progress(5)

        iopaint_config_path = write_iopaint_config(
            work_dir / "iopaint_config.json",
            hd_strategy=config.hd_strategy,
            resize_limit=config.resize_limit,
        )
        emit_log(
            f"IOPaint config: {config.hd_strategy} "
            f"{json.loads(iopaint_config_path.read_text(encoding='utf-8'))['hd_strategy_resize_limit']}"
        )

        if reference_frame_path:
            preview_path = TEMP / f"mask_preview_{job_id}.png"
            build_mask_preview(reference_frame_path, mask_path, preview_path)

        all_inpainted = work_dir / "all_inpainted"
        all_inpainted.mkdir(parents=True, exist_ok=True)
        frames_dir = work_dir / "frames"
        kept_dir = work_dir / "kept"
        inpainted_dir = work_dir / "inpainted"

        processed = 0
        for batch_start in range(0, total_frames, BATCH_SIZE):
            expected_batch_count = min(BATCH_SIZE, total_frames - batch_start)
            batch_num = batch_start // BATCH_SIZE + 1

            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            if kept_dir.exists():
                shutil.rmtree(kept_dir)
            if inpainted_dir.exists():
                shutil.rmtree(inpainted_dir)

            _raise_if_cancelled(job_id)
            emit_log(f"Батч {batch_num}: извлечение до {expected_batch_count} кадров...")
            extract_started_at = time.perf_counter()
            extract_frames_range(
                input_path,
                frames_dir,
                batch_start,
                expected_batch_count,
                fps,
                register_process=lambda proc: _register_runtime_process(job_id, proc),
            )
            emit_log(f"  Extract: {time.perf_counter() - extract_started_at:.1f}s")
            batch_frames = sorted(frames_dir.glob("*.png"))
            actual_batch_count = len(batch_frames)
            if actual_batch_count == 0:
                emit_log(f"  Батч {batch_num}: ffmpeg не вернул кадров, остановка на {processed} кадрах")
                break
            actual_total_frames = max(actual_total_frames, batch_start + actual_batch_count)
            if actual_batch_count != expected_batch_count:
                emit_log(
                    f"  Батч {batch_num}: извлечено {actual_batch_count} вместо "
                    f"ожидаемых {expected_batch_count}"
                )

            _raise_if_cancelled(job_id)
            thin_started_at = time.perf_counter()
            kept = thin_frames(frames_dir, skip=skip, kept_dir=kept_dir)
            emit_log(f"  Прореживание: {actual_batch_count} → {kept} кадров")
            emit_log(f"  Thin: {time.perf_counter() - thin_started_at:.1f}s")

            _raise_if_cancelled(job_id)
            emit_log(f"Батч {batch_num}: IOPaint x{worker_count} ({device})...")
            iopaint_started_at = time.perf_counter()
            success, err = run_iopaint_parallel(
                kept_dir,
                mask_path,
                inpainted_dir,
                device=device,
                workers=worker_count,
                register_process=lambda proc: _register_runtime_process(job_id, proc),
                is_cancelled=lambda: _ensure_runtime(job_id).cancel_requested,
                config_path=iopaint_config_path,
            )
            _raise_if_cancelled(job_id)
            if not success:
                raise RuntimeError(err or "IOPaint завершился с ошибкой")
            emit_log(f"  IOPaint: {time.perf_counter() - iopaint_started_at:.1f}s")

            _raise_if_cancelled(job_id)
            emit_log(f"Батч {batch_num}: композиция финальных кадров...")
            compose_started_at = time.perf_counter()
            written = compose_inpainted_frames(
                frames_dir,
                inpainted_dir,
                all_inpainted,
                mask_path,
                feather_radius=config.feather_radius,
                output_suffix=config.output_suffix,
                output_quality=config.output_quality,
                blend_skipped=config.blend_skipped,
            )
            emit_log(f"  Compose: {time.perf_counter() - compose_started_at:.1f}s")
            if written != actual_batch_count:
                raise RuntimeError(
                    f"Не удалось собрать батч {batch_num}: "
                    f"ожидалось {actual_batch_count} кадров, получено {written}"
                )

            processed += actual_batch_count
            emit_progress(10 + int(processed / max(total_frames, 1) * 75))
            done_so_far = sum(1 for path in all_inpainted.iterdir() if path.is_file())
            emit_log(
                f"  Батч {batch_num} собран ({done_so_far}/"
                f"{actual_total_frames or total_frames} кадров)"
            )

        if actual_total_frames <= 0:
            raise RuntimeError("Не удалось извлечь кадры из видео")

        emit_log("Сборка видео...")
        emit_progress(92)
        _raise_if_cancelled(job_id)
        reassemble_started_at = time.perf_counter()
        reassemble_video(
            all_inpainted,
            input_path,
            out_path,
            fps,
            register_process=lambda proc: _register_runtime_process(job_id, proc),
        )
        emit_log(f"  Reassemble: {time.perf_counter() - reassemble_started_at:.1f}s")
        emit_log(f"Готово за {time.perf_counter() - pipeline_started_at:.1f}s")
    except subprocess.CalledProcessError as e:
        _raise_if_cancelled(job_id)
        details = (e.stderr or e.output or "").strip()
        raise RuntimeError(details or str(e)) from e
    finally:
        _cleanup_work_dir(work_dir)
        _clear_runtime(job_id)


def _run_propainter_quality(
    params: dict,
    out_path: Path,
    job_id: str,
    emit_log,
    emit_progress,
    engine_key: str,
):
    work_dir = TEMP / f"ai_{job_id}"
    runtime = _ensure_runtime(job_id)
    runtime.work_dir = work_dir
    runtime.cancel_requested = False
    config = get_engine_config(engine_key)
    started_at = time.perf_counter()

    try:
        ensure_propainter_available()
        emit_log(f"Engine: {config.label}. Video-aware quality mode.")
        run_propainter_pipeline(
            params["path"],
            params.get("regions", []),
            out_path,
            work_dir,
            config,
            emit_log=emit_log,
            emit_progress=emit_progress,
            register_process=lambda proc: _register_runtime_process(job_id, proc),
            is_cancelled=lambda: _ensure_runtime(job_id).cancel_requested,
        )
        emit_log(f"Готово за {time.perf_counter() - started_at:.1f}s")
    except subprocess.CalledProcessError as e:
        details = (e.stderr or e.output or "").strip()
        raise RuntimeError(details or str(e)) from e
    finally:
        _cleanup_work_dir(work_dir)
        _clear_runtime(job_id)


def _run_ai_pipeline(
    params: dict,
    out_path: Path,
    job_id: str,
    emit_log,
    emit_progress,
):
    engine_key = params.get("engine") or DEFAULT_AI_ENGINE
    config = get_engine_config(engine_key)
    if config.family == "propainter":
        return _run_propainter_quality(params, out_path, job_id, emit_log, emit_progress, engine_key)
    return _run_lama_pipeline(params, out_path, job_id, emit_log, emit_progress, engine_key)


def _start_ai_pipeline(loop, params: dict, out_path: Path, job_id: str) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            _run_ai_pipeline(
                params,
                out_path,
                job_id,
                lambda msg: loop.call_soon_threadsafe(queue.put_nowait, ("log", msg)),
                lambda p: loop.call_soon_threadsafe(queue.put_nowait, ("progress", p)),
            )
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except JobCancelled:
            loop.call_soon_threadsafe(queue.put_nowait, ("cancelled", None))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

    _executor.submit(run)
    return queue


@app.get("/")
async def index():
    return FileResponse(str(BASE / "static" / "index.html"))


@app.get("/api/ai/engines")
async def ai_engines():
    return list_engine_metadata()


@app.post("/api/quality/analyze")
async def quality_analyze(body: dict):
    _touch_activity()
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(_executor, lambda: _build_quality_analysis(body))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/queue")
async def queue_add(body: dict):
    _touch_activity()
    job = QueueJob(job_id=str(uuid.uuid4()), params=body)
    _queue.append(job)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/queue")
async def queue_list():
    return [
        {
            "job_id":       j.job_id,
            "name":         j.params.get("name", j.job_id),
            "status":       j.status,
            "progress":     j.progress,
            "download_url": j.download_url,
            "error":        j.error,
        }
        for j in _queue
    ]


@app.delete("/api/queue/{job_id}")
async def queue_delete(job_id: str):
    global _queue
    for j in _queue:
        if j.job_id == job_id:
            if j.status == JobStatus.PROCESSING:
                _cancel_job_runtime(job_id)
                j.status = JobStatus.CANCELLED
            elif j.status == JobStatus.QUEUED:
                j.status = JobStatus.CANCELLED
            break
    _queue = [j for j in _queue if j.job_id != job_id or j.status not in (JobStatus.CANCELLED, JobStatus.DONE, JobStatus.ERROR)]
    return {"ok": True}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    _touch_activity()
    _cleanup_stale_temp()
    ext = Path(file.filename).suffix.lower() or ".mp4"
    dest = TEMP / f"{uuid.uuid4()}{ext}"
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            await f.write(chunk)
    return _video_meta(str(dest), file.filename)


@app.get("/api/info")
async def file_info(path: str = Query(...)):
    p = Path(path)
    if not p.exists():
        return JSONResponse({"error": "Файл не найден"}, status_code=404)
    return _video_meta(path, p.name)


@app.get("/api/frame")
async def extract_frame(path: str = Query(...), time: float = 1.0):
    _touch_activity()
    out = TEMP / f"frame_{uuid.uuid4()}.jpg"
    loop = asyncio.get_event_loop()

    def _run():
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(time), "-i", path,
             "-frames:v", "1", "-q:v", "2", str(out)],
            capture_output=True, timeout=15,
        )
        return result.returncode

    rc = await loop.run_in_executor(_executor, _run)
    if rc == 0 and out.exists():
        return FileResponse(str(out), media_type="image/jpeg")
    return JSONResponse({"error": "Не удалось извлечь кадр"}, status_code=500)


@app.websocket("/ws/process")
async def process_video(ws: WebSocket):
    _touch_activity()
    await ws.accept()
    job_id = str(uuid.uuid4())
    try:
        params = await ws.receive_json()
        out_path = TEMP / f"output_{job_id}.mp4"
        mode = params.get("mode", "delogo")
        regions = params.get("regions", [])

        if not regions:
            await ws.send_json({"type": "error", "data": "Не указаны регионы для удаления"})
            return

        device = params.get("device", "cpu")
        if mode == "ai":
            await _process_ai(ws, params, regions, out_path, job_id, device=device)
        else:
            await _process_delogo(ws, params, regions, out_path, job_id)

    except WebSocketDisconnect:
        _cancel_job_runtime(job_id)
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass


async def _process_delogo(ws, params, regions, out_path, job_id):
    vf = ",".join(
        f"delogo=x={int(r['x'])}:y={int(r['y'])}:w={int(r['w'])}:h={int(r['h'])}"
        for r in regions
    )
    cmd = [
        "ffmpeg", "-y", "-i", params["path"],
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        str(out_path),
    ]

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    duration = float(params.get("duration", 0))

    def run_ffmpeg():
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        _active[job_id] = proc
        loop.call_soon_threadsafe(queue.put_nowait, ("started", None))
        for line in proc.stdout:
            loop.call_soon_threadsafe(queue.put_nowait, ("line", line.rstrip()))
        proc.wait()
        _active.pop(job_id, None)
        loop.call_soon_threadsafe(queue.put_nowait, ("done", proc.returncode))

    _executor.submit(run_ffmpeg)
    await ws.send_json({"type": "job_id", "data": job_id})
    await ws.send_json({"type": "log", "data": "Удаление водяного знака..."})

    time_re = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
    returncode = -1

    while True:
        kind, value = await queue.get()
        if kind == "started":
            pass
        elif kind == "line":
            # Only send progress updates, not raw FFmpeg output
            if duration > 0 and (m := time_re.search(value or "")):
                elapsed = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
                pct = min(int(elapsed / duration * 100), 99)
                await ws.send_json({"type": "progress", "data": pct})
        elif kind == "done":
            returncode = value
            break

    if returncode == 0:
        await ws.send_json({"type": "done", "success": True, "download_url": f"/api/download/{out_path.name}"})
    else:
        await ws.send_json({"type": "done", "success": False, "message": "Не удалось обработать видео. Проверьте корректность файла."})


async def _process_ai(ws, params, regions, out_path, job_id, device="cpu"):
    loop = asyncio.get_event_loop()
    await ws.send_json({"type": "job_id", "data": job_id})
    queue = _start_ai_pipeline(loop, params, out_path, job_id)

    while True:
        kind, value = await queue.get()
        if kind == "log":
            await ws.send_json({"type": "log", "data": value})
        elif kind == "progress":
            await ws.send_json({"type": "progress", "data": value})
        elif kind == "done":
            break
        elif kind == "cancelled":
            await ws.send_json({"type": "done", "success": False, "message": "Отменено пользователем"})
            return
        elif kind == "error":
            await ws.send_json({"type": "done", "success": False, "message": value})
            return

    await ws.send_json({"type": "progress", "data": 100})
    await ws.send_json({"type": "done", "success": True, "download_url": f"/api/download/{out_path.name}"})


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    _cancel_job_runtime(job_id)
    return {"ok": True}


@app.get("/api/download/{filename}")
async def download(filename: str):
    path = TEMP / filename
    if not path.exists():
        return JSONResponse({"error": "Файл не найден"}, status_code=404)
    return FileResponse(str(path), filename=filename, media_type="video/mp4")


@app.get("/api/file/{filename}")
async def temp_file(filename: str):
    path = TEMP / filename
    if not path.exists():
        return JSONResponse({"error": "Файл не найден"}, status_code=404)
    media_type, _ = mimetypes.guess_type(path.name)
    return FileResponse(str(path), filename=path.name, media_type=media_type or "application/octet-stream")


@app.get("/health")
async def health():
    result = {"status": "ok", "gpu_available": False, "gpu_name": None}
    try:
        import torch
        result["gpu_available"] = torch.cuda.is_available()
        if result["gpu_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return result


def _video_meta(path: str, name: str) -> dict:
    info = get_video_info(path)
    return {
        "path": path,
        "name": name,
        "width": info.width if info else 0,
        "height": info.height if info else 0,
        "duration": info.duration if info else 0,
        "fps": info.fps if info else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
