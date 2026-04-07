import json
import os
import subprocess
import time
from pathlib import Path

import requests


BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VIDEO_FILE = Path(os.environ.get("VIDEO_FILE", r"C:\Users\Пользователь\Desktop\watermark\Араб.mp4"))
SERVER_VIDEO = os.environ.get("SERVER_VIDEO", "").strip()
CLIP_SERVER_VIDEO = os.environ.get("CLIP_SERVER_VIDEO", "").strip()
CLIP_OFFSET = float(os.environ.get("BENCH_OFFSET", "0"))
CLIP_DURATION = float(os.environ.get("BENCH_DURATION", "15"))
ENGINES = [item.strip() for item in os.environ.get("BENCH_ENGINES", "lama_fast,propainter_quality").split(",") if item.strip()]
ENGINE_OPTIONS = json.loads(os.environ.get("ENGINE_OPTIONS_JSON", "{}") or "{}")
REGIONS_PATH = Path(os.environ.get("REGIONS_FILE", str(Path(__file__).resolve().parents[1] / "assets" / "arab_watermark_regions.json")))
OUTPUT_ROOT = Path(os.environ.get("BENCH_OUTPUT_DIR", str(Path(__file__).resolve().parents[1] / "output" / "quality_benchmark")))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def log(line: str):
    print(line, flush=True)


def run(command: list[str]):
    subprocess.run(command, check=True)


def create_local_clip() -> Path:
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {VIDEO_FILE}")
    clip_path = OUTPUT_ROOT / f"bench_{int(CLIP_DURATION)}s.mp4"
    if clip_path.exists():
        return clip_path
    log(f"[clip] Создаю {CLIP_DURATION:.0f}s клип: {clip_path.name}")
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(CLIP_OFFSET),
            "-i",
            str(VIDEO_FILE),
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


def upload_clip(clip_path: Path) -> dict:
    with clip_path.open("rb") as fh:
        res = requests.post(f"{BASE_URL}/api/upload", files={"file": fh}, timeout=600)
    res.raise_for_status()
    return res.json()


def get_server_clip_info() -> dict:
    remote_path = CLIP_SERVER_VIDEO or SERVER_VIDEO
    if not remote_path:
        clip_path = create_local_clip()
        log(f"[upload] Загружаю клип {clip_path.name}")
        return upload_clip(clip_path)

    if SERVER_VIDEO and not CLIP_SERVER_VIDEO:
        raise RuntimeError(
            "Для server-side benchmark укажите CLIP_SERVER_VIDEO с уже подготовленным 15s файлом "
            "или уберите SERVER_VIDEO, чтобы скрипт сам создал локальный клип и загрузил его."
        )

    if CLIP_SERVER_VIDEO and SERVER_VIDEO and Path(SERVER_VIDEO).exists() and not Path(CLIP_SERVER_VIDEO).exists():
        log(f"[clip] Создаю server-side клип: {CLIP_SERVER_VIDEO}")
        create_server_clip(Path(SERVER_VIDEO), Path(CLIP_SERVER_VIDEO))

    info = fetch_video_info(remote_path)
    if is_valid_info(info):
        return info

    if CLIP_SERVER_VIDEO and SERVER_VIDEO and Path(SERVER_VIDEO).exists():
        log(f"[clip] Пересоздаю server-side клип: {CLIP_SERVER_VIDEO}")
        create_server_clip(Path(SERVER_VIDEO), Path(CLIP_SERVER_VIDEO))
        info = fetch_video_info(CLIP_SERVER_VIDEO)
        if is_valid_info(info):
            return info

    raise RuntimeError(f"Некорректные метаданные клипа {remote_path}: {info}")


def fetch_video_info(remote_path: str) -> dict:
    res = requests.get(f"{BASE_URL}/api/info", params={"path": remote_path}, timeout=120)
    res.raise_for_status()
    return res.json()


def is_valid_info(info: dict) -> bool:
    return (
        int(info.get("width") or 0) > 0
        and int(info.get("height") or 0) > 0
        and float(info.get("duration") or 0) > 0
        and float(info.get("fps") or 0) > 0
    )


def create_server_clip(source_path: Path, clip_path: Path):
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(CLIP_OFFSET),
            "-i",
            str(source_path),
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


def load_regions() -> list[dict]:
    return json.loads(REGIONS_PATH.read_text(encoding="utf-8"))


def get_engine_options(engine: str) -> dict:
    if not isinstance(ENGINE_OPTIONS, dict):
        return {}
    options = ENGINE_OPTIONS.get(engine, {})
    return options if isinstance(options, dict) else {}


def fetch_quality_analysis(info: dict, engine: str, regions: list[dict], out_dir: Path) -> dict:
    engine_options = get_engine_options(engine)
    res = requests.post(
        f"{BASE_URL}/api/quality/analyze",
        json={
            "path": info["path"],
            "duration": info["duration"],
            "width": info["width"],
            "height": info["height"],
            "regions": regions,
            "engine": engine,
            "engine_options": engine_options,
            "autodetect": False,
        },
        timeout=600,
    )
    if not res.ok:
        log(f"[analysis:error] {res.status_code}: {res.text[:2000]}")
    res.raise_for_status()
    data = res.json()
    (out_dir / "analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    for key, filename in [("reference_url", "reference.jpg"), ("mask_preview_url", "mask_preview.png")]:
        url = data.get(key)
        if not url:
            continue
        resp = requests.get(f"{BASE_URL}{url}", timeout=300)
        resp.raise_for_status()
        (out_dir / filename).write_bytes(resp.content)
    return data


def enqueue_job(info: dict, engine: str, regions: list[dict]) -> str:
    engine_options = get_engine_options(engine)
    payload = {
        "path": info["path"],
        "name": f"{engine}_{info['name']}",
        "regions": regions,
        "duration": info["duration"],
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "mode": "ai",
        "device": "cuda",
        "engine": engine,
        "engine_options": engine_options,
    }
    res = requests.post(f"{BASE_URL}/api/queue", json=payload, timeout=120)
    if not res.ok:
        log(f"[queue:error] {res.status_code}: {res.text[:2000]}")
    res.raise_for_status()
    return res.json()["job_id"]


def poll_job(job_id: str) -> dict:
    started_at = time.time()
    last_progress = None
    while True:
        res = requests.get(f"{BASE_URL}/api/queue", timeout=120)
        res.raise_for_status()
        jobs = res.json()
        job = next((item for item in jobs if item["job_id"] == job_id), None)
        if not job:
            raise RuntimeError(f"Задание {job_id} исчезло из очереди")

        progress = job.get("progress")
        status = job.get("status")
        if progress != last_progress:
            elapsed = int(time.time() - started_at)
            log(f"[job {job_id[:8]}] {status} {progress}% ({elapsed}s)")
            last_progress = progress

        if status in {"done", "error", "cancelled"}:
            job["elapsed_seconds"] = round(time.time() - started_at, 1)
            return job
        time.sleep(5)


def download_result(download_url: str, out_path: Path):
    res = requests.get(f"{BASE_URL}{download_url}", timeout=3600)
    res.raise_for_status()
    out_path.write_bytes(res.content)


def extract_preview(video_path: Path, out_path: Path, at_time: float = 5.0):
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(at_time),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_path),
        ]
    )


def main():
    info = get_server_clip_info()
    regions = load_regions()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "base_url": BASE_URL,
        "input": info,
        "clip_duration": CLIP_DURATION,
        "engines": [],
    }

    for engine in ENGINES:
        engine_dir = run_dir / engine
        engine_dir.mkdir(parents=True, exist_ok=True)
        log(f"\n=== {engine} ===")
        analysis = fetch_quality_analysis(info, engine, regions, engine_dir)
        effective_regions = analysis.get("merged_regions") or regions
        job_id = enqueue_job(info, engine, effective_regions)
        job = poll_job(job_id)

        entry = {
            "engine": engine,
            "engine_options": get_engine_options(engine),
            "job_id": job_id,
            "status": job["status"],
            "elapsed_seconds": job["elapsed_seconds"],
            "progress": job.get("progress", 0),
            "download_url": job.get("download_url"),
            "error": job.get("error"),
            "mask_coverage": analysis.get("mask_coverage"),
            "region_count": len(effective_regions),
        }

        if job["status"] == "done" and job.get("download_url"):
            video_path = engine_dir / f"{engine}.mp4"
            frame_path = engine_dir / f"{engine}_5s.jpg"
            download_result(job["download_url"], video_path)
            extract_preview(video_path, frame_path)
            entry["video_path"] = str(video_path)
            entry["frame_path"] = str(frame_path)

        manifest["engines"].append(entry)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"\nГотово: {run_dir}")


if __name__ == "__main__":
    main()
