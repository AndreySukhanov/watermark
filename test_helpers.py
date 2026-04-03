import os
from pathlib import Path


BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VIDEO_FILE = Path(
    os.environ.get("VIDEO_FILE", r"C:\Users\Пользователь\Desktop\watermark\Араб.mp4")
)
SERVER_VIDEO = os.environ.get("SERVER_VIDEO", "").strip()
SHOTS_DIR = Path(os.environ.get("SHOTS_DIR", "output/playwright"))
SHOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_shot(page, name: str):
    path = SHOTS_DIR / name
    page.screenshot(path=str(path), full_page=True)
    return path


def ensure_video_file():
    if SERVER_VIDEO:
        return SERVER_VIDEO
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(f"Файл не найден: {VIDEO_FILE}")
    return str(VIDEO_FILE)


def open_video_in_ui(page):
    if SERVER_VIDEO:
        page.locator("#tab-path").click()
        page.locator("#local-path").fill(SERVER_VIDEO)
        page.locator("button", has_text="Открыть").click()
        return {"source": "server_path", "value": SERVER_VIDEO}

    video_path = ensure_video_file()
    page.locator("#tab-upload").click()
    page.set_input_files("#file-input", video_path)
    return {"source": "upload", "value": video_path}


def wait_for_preview(page, timeout_ms: int = 30000):
    page.wait_for_function(
        "() => document.getElementById('canvas').style.display === 'block'",
        timeout=timeout_ms,
    )
