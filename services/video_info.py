import json
import re
import subprocess
from dataclasses import dataclass


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float


def get_video_info(video_path: str, ffprobe_path: str = "ffprobe") -> VideoInfo | None:
    info = _try_ffprobe(video_path, ffprobe_path)
    if info is None:
        info = _try_ffmpeg_fallback(video_path)
    return info


def _try_ffprobe(video_path: str, ffprobe_path: str) -> VideoInfo | None:
    try:
        cmd = [
            ffprobe_path, "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        return _parse_ffprobe(json.loads(result.stdout))
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    except Exception:
        return None


def _parse_ffprobe(data: dict) -> VideoInfo | None:
    try:
        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if not video_stream:
            return None

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        fps = 0.0
        fps_str = video_stream.get("avg_frame_rate", "0/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            if int(den) != 0:
                fps = round(int(num) / int(den), 3)

        duration = float(data.get("format", {}).get("duration", 0))

        return VideoInfo(width=width, height=height, duration=duration, fps=fps)
    except Exception:
        return None


def _try_ffmpeg_fallback(video_path: str) -> VideoInfo | None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path],
            capture_output=True, text=True, timeout=10,
        )
        stderr = result.stderr

        width, height = 0, 0
        fps = 0.0
        duration = 0.0

        res_match = re.search(r"(\d{2,5})x(\d{2,5})", stderr)
        if res_match:
            width, height = int(res_match.group(1)), int(res_match.group(2))

        fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
        if fps_match:
            fps = float(fps_match.group(1))

        dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
        if dur_match:
            h, m, s = int(dur_match.group(1)), int(dur_match.group(2)), float(dur_match.group(3))
            duration = h * 3600 + m * 60 + s

        if width and height:
            return VideoInfo(width=width, height=height, duration=duration, fps=fps)
    except Exception:
        pass
    return None
