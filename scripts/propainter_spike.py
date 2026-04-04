import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.iopaint_runner import generate_mask
from services.video_info import get_video_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a standalone ProPainter spike on a video with a static mask."
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--regions-file", required=True, help="Path to JSON with mask regions")
    parser.add_argument("--propainter-dir", required=True, help="Path to official ProPainter repo")
    parser.add_argument("--work-dir", default="temp_web/propainter_spike", help="Working directory")
    parser.add_argument("--output", default="", help="Final output mp4 path")
    parser.add_argument("--python-bin", default=sys.executable, help="Python binary for ProPainter repo")
    parser.add_argument("--width", type=int, default=1280, help="Processing width")
    parser.add_argument("--height", type=int, default=720, help="Processing height")
    parser.add_argument("--subvideo-length", type=int, default=50, help="ProPainter subvideo_length")
    parser.add_argument("--neighbor-length", type=int, default=10, help="ProPainter neighbor_length")
    parser.add_argument("--ref-stride", type=int, default=10, help="ProPainter ref_stride")
    parser.add_argument("--mask-dilation", type=int, default=4, help="ProPainter mask_dilation")
    parser.add_argument("--mask-padding", type=int, default=8, help="Static mask padding")
    parser.add_argument("--mask-expand", type=int, default=4, help="Static mask dilation")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 inference")
    return parser.parse_args()


def run(cmd, cwd=None):
    started = time.perf_counter()
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)
    return time.perf_counter() - started


def extract_frames(video_path: Path, frames_dir: Path):
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        str(frames_dir / "%06d.png"),
    ])


def main():
    args = parse_args()
    video_path = Path(args.video).resolve()
    propainter_dir = Path(args.propainter_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not propainter_dir.exists():
        raise FileNotFoundError(f"ProPainter repo not found: {propainter_dir}")

    info = get_video_info(str(video_path))
    if info is None:
        raise RuntimeError(f"Unable to read video info: {video_path}")

    regions = json.loads(Path(args.regions_file).read_text(encoding="utf-8"))
    mask_path = work_dir / "mask.png"
    generate_mask(
        info.width,
        info.height,
        regions,
        mask_path,
        padding=args.mask_padding,
        dilate=args.mask_expand,
    )

    output_root = work_dir / "propainter_output"
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "source_frames"
    extract_frames(video_path, frames_dir)

    save_fps = max(1, int(round(info.fps or 25.0)))

    infer_cmd = [
        args.python_bin,
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
        str(args.width),
        "--height",
        str(args.height),
        "--subvideo_length",
        str(args.subvideo_length),
        "--neighbor_length",
        str(args.neighbor_length),
        "--ref_stride",
        str(args.ref_stride),
        "--mask_dilation",
        str(args.mask_dilation),
    ]
    if args.fp16:
        infer_cmd.append("--fp16")

    infer_elapsed = run(infer_cmd, cwd=propainter_dir)

    result_dir = output_root / frames_dir.name
    propainter_video = result_dir / "inpaint_out.mp4"
    if not propainter_video.exists():
        raise FileNotFoundError(f"ProPainter result not found: {propainter_video}")

    final_output = Path(args.output) if args.output else work_dir / f"{video_path.stem}_propainter.mp4"
    final_output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(propainter_video),
        "-i",
        str(video_path),
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
        str(final_output),
    ]
    mux_elapsed = run(ffmpeg_cmd)

    total_elapsed = infer_elapsed + mux_elapsed
    print()
    print(f"Source: {video_path}")
    print(f"Mask: {mask_path}")
    print(f"Intermediate: {propainter_video}")
    print(f"Final: {final_output}")
    print(f"Inference: {infer_elapsed:.1f}s")
    print(f"Mux/upscale: {mux_elapsed:.1f}s")
    print(f"Total: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
