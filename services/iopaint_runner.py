import math
import shutil
from pathlib import Path
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageDraw, ImageFilter

MASK_PADDING = 8
MASK_DILATE = 4

# Number of parallel IOPaint workers (LAMA uses ~1.5 GB VRAM each)
def get_worker_count(device="cpu"):
    # RTX A5000 24GB: LAMA ~1.5GB each, 4 instances = 6GB — fits easily
    return 4


def generate_mask(width, height, regions, out_path, padding=MASK_PADDING, dilate=MASK_DILATE):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for r in regions:
        x0 = max(0, int(r["x"]) - padding)
        y0 = max(0, int(r["y"]) - padding)
        x1 = min(width, int(r["x"]) + int(r["w"]) + padding)
        y1 = min(height, int(r["y"]) + int(r["h"]) + padding)
        if x1 > x0 and y1 > y0:
            draw.rectangle([x0, y0, x1, y1], fill=255)
    if dilate > 0:
        kernel = max(3, dilate * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        img = img.filter(ImageFilter.MaxFilter(kernel))
    img.save(str(out_path))


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


def extract_frames(input_path, frames_dir, register_process=None):
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    _run_managed_command(
        ["ffmpeg", "-y", "-i", str(input_path), str(frames_dir / "%06d.png")],
        register_process=register_process,
    )
    return sorted(frames_dir.glob("*.png"))


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


def thin_frames(frames_dir, skip=2):
    """Remove frames that aren't on skip boundaries, keeping correct numbering.

    Keeps frames whose 1-based number within the dir satisfies (num - first) % skip == 0.
    Returns count of remaining frames.
    """
    frames_dir = Path(frames_dir)
    all_frames = sorted(frames_dir.glob("*.png"))
    if not all_frames or skip <= 1:
        return len(all_frames)

    nums = sorted(int(f.stem) for f in all_frames)
    first = nums[0]
    keep = set()
    for i, num in enumerate(nums):
        if i % skip == 0:
            keep.add(num)

    for f in all_frames:
        if int(f.stem) not in keep:
            f.unlink()

    return len(keep)


def start_iopaint(frames_dir, mask_path, output_dir, device="cpu"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["iopaint", "run", "--model", "lama", "--device", device,
         "--image", str(frames_dir), "--mask", str(mask_path),
         "--output", str(output_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )


def run_iopaint_sync(frames_dir, mask_path, output_dir, device="cpu", register_process=None):
    """Run IOPaint synchronously, return (returncode, output_text)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = start_iopaint(frames_dir, mask_path, output_dir, device=device)
    if register_process:
        register_process(proc)
    stdout, _ = proc.communicate()
    return proc.returncode, stdout or ""


def run_iopaint_parallel(frames_dir, mask_path, output_dir, device="cpu",
                         workers=None, register_process=None, is_cancelled=None):
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

    # If fewer frames than workers, just run single process
    if len(all_frames) <= workers:
        rc, out = run_iopaint_sync(
            frames_dir,
            mask_path,
            output_dir,
            device,
            register_process=register_process,
        )
        if rc != 0:
            return False, f"IOPaint error (code {rc}): {out[-500:]}"
        return True, None

    # Split frames into sub-dirs
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

    # Run in parallel
    with ThreadPoolExecutor(max_workers=len(sub_dirs)) as pool:
        futures = {
            pool.submit(
                run_iopaint_sync,
                sd,
                mask_path,
                so,
                device,
                _register_subprocess,
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

    # Merge sub-outputs into output_dir
    for so in sub_outs:
        for f in so.glob("*.png"):
            shutil.move(str(f), str(output_dir / f.name))
        shutil.rmtree(so, ignore_errors=True)

    # Cleanup sub-input dirs
    for sd in sub_dirs:
        shutil.rmtree(sd, ignore_errors=True)

    return True, None


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

    # For each missing frame, copy from nearest existing
    existing_set = set(existing_nums)
    # Build a lookup: for any frame number, find nearest existing
    prev_existing = existing_nums[0]
    for num in range(1, total_frames + 1):
        if num in existing_set:
            prev_existing = num
            continue
        # Copy from the last existing frame we saw
        src = all_dir / f"{prev_existing:06d}.png"
        dst = all_dir / f"{num:06d}.png"
        shutil.copy2(str(src), str(dst))


def reassemble_video(inpainted_dir, input_video, output_path, fps, register_process=None):
    _run_managed_command([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(Path(inpainted_dir) / "%06d.png"),
        "-i", str(input_video),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output_path),
    ], register_process=register_process)
