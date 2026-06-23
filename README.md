# GhostMark · video watermark remover

[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-RTX%204090-76B900?logo=nvidia&logoColor=white)](#benchmarks)
[![LaMa](https://img.shields.io/badge/inpaint-LaMa-blue)](https://github.com/advimman/lama)
[![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-FF6F00)](https://github.com/JaidedAI/EasyOCR)
[![License: Source-Available](https://img.shields.io/badge/license-Source--Available-orange)](LICENSE)

**Auto-detect and erase tiled watermarks from video — with surgical precision.**
On a 3-minute 1080p clip with 11 diagonally-tiled watermarks, removes every readable trace while leaving faces, hands, and background untouched. **5.3× realtime on RTX 4090.**



---

## Why this exists

Most "watermark removers" do one of two things:
1. Ask you to draw a rectangle and then blur/clone-stamp inside it (visible smudge, no detail recovery).
2. Run a generic image-inpainter on every frame (slow, flickers, eats faces that overlap the mask).

**GhostMark does neither.** It treats a tiled watermark as a structured signal: a known string of glyphs, repeated on a static grid across the frame, sitting on top of moving content. The pipeline exploits all three properties.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Pipeline                                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   video ──► sample 20 frames ──► median ──► high-pass                       │
│                                       │                                     │
│                                       ▼                                     │
│                              EasyOCR finds tiles                            │
│                              that contain known                             │
│                              token substrings                               │
│                                       │                                     │
│                                       ▼                                     │
│                          pick cleanest tile as template                     │
│                                       │                                     │
│                                       ▼                                     │
│                       cv2.matchTemplate → recover                           │
│                       interior tiles obscured by content                    │
│                                       │                                     │
│                                       ▼                                     │
│                  per-tile high-pass threshold → glyph-tight mask            │
│                  (only the actual letter strokes, NOT the bbox)             │
│                                       │                                     │
│                                       ▼                                     │
│                          optional: subtract face zones                      │
│                          (mediapipe / opencv haar)                          │
│                                       │                                     │
│                                       ▼                                     │
│                   ffmpeg extract → LaMa inpaint (8 workers) → ffmpeg mux    │
│                                       │                                     │
│                                       ▼                                     │
│                                 output.mp4                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The mask is **per-glyph**, not per-rectangle. LaMa touches the actual letter strokes; everything else (face, background, subtitles) is bit-identical to the source.

OCR + template-match runs once per video. The mask is hashed by `sha256(video bytes)[:12]` + param hash and cached, so a 17-minute end-to-end run on the same video reuses the ~2 min OCR cost on subsequent passes.

---

## Features

- **Auto-detect tiled watermarks** — no manual region drawing required for the common case (text-based site watermarks).
- **Manual region mode** — for cases where OCR can't help (logos, glyphless icons, opaque blocks), draw rectangles on a canvas preview.
- **Two engines, one server:**
  - `lama_fast` — single-frame LaMa inpainting, GPU-batched. Production default.
  - `propainter_quality` — temporal flow-aware inpainting (slower, kept for cases where neighbour frames help).
- **GPU-parallel processing** — N parallel IOPaint workers (default 8), batched 500 frames at a time to fit VRAM.
- **Frame-skip interpolation** — process every 3rd–4th frame, copy nearest-neighbour into skipped slots. Static watermark → invisible quality cost, ~3–4× speedup.
- **WebSocket progress stream** — real-time frame counter, log tail, cancellable mid-run.
- **Batch queue** — drop a folder of videos, walk away.
- **Face-safe mode** — runs face detection over the same sample frames, subtracts face zones from the mask before LaMa sees it. Belt-and-braces against the rare case where a watermark stroke overlaps a face.
- **RunPod-ready** — Docker + `deploy_runpod.sh`, idle watchdog that auto-stops the pod after N minutes of inactivity (cost control).

---

## Benchmarks

Validated end-to-end on a 3:10 / 1080p / 25 fps clip with 11 diagonally-tiled text watermarks.

| Hardware | Engine | Workers | Runtime | × realtime | Visual result |
|---|---|---|---|---|---|
| RTX 4090 | `lama_fast` + OCR mask | 8 | ~17 min | **5.3×** | 0 readable watermarks across all sampled timestamps; faces/subtitles/background intact; no mask flicker |
| RTX A5000 | `lama_fast` (manual mask) | 4 | 6× target | 6× | Watermarks visibly suppressed but not eliminated |
| RTX A5000 | `propainter_quality` | 4 | ~134× | far too slow | Slightly better face preservation but watermark still readable |

The takeaway: a glyph-tight mask is the unlock. Once LaMa sees the right pixels, even the "fast" engine wins.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/`                    | Web UI (canvas preview, region drawing, mode toggle, progress) |
| `POST` | `/api/upload`          | Upload a video (multipart) |
| `GET`  | `/api/info?path=`      | Resolution, duration, fps |
| `GET`  | `/api/frame?path=&time=` | Single frame as JPEG (for canvas preview) |
| `WS`   | `/ws/process`          | Start a job, stream progress |
| `GET`  | `/api/download/{file}` | Get the finished MP4 |
| `GET`  | `/api/queue`           | List queued/running jobs |
| `POST` | `/api/queue`           | Enqueue a video |
| `POST` | `/api/cancel/{job_id}` | Cancel a running job (kills IOPaint workers) |
| `GET`  | `/health`              | Status + GPU info (cuda available, device name, VRAM) |

Full OpenAPI at `/docs` once the server is up.

---

## Quick start

### Local (CPU, slow but works)

```bash
pip install -r requirements_web.txt
apt-get install -y ffmpeg  # or brew install ffmpeg
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, drop a video, draw a region or hit "Auto-detect" if your watermark has readable text.

### RunPod (GPU, recommended)

1. Create a pod with `pytorch:2.x-cuda12.x` template and an RTX 4090 / A5000.
2. SSH in (key-based, no passwords).
3. Run the bootstrap:
   ```bash
   curl -sSL https://raw.githubusercontent.com/<your-fork>/main/deploy_runpod.sh | bash
   ```
4. Open the proxy URL: `https://<POD_ID>-8000.proxy.runpod.net/`.

The pod will auto-stop after `IDLE_TIMEOUT_MINUTES` of no activity (default 30) if `RUNPOD_API_KEY` and `RUNPOD_POD_ID` are set.

### Docker

```bash
docker build -t ghostmark .
docker run --gpus all -p 8000:8000 -v $(pwd)/output:/app/output ghostmark
```

---

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `BATCH_SIZE`              | `500`   | Frames per IOPaint batch (VRAM-bounded) |
| `FRAME_SKIP`              | `4`     | Process every Nth frame, copy into skipped slots |
| `IDLE_TIMEOUT_MINUTES`    | `30`    | Auto-stop pod after this much idle time |
| `RUNPOD_POD_ID`           | —       | Required for auto-stop |
| `RUNPOD_API_KEY`          | —       | Required for auto-stop (or `~/.runpod/config.toml`) |
| `WATERMARK_TOKENS`        | `.com,.net,.io,.tv,www.,http` | Comma-separated substrings OCR uses to flag tiles as "watermark text". Override for your domain pattern. |
| `OCR_MODEL_DIR`           | `./models/easyocr` | Where EasyOCR caches its model weights |

---

## Project layout

```
server.py                     FastAPI + WebSocket, job orchestration, idle watchdog
services/
  watermark_ocr.py            EasyOCR + matchTemplate + glyph-mask builder
  watermark_segmenter.py      HF segmenter alternative (kept for non-text watermarks)
  watermark_detector.py       Repeated-region heuristics (pre-OCR fallback)
  iopaint_runner.py           LaMa pipeline: extract, thin, parallel inpaint, reassemble
  propainter_runner.py        Temporal engine (alternative path)
  ai_engines.py               Engine registry + preset resolution
  face_safe.py                Face detector wrapper (mediapipe / haar)
  face_restore.py             Optional GFPGAN-style face restore pass
  video_info.py               ffprobe wrapper
static/                       Web UI (vanilla JS + canvas)
assets/quality_presets/       Engine configs (currently: lama_ocr)
scripts/setup_propainter_runtime.sh   ProPainter weights / runtime bootstrap
Dockerfile
deploy_runpod.sh
requirements_web.txt
```

---

## What it doesn't do (yet)

- **Animated watermarks** (rolling tickers, sliding domain banners). Current pipeline assumes the watermark is static across frames. A `temporal_ocr_mask` mode that re-detects per N seconds is on the roadmap.
- **Audio watermarks.** Out of scope.
- **Mobile** / no web build target. Server-side only.

---

## License

[Source-Available](LICENSE). Read, study, learn — but no use in products or services without prior written permission. Contact `avsukhanov21@gmail.com` for licensing.
