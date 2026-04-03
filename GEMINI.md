# GEMINI.md

## Project Overview
**Watermark Remover (Web)** is a high-performance web service designed for automated and AI-powered watermark removal from videos. The project is focused on a web-first approach, leveraging FastAPI for the backend and a modern web UI for interaction.

### Core Architecture
*   **Backend:** FastAPI (Python) handles file uploads, video metadata extraction, and process orchestration.
*   **Processing Engine:** 
    *   **FFmpeg Delogo:** Fast, standard filtering for static watermarks.
    *   **AI Inpainting (IOPaint/LAMA):** High-quality, frame-by-frame reconstruction for complex or dynamic logos.
*   **Concurrency:** 
    *   `asyncio` for non-blocking I/O (WebSockets, API).
    *   `ThreadPoolExecutor` for CPU-intensive processing tasks.
    *   **Batch Queue:** A persistent (in-memory) queue system allowing multiple videos to be processed sequentially.
*   **Real-time Communication:** WebSockets stream progress, logs, and job status directly to the frontend.

## Key Technologies
*   **Backend:** FastAPI, Uvicorn, WebSockets, `aiofiles`.
*   **Frontend:** Vanilla HTML5/CSS3/JavaScript (ES6+).
*   **Video Processing:** FFmpeg, `ffprobe`.
*   **AI/ML:** `iopaint` (LAMA model), `torch` (CUDA for GPU acceleration).
*   **Deployment:** Docker, RunPod (optimized for GPU instances).

## Directory Structure
*   `server.py`: Main entry point for the FastAPI server.
*   `static/`: All frontend assets (HTML, CSS, JS).
*   `services/`:
    *   `iopaint_runner.py`: Core AI logic (batching, parallelizing, gap-filling).
    *   `video_info.py`: Metadata extraction using `ffprobe`.
*   `temp_web/`: Working directory for uploads, frames, and results.
*   `test_batch_queue.py`: Automated testing script for the web pipeline.

## Running the Application
```powershell
# Install dependencies
pip install -r requirements_web.txt

# Install AI components (optional but recommended for AI mode)
pip install iopaint torch torchvision

# Start the server
python server.py
```
*Access the UI at `http://localhost:8000`.*

## Development Focus
1.  **Web-Only:** No Desktop/Qt components. Focus on browser-based UX.
2.  **Scalability:** Optimize frame batching to handle very long videos (1h+) without running out of disk or VRAM.
3.  **Parallelism:** Maximize GPU/CPU utilization during AI processing.
4.  **Deployment:** Ensure smooth operation inside Docker containers and on GPU-enabled VPS (RunPod/Lambda).

## Roadmap
- [x] Multi-file batch uploading and queueing.
- [x] Standard `delogo` processing with real-time feedback.
- [x] Parallel AI inpainting pipeline (LAMA).
- [ ] User authentication and private workspaces.
- [ ] Persistent database for job history (SQLAlchemy/PostgreSQL).
- [ ] Dynamic mask detection (Auto-detection of logos).
- [ ] Multi-node processing (Distributing jobs across multiple servers).
