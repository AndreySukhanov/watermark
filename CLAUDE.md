# CLAUDE.md (Web Edition)

This file provides guidance to Claude Code (claude.ai/code) when working with the **Watermark Remover Web Service**.

## Project Focus
A modern, web-based platform for video watermark removal using FFmpeg and AI inpainting (LAMA).

## Commands
```bash
# Install core web/API dependencies
pip install -r requirements_web.txt

# Install AI components (for GPU/AI mode)
pip install iopaint torch torchvision

# Run the backend (FastAPI)
python server.py

# Run automated tests
python test_batch_queue.py
```

## Architecture (Web-Only)
- **FastAPI (`server.py`)**: Unified backend for uploads, WebSockets, and job management.
- **Frontend (`static/`)**: Vanilla JS/HTML/CSS. Communicates via REST and WS.
- **AI Runner (`services/iopaint_runner.py`)**: Orhcestrates LAMA inpainting.
- **Batch Processing**: Uses `ThreadPoolExecutor` and internal `QueueJob` list.

## Implementation Rules
- **Non-blocking UI**: Long tasks (AI/FFmpeg) must run in the background (threads/processes).
- **Communication**: Use WebSockets for real-time logs and progress updates.
- **Storage**: Temporary files go to `temp_web/`. Clean up old files regularly.
- **AI Scalability**: Process long videos in small frame batches to prevent disk overflow.

## Error Handling
- Capture FFmpeg/IOPaint stderr and stream it back to the UI log.
- Handle WebSockets disconnects (auto-terminate/cleanup related jobs).
- Provide detailed API error responses in JSON format.
