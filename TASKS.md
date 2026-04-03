# TASKS (Web Edition)

## Phase 1 — Core Web Infrastructure
- [x] FastAPI skeleton (`server.py`)
- [x] Static files mounting
- [x] Basic HTML/JS frontend
- [x] File upload endpoint (`/api/upload`)
- [x] Video metadata extraction (`/api/info`)
- [x] Frame extraction endpoint (`/api/frame`)

## Phase 2 — Multi-Video Batching
- [x] Multiple file selection in UI
- [x] Batch upload logic in JS
- [x] Batch queue management backend
- [x] "Add all to queue" button with shared regions
- [x] Sequential job execution from queue

## Phase 3 — AI Inpainting Pipeline (LAMA)
- [x] Frame extraction from video range
- [x] AI mask generation from UI regions
- [x] Parallel IOPaint execution (LAMA model)
- [x] Frame thinning (Skip/Fill logic for speed)
- [x] Video reassembly with original audio

## Phase 4 — UI/UX Polish
- [x] Progress bar updates for both modes
- [x] Real-time log streaming via WebSockets
- [x] Dynamic region drawing on canvas
- [x] Error display and job cancellation
- [ ] Dark theme refinement
- [ ] Responsive layout for mobile devices

## Phase 5 — Advanced AI Features (Roadmap)
- [ ] **Auto-Detection**: Use YOLO or Segment-Anything to find logos automatically.
- [ ] **Temporal Consistency**: Improved flow-based frame blending to prevent flickering.
- [ ] **Custom Models**: Support for different inpainting models (ZITS, MAT, etc.).

## Phase 6 — Security & History (Roadmap)
- [ ] **Auth**: Login/Password for multiple users.
- [ ] **Database**: Job history, statistics, and persistence across server restarts.
- [ ] **Storage S3**: Export final results to external storage.

## Phase 7 — Production Readiness
- [x] GPU acceleration (CUDA support)
- [x] Dockerfile for easy deployment
- [x] RunPod auto-stop integration
- [ ] Comprehensive logging system (Loguru/ELK)
- [ ] Unit & Integration tests for all API endpoints

## Acceptance Checklist (Web)
- [x] Select multiple videos at once
- [x] Draw regions on the first video
- [x] Queue all videos for processing
- [x] Real-time progress visible for each job
- [x] Resulting video has watermark removed (standard/AI)
- [x] Audio preserved in the output
- [x] No memory/disk leaks during long batch jobs
