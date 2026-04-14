# TASKS (Web Edition)

## Current Stage

- Web-only product is working on RunPod with FastAPI, queue, WebSocket progress, and GPU processing.
- The main unresolved issue is still final watermark removal quality on the target tiled watermark case.
- The current quality baseline is `assets/quality_presets/propainter_detail_temporal_hf_v20.json`.

## Completed

### Core Web Infrastructure
- [x] FastAPI app (`server.py`)
- [x] Static frontend (`static/index.html`, `static/app.js`, `static/style.css`)
- [x] Upload, metadata, frame preview, download endpoints
- [x] Queue processing, cancel API, WebSocket progress
- [x] Health endpoint with GPU detection

### AI Pipelines
- [x] `lama_fast` pipeline
- [x] `lama_quality` pipeline
- [x] `propainter_quality` pipeline
- [x] Quality analyze endpoint with mask preview and crop preview
- [x] Repeated watermark planning and crop grouping
- [x] Local probe and sweep scripts for short clips

### Deploy / Operations
- [x] RunPod deploy script
- [x] Auto-stop / idle watchdog integration
- [x] Deploy script can target a new RunPod pod through env variables

## Current Validation Snapshot (2026-04-15)

### Local planning baseline
- [x] Preset fixed in `assets/quality_presets/propainter_detail_temporal_hf_v20.json`
- [x] Local sweep on `0s / 60s / 120s / 180s`
- [x] `risky_groups = 0`
- [x] `empty_groups = 0`
- [x] `crop_area_pct` stable in the `9.796% .. 12.304%` range

### RunPod 15s benchmark
- [x] `lama_fast` benchmark completed
- [x] `propainter_quality` benchmark completed
- [ ] Quality is acceptable on the target watermark case
- [ ] Quality mode runtime is acceptable for regular use

## Current Measured Results

- `lama_fast` on a `15s` clip: `90s` (`~6x`)
- `propainter_quality` on the same `15s` clip: `2013s` (`~134x`)
- `propainter_quality` preserves face and background better than `lama_fast`
- Watermark is still clearly readable in both modes

## Open Problems

### P0
- [ ] Remove the tiled watermark well enough on the target video
- [ ] Produce at least one acceptable `15s` quality sample

### P1
- [ ] Make `propainter_quality` materially faster, or prove it is only a premium slow mode
- [ ] Improve actual mask hit quality, not just planning metrics

### P2
- [ ] Expand API and smoke coverage around quality analyze and benchmark flows
- [ ] Improve project status reporting and long-run observability

## Acceptance Checklist

- [x] Upload and inspect video in the browser
- [x] Draw or analyze watermark regions
- [x] Queue jobs and monitor progress
- [x] Preserve output audio
- [x] Complete a full RunPod processing cycle
- [ ] Remove the watermark at acceptable quality on the target case
