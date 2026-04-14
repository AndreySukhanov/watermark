# Status

## Snapshot

Date: `2026-04-15`

Project status:
- Web-only service is deployable and working on RunPod.
- GPU processing, queue, WebSocket progress, and download flow are operational.
- The project is still blocked on final watermark removal quality for the target tiled watermark case.

## Current Baseline

Primary quality preset:
- `assets/quality_presets/propainter_detail_temporal_hf_v20.json`

Why this preset matters:
- It is the current stable planning baseline for `propainter_quality`.
- It removes `risky` and `empty` crop groups on local short-clip planning runs.
- It is now the single preset used for local probe, local sweep, and future GPU benchmarks.

## Latest Measured Results

### Local planning

Preset:
- `propainter_detail_temporal_hf_v20.json`

Sweep over `5s` windows at `0s / 60s / 120s / 180s`:
- `crop_groups = 11`
- `risky_groups = 0`
- `empty_groups = 0`
- `crop_area_pct = 9.796% .. 12.304%`

Local `15s` planning:
- `mask_coverage = 2.76%`
- `crop_groups = 11`
- `crop_area_pct = 14.172%`
- `risky_groups = 0`
- `empty_groups = 0`

Artifacts:
- `test001/local_quality_probe_sweep_v20/20260415_004216/summary.md`
- `test001/local_quality_probe_detail_temporal_hf_v20_15s/20260415_015258/analysis.json`

### RunPod benchmark

Hardware:
- `NVIDIA RTX A5000`

Server state during validation:
- `health = ok`
- `gpu_available = true`

Results on the same `15s` clip:
- `lama_fast`: `90s` (`~6x`)
- `propainter_quality`: `2013s` (`~134x`)

Visual conclusion from the `5s` preview frame:
- `propainter_quality` preserves the face and background better than `lama_fast`
- watermark remains clearly readable in both modes

Artifacts:
- `test001/runpod_15s_compare_20260415/summary.md`
- `test001/runpod_15s_compare_20260415/lama_fast_15s.mp4`
- `test001/runpod_15s_compare_20260415/propainter_quality_15s.mp4`

## What Works

- FastAPI web product
- RunPod deployment
- Queue and cancellation flow
- Local quality planning workflow
- Unified preset-based benchmarking workflow

## Main Problems

### Quality

- The target watermark is not removed well enough yet.
- Even the best current quality path still leaves the watermark readable.
- No acceptable `15s` quality sample exists yet.

### Runtime

- `lama_fast` is near the intended fast mode range.
- `propainter_quality` is far too slow for routine use.

### Product readiness

- The system is operational as an R&D platform.
- It is not ready to claim success on the target watermark-removal requirement.

## Next Step

The next useful step is not another broad parameter sweep.

The next useful step is:
- run the next GPU experiment only from the fixed preset baseline,
- compare actual video quality improvements against the current `15s` compare set,
- focus changes on real watermark suppression quality rather than only planning metrics.
