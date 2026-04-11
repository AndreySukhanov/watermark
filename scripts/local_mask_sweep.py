import json
import os
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from scripts.local_quality_probe import (  # noqa: E402
    VIDEO_FILE,
    create_clip,
    load_regions,
    build_quality_analysis,
)


OUTPUT_ROOT = Path(
    os.environ.get("OUTPUT_DIR", str(BASE / "test001" / "local_mask_sweep"))
)
ENGINE = os.environ.get("ENGINE", "propainter_quality")
AUTODETECT = os.environ.get("AUTODETECT", "0").lower() not in {"0", "false", "no"}


DETAIL_OVERRIDES = {
    "propainter_crop_padding": 48,
    "propainter_crop_merge_gap": 16,
    "propainter_crop_max_width": 720,
    "propainter_crop_max_height": 320,
}

SWEEP_PRESETS = [
    {
        "name": "auto_detail",
        "options": {
            "mask_shape": "auto",
            **DETAIL_OVERRIDES,
        },
    },
    {
        "name": "temporal_auto_detail",
        "options": {
            "mask_shape": "auto",
            "temporal_mask_samples": 6,
            "temporal_mask_min_hits": 2,
            **DETAIL_OVERRIDES,
        },
    },
    {
        "name": "hf018_detail",
        "options": {
            "mask_shape": "hf_segmenter",
            "segmenter_weights": "segmenter_universal.pth",
            "segmenter_threshold": 0.18,
            **DETAIL_OVERRIDES,
        },
    },
    {
        "name": "hf025_detail",
        "options": {
            "mask_shape": "hf_segmenter",
            "segmenter_weights": "segmenter_universal.pth",
            "segmenter_threshold": 0.25,
            **DETAIL_OVERRIDES,
        },
    },
    {
        "name": "temporal_hf025_detail",
        "options": {
            "mask_shape": "temporal_hf_segmenter",
            "segmenter_weights": "segmenter_universal.pth",
            "segmenter_threshold": 0.25,
            "temporal_mask_samples": 6,
            "temporal_mask_min_hits": 2,
            **DETAIL_OVERRIDES,
        },
    },
    {
        "name": "hybrid018_detail",
        "options": {
            "mask_shape": "hybrid_segmenter",
            "segmenter_weights": "segmenter_universal.pth",
            "segmenter_threshold": 0.18,
            **DETAIL_OVERRIDES,
        },
    },
    {
        "name": "hybrid025_detail",
        "options": {
            "mask_shape": "hybrid_segmenter",
            "segmenter_weights": "segmenter_universal.pth",
            "segmenter_threshold": 0.25,
            **DETAIL_OVERRIDES,
        },
    },
]


def log(line: str):
    print(line, flush=True)


def main():
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(f"Видео не найдено: {VIDEO_FILE}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_root = OUTPUT_ROOT / stamp
    sweep_root.mkdir(parents=True, exist_ok=True)

    clip_path = create_clip(VIDEO_FILE, OUTPUT_ROOT)
    regions = load_regions()
    summary = []

    for preset in SWEEP_PRESETS:
        name = preset["name"]
        run_dir = sweep_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        log(f"[probe] {name}")
        started = time.perf_counter()
        analysis = build_quality_analysis(
            input_path=clip_path,
            regions=regions,
            engine_key=ENGINE,
            engine_options=preset["options"],
            autodetect=AUTODETECT,
            run_dir=run_dir,
        )
        elapsed = round(time.perf_counter() - started, 2)
        (run_dir / "analysis.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary.append(
            {
                "name": name,
                "elapsed_s": elapsed,
                "mask_runtime_shape": analysis.get("mask_runtime_shape"),
                "mask_runtime_warning": analysis.get("mask_runtime_warning"),
                "mask_coverage": analysis.get("mask_coverage"),
                "crop_groups": len(analysis.get("crop_groups") or []),
                "mask_preview_path": analysis.get("mask_preview_path"),
                "crop_preview_path": analysis.get("crop_preview_path"),
            }
        )

    (sweep_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "| Preset | Time | Mask Shape | Coverage | Crop Groups | Warning |",
        "|---|---:|---|---:|---:|---|",
    ]
    for item in summary:
        warning = "yes" if item["mask_runtime_warning"] else ""
        lines.append(
            f"| `{item['name']}` | `{item['elapsed_s']:.2f}s` | "
            f"`{item['mask_runtime_shape']}` | `{item['mask_coverage']}` | "
            f"`{item['crop_groups']}` | `{warning}` |"
        )
    (sweep_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    latest = OUTPUT_ROOT / "latest"
    if latest.exists():
        shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(sweep_root, latest)
    log(json.dumps({"sweep_root": str(sweep_root), "count": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
