import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PRESETS = [
    {
        "name": "hybrid_segmenter_960",
        "options": {
            "mask_shape": "hybrid_segmenter",
            "refine_mask": False,
            "propainter_width": 960,
            "propainter_height": 540,
            "propainter_subvideo_length": 30,
            "propainter_neighbor_length": 10,
            "propainter_ref_stride": 10,
            "propainter_mask_dilation": 6,
            "mask_padding": 12,
            "mask_dilate": 4,
            "segmenter_threshold": 0.42,
            "feather_radius": 3,
            "temporal_mask_samples": 0,
            "propainter_fp16": True,
        },
    },
    {
        "name": "hf_segmenter_960",
        "options": {
            "mask_shape": "hf_segmenter",
            "refine_mask": False,
            "propainter_width": 960,
            "propainter_height": 540,
            "propainter_subvideo_length": 30,
            "propainter_neighbor_length": 10,
            "propainter_ref_stride": 10,
            "propainter_mask_dilation": 6,
            "mask_padding": 12,
            "mask_dilate": 4,
            "segmenter_threshold": 0.45,
            "feather_radius": 3,
            "temporal_mask_samples": 0,
            "propainter_fp16": True,
        },
    },
    {
        "name": "glyph_mask_tight_960",
        "options": {
            "refine_mask": True,
            "propainter_width": 960,
            "propainter_height": 540,
            "propainter_subvideo_length": 30,
            "propainter_neighbor_length": 10,
            "propainter_ref_stride": 10,
            "propainter_mask_dilation": 4,
            "mask_padding": 4,
            "mask_dilate": 3,
            "feather_radius": 2,
            "temporal_mask_samples": 0,
            "propainter_fp16": True,
        },
    },
    {
        "name": "glyph_mask_medium_960",
        "options": {
            "refine_mask": True,
            "propainter_width": 960,
            "propainter_height": 540,
            "propainter_subvideo_length": 30,
            "propainter_neighbor_length": 10,
            "propainter_ref_stride": 10,
            "propainter_mask_dilation": 6,
            "mask_padding": 8,
            "mask_dilate": 4,
            "feather_radius": 3,
            "temporal_mask_samples": 0,
            "propainter_fp16": True,
        },
    },
]


def load_presets() -> list[dict]:
    raw = os.environ.get("PROPAINTER_SWEEP_PRESETS_JSON", "").strip()
    if not raw:
        return DEFAULT_PRESETS
    data = json.loads(raw)
    if not isinstance(data, list):
        raise RuntimeError("PROPAINTER_SWEEP_PRESETS_JSON должен быть JSON-массивом")
    return data


def main():
    root = Path(
        os.environ.get(
            "PROPAINTER_SWEEP_OUTPUT_DIR",
            str(Path(__file__).resolve().parents[1] / "output" / "propainter_sweep"),
        )
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = root / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).with_name("benchmark_quality.py")
    start_index = max(0, int(os.environ.get("PROPAINTER_SWEEP_START_INDEX", "0")))
    presets = load_presets()[start_index:]
    summary = {"started_at": stamp, "presets": []}

    for preset in presets:
        name = str(preset["name"])
        options = preset.get("options") or {}
        if not isinstance(options, dict):
            raise RuntimeError(f"Некорректные options для preset {name}")

        preset_dir = run_root / name
        env = os.environ.copy()
        env["BENCH_ENGINES"] = "propainter_quality"
        env["ENGINE_OPTIONS_JSON"] = json.dumps({"propainter_quality": options}, ensure_ascii=False)
        env["BENCH_OUTPUT_DIR"] = str(preset_dir)
        if env.get("SERVER_VIDEO") and env.get("CLIP_SERVER_VIDEO"):
            base_clip = Path(env["CLIP_SERVER_VIDEO"])
            env["CLIP_SERVER_VIDEO"] = str(base_clip.with_name(f"{base_clip.stem}_{stamp}_{name}{base_clip.suffix}"))

        print(f"\n=== {name} ===", flush=True)
        print(json.dumps(options, ensure_ascii=False), flush=True)
        started = time.time()
        result = subprocess.run([sys.executable, str(script_path)], env=env)
        elapsed = round(time.time() - started, 1)

        summary["presets"].append(
            {
                "name": name,
                "options": options,
                "elapsed_seconds": elapsed,
                "returncode": result.returncode,
                "output_dir": str(preset_dir),
            }
        )
        (run_root / "sweep_manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print(f"\nГотово: {run_root}", flush=True)


if __name__ == "__main__":
    main()
