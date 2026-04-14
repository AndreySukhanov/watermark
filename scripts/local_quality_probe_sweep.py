import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
PYTHON_EXE = Path(os.environ.get("PYTHON_EXE", sys.executable))
ENGINE = os.environ.get("ENGINE", "propainter_quality")
OUTPUT_ROOT = Path(
    os.environ.get(
        "OUTPUT_ROOT",
        str(BASE / "test001" / "local_quality_probe_sweep"),
    )
)
ENGINE_OPTIONS_FILE = os.environ.get("ENGINE_OPTIONS_FILE")
ENGINE_OPTIONS_JSON = os.environ.get("ENGINE_OPTIONS_JSON", "{}")
CLIP_DURATION = os.environ.get("CLIP_DURATION", "5")
OFFSETS = [
    float(item.strip())
    for item in os.environ.get("OFFSETS", "0,60,120,180").split(",")
    if item.strip()
]


def load_engine_options_json() -> str:
    if ENGINE_OPTIONS_FILE:
        return Path(ENGINE_OPTIONS_FILE).read_text(encoding="utf-8")
    return ENGINE_OPTIONS_JSON


def run_probe(offset: float, output_dir: Path) -> dict:
    env = os.environ.copy()
    env["ENGINE"] = ENGINE
    env["CLIP_DURATION"] = CLIP_DURATION
    env["CLIP_OFFSET"] = str(offset)
    env["OUTPUT_DIR"] = str(output_dir)
    env["ENGINE_OPTIONS_JSON"] = load_engine_options_json()

    cmd = [str(PYTHON_EXE), str(BASE / "scripts" / "local_quality_probe.py")]
    proc = subprocess.run(cmd, cwd=BASE, env=env, text=True, capture_output=True, check=True)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Probe offset {offset} returned no output")
    result = json.loads(lines[-1])
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    return result


def build_markdown(summary: dict) -> str:
    lines = [
        "# Local Quality Probe Sweep",
        "",
        f"- Engine: `{summary['engine']}`",
        f"- Clip duration: `{summary['clip_duration']}s`",
        f"- Offsets: `{', '.join(summary['offset_labels'])}`",
        "",
        "| Offset | Mask % | Crop groups | Crop area % | Risky | Empty |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["runs"]:
        lines.append(
            f"| `{item['offset']}s` | `{item['mask_coverage']:.3f}` | `{item['crop_groups']}` | "
            f"`{item['crop_area_pct']:.3f}` | `{item['risky_groups']}` | `{item['empty_groups']}` |"
        )
    lines.extend(
        [
            "",
            f"- Max crop area: `{summary['max_crop_area_pct']:.3f}%`",
            f"- Max risky groups: `{summary['max_risky_groups']}`",
            f"- Max empty groups: `{summary['max_empty_groups']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    run_root = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for offset in OFFSETS:
        label = str(int(offset)) if float(offset).is_integer() else str(offset).replace(".", "_")
        output_dir = run_root / f"offset_{label}"
        started = time.perf_counter()
        result = run_probe(offset, output_dir)
        result["offset"] = offset
        result["elapsed_sec"] = round(time.perf_counter() - started, 2)
        runs.append(result)
        print(
            json.dumps(
                {
                    "offset": offset,
                    "mask_coverage": result["mask_coverage"],
                    "crop_area_pct": result["crop_area_pct"],
                    "risky_groups": result["risky_groups"],
                    "empty_groups": result["empty_groups"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "engine": ENGINE,
        "clip_duration": float(CLIP_DURATION),
        "offset_labels": [str(int(item)) if float(item).is_integer() else str(item) for item in OFFSETS],
        "runs": [
            {
                "offset": item["offset"],
                "run_dir": item["run_dir"],
                "mask_coverage": item["mask_coverage"],
                "crop_groups": item["crop_groups"],
                "crop_area_pct": item["crop_area_pct"],
                "risky_groups": item["risky_groups"],
                "empty_groups": item["empty_groups"],
                "elapsed_sec": item["elapsed_sec"],
            }
            for item in runs
        ],
        "max_crop_area_pct": max(item["crop_area_pct"] for item in runs) if runs else 0.0,
        "max_risky_groups": max(item["risky_groups"] for item in runs) if runs else 0,
        "max_empty_groups": max(item["empty_groups"] for item in runs) if runs else 0,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "summary.md").write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"run_root": str(run_root), **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
