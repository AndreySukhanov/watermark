import os
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class AIEngineConfig:
    key: str
    label: str
    family: str
    description: str
    estimate_multiplier: float
    skip: int
    refine_mask: bool
    mask_padding: int
    mask_dilate: int
    feather_radius: int
    blend_skipped: bool
    worker_count_override: int = 0
    hd_strategy: str = "Resize"
    resize_limit: int = 1024
    output_suffix: str = ".jpg"
    output_quality: int = 99
    mask_shape: str = "auto"
    segmenter_threshold: float = 0.45
    segmenter_weights: str = "segmenter.pth"
    propainter_width: int = 960
    propainter_height: int = 540
    propainter_subvideo_length: int = 30
    propainter_neighbor_length: int = 10
    propainter_ref_stride: int = 10
    propainter_mask_dilation: int = 4
    propainter_fp16: bool = True
    propainter_use_crops: bool = False
    propainter_tighten_regions: bool = True
    propainter_crop_mask_boost: bool = False
    propainter_crop_padding: int = 64
    propainter_crop_merge_gap: int = 64
    propainter_crop_max_width: int = 1120
    propainter_crop_max_height: int = 640
    temporal_mask_samples: int = 0
    temporal_mask_min_hits: int = 2

    def to_metadata(self) -> dict:
        data = asdict(self)
        return {
            "key": data["key"],
            "label": data["label"],
            "family": data["family"],
            "description": data["description"],
            "estimate_multiplier": data["estimate_multiplier"],
            "skip": data["skip"],
            "refine_mask": data["refine_mask"],
            "mask_shape": data["mask_shape"],
            "segmenter_threshold": data["segmenter_threshold"],
            "segmenter_weights": data["segmenter_weights"],
            "propainter_use_crops": data["propainter_use_crops"],
            "propainter_tighten_regions": data["propainter_tighten_regions"],
            "propainter_crop_mask_boost": data["propainter_crop_mask_boost"],
            "temporal_mask_samples": data["temporal_mask_samples"],
        }


AI_ENGINES: dict[str, AIEngineConfig] = {
    "lama_fast": AIEngineConfig(
        key="lama_fast",
        label="LaMa Fast",
        family="lama",
        description="Максимально быстрый режим для длинных роликов.",
        estimate_multiplier=float(os.environ.get("ENGINE_LAMA_FAST_X", "4.9")),
        skip=int(os.environ.get("ENGINE_LAMA_FAST_SKIP", os.environ.get("FRAME_SKIP", "4"))),
        refine_mask=False,
        mask_padding=int(os.environ.get("ENGINE_LAMA_FAST_MASK_PADDING", "8")),
        mask_dilate=int(os.environ.get("ENGINE_LAMA_FAST_MASK_DILATE", "4")),
        feather_radius=int(os.environ.get("ENGINE_LAMA_FAST_FEATHER", "2")),
        blend_skipped=True,
        worker_count_override=int(os.environ.get("ENGINE_LAMA_FAST_WORKERS", "8")),
        resize_limit=int(os.environ.get("ENGINE_LAMA_FAST_RESIZE", "1024")),
        output_quality=int(os.environ.get("ENGINE_LAMA_FAST_JPEG_QUALITY", "99")),
        mask_shape=os.environ.get("ENGINE_LAMA_FAST_MASK_SHAPE", "auto"),
        segmenter_threshold=float(os.environ.get("ENGINE_LAMA_FAST_SEGMENTER_THRESHOLD", "0.45")),
        segmenter_weights=os.environ.get("ENGINE_LAMA_FAST_SEGMENTER_WEIGHTS", "segmenter.pth"),
    ),
    "propainter_quality": AIEngineConfig(
        key="propainter_quality",
        label="ProPainter",
        family="propainter",
        description="Video-aware quality mode с лучшей temporal consistency.",
        estimate_multiplier=float(os.environ.get("ENGINE_PROPAINTER_X", "19.0")),
        skip=1,
        refine_mask=True,
        mask_padding=int(os.environ.get("ENGINE_PROPAINTER_MASK_PADDING", "4")),
        mask_dilate=int(os.environ.get("ENGINE_PROPAINTER_MASK_DILATE", "3")),
        feather_radius=int(os.environ.get("ENGINE_PROPAINTER_FEATHER", "3")),
        blend_skipped=False,
        worker_count_override=0,
        resize_limit=int(os.environ.get("ENGINE_PROPAINTER_RESIZE", "960")),
        mask_shape=os.environ.get("ENGINE_PROPAINTER_MASK_SHAPE", "auto"),
        segmenter_threshold=float(os.environ.get("ENGINE_PROPAINTER_SEGMENTER_THRESHOLD", "0.45")),
        segmenter_weights=os.environ.get("ENGINE_PROPAINTER_SEGMENTER_WEIGHTS", "segmenter.pth"),
        propainter_width=int(os.environ.get("ENGINE_PROPAINTER_WIDTH", "960")),
        propainter_height=int(os.environ.get("ENGINE_PROPAINTER_HEIGHT", "540")),
        propainter_subvideo_length=int(os.environ.get("ENGINE_PROPAINTER_SUBVIDEO", "30")),
        propainter_neighbor_length=int(os.environ.get("ENGINE_PROPAINTER_NEIGHBOR", "10")),
        propainter_ref_stride=int(os.environ.get("ENGINE_PROPAINTER_REF_STRIDE", "10")),
        propainter_mask_dilation=int(os.environ.get("ENGINE_PROPAINTER_MASK_DILATION", "4")),
        propainter_fp16=os.environ.get("ENGINE_PROPAINTER_FP16", "1").lower() not in {"0", "false", "no"},
        propainter_use_crops=os.environ.get("ENGINE_PROPAINTER_USE_CROPS", "1").lower() not in {"0", "false", "no"},
        propainter_tighten_regions=os.environ.get("ENGINE_PROPAINTER_TIGHTEN_REGIONS", "1").lower() not in {"0", "false", "no"},
        propainter_crop_mask_boost=os.environ.get("ENGINE_PROPAINTER_CROP_MASK_BOOST", "0").lower() not in {"0", "false", "no"},
        propainter_crop_padding=int(os.environ.get("ENGINE_PROPAINTER_CROP_PADDING", "64")),
        propainter_crop_merge_gap=int(os.environ.get("ENGINE_PROPAINTER_CROP_GAP", "64")),
        propainter_crop_max_width=int(os.environ.get("ENGINE_PROPAINTER_CROP_MAX_WIDTH", "1120")),
        propainter_crop_max_height=int(os.environ.get("ENGINE_PROPAINTER_CROP_MAX_HEIGHT", "640")),
        temporal_mask_samples=int(os.environ.get("ENGINE_PROPAINTER_TEMPORAL_MASK_SAMPLES", "0")),
        temporal_mask_min_hits=int(os.environ.get("ENGINE_PROPAINTER_TEMPORAL_MASK_MIN_HITS", "2")),
    ),
}

DEFAULT_AI_ENGINE = "lama_fast"

_INT_OVERRIDE_LIMITS: dict[str, tuple[int, int]] = {
    "skip": (1, 8),
    "mask_padding": (0, 80),
    "mask_dilate": (0, 80),
    "feather_radius": (0, 24),
    "worker_count_override": (0, 16),
    "resize_limit": (256, 2160),
    "output_quality": (70, 100),
    "propainter_width": (320, 1920),
    "propainter_height": (180, 1080),
    "propainter_subvideo_length": (8, 120),
    "propainter_neighbor_length": (1, 60),
    "propainter_ref_stride": (1, 80),
    "propainter_mask_dilation": (0, 120),
    "propainter_crop_padding": (0, 320),
    "propainter_crop_merge_gap": (0, 320),
    "propainter_crop_max_width": (128, 1920),
    "propainter_crop_max_height": (64, 1080),
    "temporal_mask_samples": (0, 16),
    "temporal_mask_min_hits": (1, 16),
}
_FLOAT_OVERRIDE_LIMITS: dict[str, tuple[float, float]] = {
    "segmenter_threshold": (0.05, 0.95),
}
_BOOL_OVERRIDE_KEYS = {
    "refine_mask",
    "blend_skipped",
    "propainter_fp16",
    "propainter_use_crops",
    "propainter_tighten_regions",
    "propainter_crop_mask_boost",
}
_STR_OVERRIDE_KEYS = {
    "hd_strategy": {"Original", "Resize", "Crop"},
    "output_suffix": {".jpg", ".jpeg", ".png"},
    "mask_shape": {"auto", "hf_segmenter", "hybrid_segmenter", "temporal_hf_segmenter"},
    "segmenter_weights": {"segmenter.pth", "segmenter_universal.pth"},
}


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _coerce_int(value, min_value: int, max_value: int) -> int:
    result = int(value)
    return max(min_value, min(max_value, result))


def _coerce_float(value, min_value: float, max_value: float) -> float:
    result = float(value)
    return max(min_value, min(max_value, result))


def resolve_engine_config(engine: str | None, overrides: dict | None = None) -> AIEngineConfig:
    config = get_engine_config(engine)
    if not isinstance(overrides, dict) or not overrides:
        return config

    updates = {}
    for key, value in overrides.items():
        if key in _INT_OVERRIDE_LIMITS:
            try:
                updates[key] = _coerce_int(value, *_INT_OVERRIDE_LIMITS[key])
            except (TypeError, ValueError):
                continue
        elif key in _BOOL_OVERRIDE_KEYS:
            updates[key] = _coerce_bool(value)
        elif key in _FLOAT_OVERRIDE_LIMITS:
            try:
                updates[key] = _coerce_float(value, *_FLOAT_OVERRIDE_LIMITS[key])
            except (TypeError, ValueError):
                continue
        elif key in _STR_OVERRIDE_KEYS and value in _STR_OVERRIDE_KEYS[key]:
            updates[key] = value

    if not updates:
        return config
    return replace(config, **updates)


def get_engine_config(engine: str | None) -> AIEngineConfig:
    if engine in AI_ENGINES:
        return AI_ENGINES[engine]
    return AI_ENGINES[DEFAULT_AI_ENGINE]


def list_engine_metadata() -> list[dict]:
    return [cfg.to_metadata() for cfg in AI_ENGINES.values()]
