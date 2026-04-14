from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

HF_WATERMARK_MODEL = "christophernavas/watermark-remover"
HF_WATERMARK_DEFAULT_WEIGHTS = "segmenter.pth"


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    return {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in checkpoint.items()
        if hasattr(value, "shape")
    }


@lru_cache(maxsize=4)
def _load_segmenter(weights_name: str = HF_WATERMARK_DEFAULT_WEIGHTS):
    try:
        import torch
        import segmentation_models_pytorch as smp
        from huggingface_hub import hf_hub_download
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "HF watermark segmenter dependencies are missing. Install segmentation-models-pytorch, "
            "huggingface_hub and torchvision."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    weights_path = hf_hub_download(HF_WATERMARK_MODEL, weights_name)
    state = _extract_state_dict(torch.load(weights_path, map_location="cpu"))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform, torch, device


def _build_region_limit(size: tuple[int, int], regions: list[dict] | None, padding: int) -> Image.Image | None:
    if not regions:
        return None
    width, height = size
    limit = Image.new("L", size, 0)
    draw = ImageDraw.Draw(limit)
    for region in regions:
        try:
            x0 = max(0, int(region["x"]) - padding)
            y0 = max(0, int(region["y"]) - padding)
            x1 = min(width, int(region["x"]) + int(region["w"]) + padding)
            y1 = min(height, int(region["y"]) + int(region["h"]) + padding)
        except Exception:
            continue
        if x1 > x0 and y1 > y0:
            draw.rectangle((x0, y0, x1, y1), fill=255)
    return limit


def _iter_region_boxes(width: int, height: int, regions: list[dict] | None, padding: int):
    if not regions:
        yield (0, 0, width, height)
        return
    for region in regions:
        try:
            x0 = max(0, int(region["x"]) - padding)
            y0 = max(0, int(region["y"]) - padding)
            x1 = min(width, int(region["x"]) + int(region["w"]) + padding)
            y1 = min(height, int(region["y"]) + int(region["h"]) + padding)
        except Exception:
            continue
        if x1 > x0 and y1 > y0:
            yield (x0, y0, x1, y1)


def _iter_region_entries(width: int, height: int, regions: list[dict] | None, padding: int):
    if not regions:
        yield None, (0, 0, width, height)
        return
    for region in regions:
        try:
            x0 = max(0, int(region["x"]) - padding)
            y0 = max(0, int(region["y"]) - padding)
            x1 = min(width, int(region["x"]) + int(region["w"]) + padding)
            y1 = min(height, int(region["y"]) + int(region["h"]) + padding)
        except Exception:
            continue
        if x1 > x0 and y1 > y0:
            yield region, (x0, y0, x1, y1)


def _open_reference_image(reference_frame_path, width: int, height: int) -> Image.Image:
    with Image.open(reference_frame_path) as source_image:
        image = source_image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    return image


def _predict_probability_mask(
    image: Image.Image,
    *,
    weights_name: str = HF_WATERMARK_DEFAULT_WEIGHTS,
) -> Image.Image:
    model, transform, torch, device = _load_segmenter(weights_name)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor))[0, 0].detach().float().cpu().numpy()
    prob = Image.fromarray((prediction * 255).clip(0, 255).astype("uint8"), mode="L")
    return prob.resize(image.size, Image.Resampling.BILINEAR)


def _threshold_mask(probability_mask: Image.Image, threshold: float) -> Image.Image:
    cutoff = int(max(0.0, min(1.0, threshold)) * 255)
    return probability_mask.point(lambda p: 255 if p >= cutoff else 0)


def _mask_ratio(mask: Image.Image) -> float:
    hist = mask.histogram()
    total = mask.size[0] * mask.size[1]
    if total <= 0:
        return 0.0
    return hist[255] / float(total)


def _dilate_mask(mask: Image.Image, dilate: int) -> Image.Image:
    if dilate <= 0:
        return mask
    kernel = max(3, int(dilate) * 2 + 1)
    if kernel % 2 == 0:
        kernel += 1
    return mask.filter(ImageFilter.MaxFilter(kernel)).point(lambda p: 255 if p >= 18 else 0)


def _build_text_band_mask(width: int, height: int) -> Image.Image:
    band = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(band)
    left = max(0, int(width * 0.10))
    right = min(width, int(width * 0.90))
    top = max(0, int(height * 0.36))
    bottom = min(height, int(height * 0.64))
    if right > left and bottom > top:
        draw.rectangle((left, top, right, bottom), fill=255)
    band = band.filter(ImageFilter.GaussianBlur(1.2))
    return band.point(lambda p: 255 if p >= 18 else 0)


def _build_text_core_mask(width: int, height: int) -> Image.Image:
    core = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(core)
    left = max(0, int(width * 0.18))
    right = min(width, int(width * 0.82))
    center_y = height / 2.0
    band_height = max(10, int(round(height * 0.18)))
    top = max(0, int(round(center_y - band_height / 2.0)))
    bottom = min(height, top + band_height)
    if right > left and bottom > top:
        draw.rectangle((left, top, right, bottom), fill=255)
    core = core.filter(ImageFilter.GaussianBlur(0.8))
    return core.point(lambda p: 255 if p >= 18 else 0)


def _build_region_support_mask(
    crop_width: int,
    crop_height: int,
    *,
    region: dict | None = None,
    box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    if not region or box is None:
        return _build_text_core_mask(crop_width, crop_height)

    try:
        local_x0 = int(region["x"]) - int(box[0])
        local_y0 = int(region["y"]) - int(box[1])
        local_x1 = local_x0 + int(region["w"])
        local_y1 = local_y0 + int(region["h"])
    except Exception:
        return _build_text_core_mask(crop_width, crop_height)

    pad_x = max(4, int(round(max(1, local_x1 - local_x0) * 0.16)))
    pad_y = max(3, int(round(max(1, local_y1 - local_y0) * 0.34)))
    x0 = max(0, local_x0 - pad_x)
    y0 = max(0, local_y0 - pad_y)
    x1 = min(crop_width, local_x1 + pad_x)
    y1 = min(crop_height, local_y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return _build_text_core_mask(crop_width, crop_height)

    support = Image.new("L", (crop_width, crop_height), 0)
    draw = ImageDraw.Draw(support)
    draw.rectangle((x0, y0, x1, y1), fill=255)

    focused_band = ImageChops.multiply(_build_text_band_mask(crop_width, crop_height), support)
    focused_band_ratio = _mask_ratio(focused_band)
    if 0.002 <= focused_band_ratio <= 0.18:
        return focused_band.point(lambda p: 255 if p >= 18 else 0)

    focused_core = ImageChops.multiply(_build_text_core_mask(crop_width, crop_height), support)
    focused_core_ratio = _mask_ratio(focused_core)
    if focused_core_ratio >= 0.002:
        return focused_core.point(lambda p: 255 if p >= 18 else 0)

    if focused_band_ratio >= 0.002:
        return focused_band.point(lambda p: 255 if p >= 18 else 0)

    stripe_height = max(12, int(round((y1 - y0) * 0.44)))
    stripe_y0 = max(0, int(round((y0 + y1 - stripe_height) / 2.0)))
    stripe_y1 = min(crop_height, stripe_y0 + stripe_height)
    stripe = Image.new("L", (crop_width, crop_height), 0)
    stripe_draw = ImageDraw.Draw(stripe)
    stripe_draw.rectangle((x0, stripe_y0, x1, stripe_y1), fill=255)
    return stripe


def _ensure_min_candidate_coverage(
    candidate: Image.Image,
    crop_width: int,
    crop_height: int,
    *,
    min_ratio: float = 0.028,
    max_ratio: float = 0.24,
) -> Image.Image:
    ratio = _mask_ratio(candidate)
    if ratio >= min_ratio:
        return candidate

    fallback_candidates = [
        _build_text_band_mask(crop_width, crop_height),
        _build_text_core_mask(crop_width, crop_height),
    ]
    for fallback in fallback_candidates:
        boosted = ImageChops.lighter(candidate, fallback)
        boosted_ratio = _mask_ratio(boosted)
        if min_ratio <= boosted_ratio <= max_ratio:
            return boosted
        fallback_ratio = _mask_ratio(fallback)
        if min_ratio <= fallback_ratio <= max_ratio:
            return fallback

    for fallback in reversed(fallback_candidates):
        boosted = ImageChops.lighter(candidate, fallback)
        boosted_ratio = _mask_ratio(boosted)
        if boosted_ratio > 0:
            return boosted
        fallback_ratio = _mask_ratio(fallback)
        if fallback_ratio > 0:
            return fallback
    return candidate


def _postprocess_candidate(mask: Image.Image, crop_width: int, crop_height: int) -> Image.Image:
    from services.iopaint_runner import _filter_text_like_components
    import numpy as np

    candidate_np = _filter_text_like_components(
        np.asarray(mask, dtype=np.uint8),
        crop_width,
        crop_height,
    )
    candidate = Image.fromarray(candidate_np, mode="L")
    candidate = candidate.filter(ImageFilter.MaxFilter(3))
    candidate = candidate.filter(ImageFilter.GaussianBlur(0.8))
    return candidate.point(lambda p: 255 if p >= 18 else 0)


def _trim_candidate_to_projection(mask: Image.Image, crop_width: int, crop_height: int) -> Image.Image:
    import numpy as np

    candidate_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if candidate_np.max() == 0:
        return mask

    row_hits = candidate_np.sum(axis=1)
    col_hits = candidate_np.sum(axis=0)
    if row_hits.max() <= 0 or col_hits.max() <= 0:
        return mask

    row_threshold = max(2, int(row_hits.max() * 0.16))
    col_threshold = max(2, int(col_hits.max() * 0.08))
    row_idx = np.where(row_hits >= row_threshold)[0]
    col_idx = np.where(col_hits >= col_threshold)[0]
    if row_idx.size == 0 or col_idx.size == 0:
        return mask

    pad_y = max(3, int(round(crop_height * 0.08)))
    pad_x = max(6, int(round(crop_width * 0.06)))
    y0 = max(0, int(row_idx[0]) - pad_y)
    y1 = min(crop_height, int(row_idx[-1]) + 1 + pad_y)
    x0 = max(0, int(col_idx[0]) - pad_x)
    x1 = min(crop_width, int(col_idx[-1]) + 1 + pad_x)
    if x1 <= x0 or y1 <= y0:
        return mask

    focus = Image.new("L", (crop_width, crop_height), 0)
    ImageDraw.Draw(focus).rectangle((x0, y0, x1, y1), fill=255)
    trimmed = ImageChops.multiply(mask.convert("L"), focus)
    return trimmed.point(lambda p: 255 if p >= 18 else 0)


def _compose_temporal_hf_candidate(
    hf_crop: Image.Image,
    temporal_crop: Image.Image,
    crop_width: int,
    crop_height: int,
) -> Image.Image:
    temporal_candidate = _trim_candidate_to_projection(
        _postprocess_candidate(temporal_crop, crop_width, crop_height),
        crop_width,
        crop_height,
    )
    hf_candidate = _trim_candidate_to_projection(
        _postprocess_candidate(hf_crop, crop_width, crop_height),
        crop_width,
        crop_height,
    )

    temporal_ratio = _mask_ratio(temporal_candidate)
    hf_ratio = _mask_ratio(hf_candidate)
    if temporal_ratio <= 0 and hf_ratio <= 0:
        return Image.new("L", (crop_width, crop_height), 0)
    if temporal_ratio <= 0:
        return _ensure_min_candidate_coverage(hf_candidate, crop_width, crop_height, min_ratio=0.010, max_ratio=0.18)
    if hf_ratio <= 0:
        return _ensure_min_candidate_coverage(temporal_candidate, crop_width, crop_height, min_ratio=0.010, max_ratio=0.18)

    temporal_support = _dilate_mask(temporal_candidate, 2)
    guided_hf = Image.composite(hf_candidate, Image.new("L", hf_candidate.size, 0), temporal_support)
    candidate = ImageChops.lighter(temporal_candidate, guided_hf)
    candidate = _trim_candidate_to_projection(candidate, crop_width, crop_height)
    candidate = _postprocess_candidate(candidate, crop_width, crop_height)

    candidate_ratio = _mask_ratio(candidate)
    if candidate_ratio > 0.18:
        candidate = temporal_candidate
    elif candidate_ratio < 0.003 and hf_ratio > temporal_ratio:
        candidate = hf_candidate
    return _ensure_min_candidate_coverage(candidate, crop_width, crop_height, min_ratio=0.010, max_ratio=0.18)


def _build_temporal_region_fallback_candidate(
    frame_crop: Image.Image,
    crop_width: int,
    crop_height: int,
    *,
    region: dict | None = None,
    box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    from services.iopaint_runner import _refine_region_mask

    glyph_candidate = _trim_candidate_to_projection(
        _postprocess_candidate(_refine_region_mask(frame_crop), crop_width, crop_height),
        crop_width,
        crop_height,
    )
    support_candidate = _build_region_support_mask(
        crop_width,
        crop_height,
        region=region,
        box=box,
    )
    candidate = ImageChops.lighter(glyph_candidate, support_candidate)
    candidate_ratio = _mask_ratio(candidate)
    support_ratio = _mask_ratio(support_candidate)
    if candidate_ratio <= 0:
        candidate = support_candidate
    elif candidate_ratio > 0.20 and 0 < support_ratio < candidate_ratio:
        candidate = support_candidate
    return _ensure_min_candidate_coverage(
        candidate,
        crop_width,
        crop_height,
        min_ratio=0.006,
        max_ratio=0.16,
    )


def generate_hf_segmenter_mask(
    reference_frame_path,
    out_path,
    *,
    width: int,
    height: int,
    regions: list[dict] | None = None,
    padding: int = 12,
    dilate: int = 4,
    threshold: float = 0.45,
    weights_name: str = HF_WATERMARK_DEFAULT_WEIGHTS,
):
    reference_frame_path = Path(reference_frame_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_reference_image(reference_frame_path, width, height) as image:
        probability_mask = _predict_probability_mask(image, weights_name=weights_name)
        hard_mask = _threshold_mask(probability_mask, threshold)
        mask = Image.new("L", (width, height), 0)

        for box in _iter_region_boxes(width, height, regions, padding):
            x0, y0, x1, y1 = box
            hard_crop = hard_mask.crop(box)
            candidate = _postprocess_candidate(hard_crop, x1 - x0, y1 - y0)
            if _mask_ratio(candidate) > 0.24:
                candidate = hard_crop
            candidate = _ensure_min_candidate_coverage(candidate, x1 - x0, y1 - y0)
            existing = mask.crop(box)
            mask.paste(ImageChops.lighter(existing, candidate), (x0, y0))

    mask = _dilate_mask(mask, dilate)
    mask.save(out_path)
    return out_path


def generate_hybrid_segmenter_mask(
    reference_frame_path,
    out_path,
    *,
    width: int,
    height: int,
    regions: list[dict] | None = None,
    padding: int = 12,
    dilate: int = 4,
    threshold: float = 0.45,
    weights_name: str = HF_WATERMARK_DEFAULT_WEIGHTS,
):
    from services.iopaint_runner import _refine_region_mask

    reference_frame_path = Path(reference_frame_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_reference_image(reference_frame_path, width, height) as image:
        probability_mask = _predict_probability_mask(image, weights_name=weights_name)
        hard_mask = _threshold_mask(probability_mask, threshold)
        soft_mask = _threshold_mask(probability_mask, max(0.12, threshold * 0.55))
        mask = Image.new("L", (width, height), 0)

        for box in _iter_region_boxes(width, height, regions, padding):
            x0, y0, x1, y1 = box
            frame_crop = image.crop(box)
            glyph_mask = _refine_region_mask(frame_crop)
            hard_crop = hard_mask.crop(box)
            soft_crop = soft_mask.crop(box)
            glyph_expanded = _dilate_mask(glyph_mask, 3)
            guided_soft = Image.composite(soft_crop, Image.new("L", soft_crop.size, 0), glyph_expanded)

            candidate = ImageChops.lighter(hard_crop, glyph_mask)
            candidate = ImageChops.lighter(candidate, guided_soft)
            candidate = _postprocess_candidate(candidate, x1 - x0, y1 - y0)

            candidate_ratio = _mask_ratio(candidate)
            glyph_ratio = _mask_ratio(glyph_mask)
            hard_ratio = _mask_ratio(hard_crop)

            if candidate_ratio > 0.26:
                if glyph_ratio > 0:
                    candidate = _dilate_mask(glyph_mask, 2)
                else:
                    candidate = hard_crop
            elif candidate_ratio < 0.001:
                candidate = hard_crop if hard_ratio > 0 else glyph_mask

            if _mask_ratio(candidate) > 0.28 and hard_ratio > 0:
                candidate = hard_crop
            candidate = _ensure_min_candidate_coverage(candidate, x1 - x0, y1 - y0)

            existing = mask.crop(box)
            mask.paste(ImageChops.lighter(existing, candidate), (x0, y0))

    mask = _dilate_mask(mask, dilate)
    mask.save(out_path)
    return out_path


def generate_temporal_hf_segmenter_mask(
    input_video,
    reference_frame_path,
    out_path,
    *,
    width: int,
    height: int,
    duration: float,
    regions: list[dict] | None = None,
    work_dir,
    padding: int = 12,
    dilate: int = 4,
    threshold: float = 0.45,
    weights_name: str = HF_WATERMARK_DEFAULT_WEIGHTS,
    samples: int = 6,
    min_hits: int = 2,
    register_process=None,
):
    from services.iopaint_runner import generate_temporal_mask

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    hf_mask_path = work_dir / "mask_hf.png"
    temporal_mask_path = work_dir / "mask_temporal.png"
    generate_hf_segmenter_mask(
        reference_frame_path,
        hf_mask_path,
        width=width,
        height=height,
        regions=regions,
        padding=padding,
        dilate=dilate,
        threshold=threshold,
        weights_name=weights_name,
    )
    generate_temporal_mask(
        input_video,
        width,
        height,
        duration,
        regions or [],
        temporal_mask_path,
        work_dir=work_dir,
        padding=padding,
        dilate=dilate,
        samples=samples,
        min_hits=min_hits,
        register_process=register_process,
    )

    mask = Image.new("L", (width, height), 0)
    with (
        _open_reference_image(reference_frame_path, width, height) as reference_image,
        Image.open(hf_mask_path) as hf_mask_image,
        Image.open(temporal_mask_path) as temporal_mask_image,
    ):
        hf_mask = hf_mask_image.convert("L")
        temporal_mask = temporal_mask_image.convert("L")
        for region, box in _iter_region_entries(width, height, regions, padding):
            x0, y0, x1, y1 = box
            crop_width = x1 - x0
            crop_height = y1 - y0
            candidate = _compose_temporal_hf_candidate(
                hf_mask.crop(box),
                temporal_mask.crop(box),
                crop_width,
                crop_height,
            )
            candidate_ratio = _mask_ratio(candidate)
            if candidate_ratio < 0.003:
                frame_crop = reference_image.crop(box)
                fallback_candidate = _build_temporal_region_fallback_candidate(
                    frame_crop,
                    crop_width,
                    crop_height,
                    region=region,
                    box=box,
                )
                fallback_ratio = _mask_ratio(fallback_candidate)
                if candidate_ratio <= 0:
                    candidate = fallback_candidate
                elif fallback_ratio > 0:
                    candidate = ImageChops.lighter(candidate, fallback_candidate)
                else:
                    candidate = _ensure_min_candidate_coverage(
                        candidate,
                        crop_width,
                        crop_height,
                        min_ratio=0.006,
                        max_ratio=0.16,
                    )
            existing = mask.crop(box)
            mask.paste(ImageChops.lighter(existing, candidate), (x0, y0))
    mask = _dilate_mask(mask, dilate)
    mask.save(out_path)
    return out_path
