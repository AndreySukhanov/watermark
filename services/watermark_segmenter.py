from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

HF_WATERMARK_MODEL = "christophernavas/watermark-remover"
HF_WATERMARK_WEIGHTS = "segmenter.pth"


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


@lru_cache(maxsize=1)
def _load_segmenter():
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
    weights_path = hf_hub_download(HF_WATERMARK_MODEL, HF_WATERMARK_WEIGHTS)
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


def _open_reference_image(reference_frame_path, width: int, height: int) -> Image.Image:
    with Image.open(reference_frame_path) as source_image:
        image = source_image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    return image


def _predict_probability_mask(image: Image.Image) -> Image.Image:
    model, transform, torch, device = _load_segmenter()
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
):
    reference_frame_path = Path(reference_frame_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_reference_image(reference_frame_path, width, height) as image:
        probability_mask = _predict_probability_mask(image)
        hard_mask = _threshold_mask(probability_mask, threshold)
        mask = Image.new("L", (width, height), 0)

        for box in _iter_region_boxes(width, height, regions, padding):
            x0, y0, x1, y1 = box
            hard_crop = hard_mask.crop(box)
            candidate = _postprocess_candidate(hard_crop, x1 - x0, y1 - y0)
            if _mask_ratio(candidate) > 0.24:
                candidate = hard_crop
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
):
    from services.iopaint_runner import _refine_region_mask

    reference_frame_path = Path(reference_frame_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_reference_image(reference_frame_path, width, height) as image:
        probability_mask = _predict_probability_mask(image)
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

            existing = mask.crop(box)
            mask.paste(ImageChops.lighter(existing, candidate), (x0, y0))

    mask = _dilate_mask(mask, dilate)
    mask.save(out_path)
    return out_path
