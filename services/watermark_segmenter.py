from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

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
    model, transform, torch, device = _load_segmenter()
    reference_frame_path = Path(reference_frame_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(reference_frame_path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor))[0, 0].detach().float().cpu().numpy()

    prob = Image.fromarray((prediction * 255).clip(0, 255).astype("uint8"), mode="L")
    prob = prob.resize((width, height), Image.Resampling.BILINEAR)
    mask = prob.point(lambda p: 255 if p >= int(max(0.0, min(1.0, threshold)) * 255) else 0)

    limit = _build_region_limit((width, height), regions, padding)
    if limit is not None:
        mask = Image.composite(mask, Image.new("L", (width, height), 0), limit)

    if dilate > 0:
        kernel = max(3, int(dilate) * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        mask = mask.filter(ImageFilter.MaxFilter(kernel))
        mask = mask.point(lambda p: 255 if p >= 18 else 0)
    mask.save(out_path)
    return out_path
