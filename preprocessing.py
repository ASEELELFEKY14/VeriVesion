import cv2
import numpy as np
import albumentations as A
import torch
from dataclasses import dataclass
from albumentations.pytorch import ToTensorV2

from config import CROP_FACTOR, IMG_SIZE

try:
    from config import (
        MAX_BRIGHTNESS,
        MAX_DARK_RATIO,
        MAX_LARGE_DARK_REGION_RATIO,
        MAX_SPECULAR_RATIO,
        MIN_BRIGHTNESS,
        MIN_FRAME_CONTRAST,
        MIN_FRAME_SHARPNESS,
        MIN_TISSUE_RATIO,
    )
except ImportError:
    MIN_BRIGHTNESS = 30.0
    MAX_BRIGHTNESS = 240.0
    MIN_FRAME_SHARPNESS = 35.0
    MIN_FRAME_CONTRAST = 18.0
    MIN_TISSUE_RATIO = 0.35
    MAX_SPECULAR_RATIO = 0.10
    MAX_DARK_RATIO = 0.45
    MAX_LARGE_DARK_REGION_RATIO = 0.18


@dataclass
class FrameQuality:
    accepted: bool
    score: float
    reason: str
    sharpness: float
    brightness: float
    contrast: float
    tissue_ratio: float
    specular_ratio: float
    dark_ratio: float
    large_dark_region_ratio: float


def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 1.2,
    grid_size: tuple = (8, 8),
    **kwargs
):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size
    )

    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def detect_viewport(image_rgb: np.ndarray) -> np.ndarray:
    """
    Detect the visible endoscopy viewport and remove
    surrounding black borders automatically.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(
        gray,
        15,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((7, 7), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return image_rgb

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    padding = 5

    x = max(0, x - padding)
    y = max(0, y - padding)

    w = min(image_rgb.shape[1] - x, w + 2 * padding)
    h = min(image_rgb.shape[0] - y, h + 2 * padding)

    return image_rgb[y:y + h, x:x + w]


def crop_roi(
    image_rgb: np.ndarray,
    crop_factor: float = CROP_FACTOR
):
    image_rgb = detect_viewport(image_rgb)

    h, w = image_rgb.shape[:2]

    size = int(min(h, w) * crop_factor)

    left = (w - size) // 2
    top = (h - size) // 2

    roi = image_rgb[
        top:top + size,
        left:left + size
    ]

    roi = cv2.resize(
        roi,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LANCZOS4
    )

    return roi


def _quality_roi(
    image_rgb: np.ndarray,
    crop_factor: float = CROP_FACTOR
) -> np.ndarray:
    image_rgb = detect_viewport(image_rgb)

    h, w = image_rgb.shape[:2]

    size = int(min(h, w) * crop_factor)

    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)

    return image_rgb[top:top + size, left:left + size]


def evaluate_frame_quality(
    frame_bgr: np.ndarray,
    strict: bool = True
) -> FrameQuality:
    """
    Used for video frames only.
    It rejects frames that are mostly glare, black lumen, blurry, or low tissue content.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return FrameQuality(
            False, 0.0, "empty frame",
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0
        )

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    roi_rgb = _quality_roi(image_rgb)

    if roi_rgb is None or roi_rgb.size == 0:
        return FrameQuality(
            False, 0.0, "empty roi",
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0
        )

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

    total_pixels = float(gray.size)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    specular_ratio = float(np.mean(gray >= 245))
    dark_ratio = float(np.mean(gray <= 25))

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    tissue_mask = (
        (saturation >= 25)
        & (value >= 35)
        & (value <= 245)
    )

    tissue_ratio = float(np.mean(tissue_mask))

    dark_mask = (gray <= 35).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        dark_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    large_dark_region_ratio = 0.0

    if contours:
        largest_dark_area = max(cv2.contourArea(c) for c in contours)
        large_dark_region_ratio = float(largest_dark_area / total_pixels)

    brightness_min = MIN_BRIGHTNESS if strict else 15.0
    brightness_max = MAX_BRIGHTNESS if strict else 250.0

    min_sharpness = MIN_FRAME_SHARPNESS if strict else 8.0
    min_contrast = MIN_FRAME_CONTRAST if strict else 8.0
    min_tissue_ratio = MIN_TISSUE_RATIO if strict else 0.18

    max_specular_ratio = MAX_SPECULAR_RATIO if strict else 0.22
    max_dark_ratio = MAX_DARK_RATIO if strict else 0.65
    max_large_dark = MAX_LARGE_DARK_REGION_RATIO if strict else 0.35

    checks = [
        (brightness_min <= brightness <= brightness_max, "bad brightness"),
        (sharpness >= min_sharpness, "blurry"),
        (contrast >= min_contrast, "low contrast"),
        (tissue_ratio >= min_tissue_ratio, "low tissue content"),
        (specular_ratio <= max_specular_ratio, "too much glare"),
        (dark_ratio <= max_dark_ratio, "too dark"),
        (large_dark_region_ratio <= max_large_dark, "large dark lumen"),
    ]

    accepted = True
    reason = "accepted"

    for ok, failed_reason in checks:
        if not ok:
            accepted = False
            reason = failed_reason
            break

    score = (
        min(sharpness / 160.0, 1.0) * 0.30
        + min(contrast / 55.0, 1.0) * 0.20
        + min(tissue_ratio / 0.75, 1.0) * 0.25
        + (1.0 - min(specular_ratio / max(max_specular_ratio, 1e-6), 1.0)) * 0.15
        + (1.0 - min(large_dark_region_ratio / max(max_large_dark, 1e-6), 1.0)) * 0.10
    )

    return FrameQuality(
        accepted=accepted,
        score=float(score),
        reason=reason,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        tissue_ratio=tissue_ratio,
        specular_ratio=specular_ratio,
        dark_ratio=dark_ratio,
        large_dark_region_ratio=large_dark_region_ratio,
    )


def build_inference_transform():
    return A.Compose([
        A.Lambda(image=apply_clahe),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def preprocess_frame(frame_bgr, transform):
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    roi_rgb = crop_roi(frame_rgb)

    tensor = transform(
        image=roi_rgb
    )["image"]

    return tensor, roi_rgb