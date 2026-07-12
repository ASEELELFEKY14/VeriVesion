from dataclasses import dataclass
import cv2 as cv
import numpy as np
import torch

from config import CLASS_LABELS, NEGATIVE_CLASS, POSITIVE_CLASS, THRESHOLD
from .preprocessing import build_inference_transform, preprocess_frame
from .visualizer import GradCAM, generate_heatmap_overlay

@dataclass
class ImageAnalysisResult:
    probability: float
    predicted_class: int
    label: str
    image_bgr: np.ndarray
    roi_rgb: np.ndarray
    heatmap_rgb: np.ndarray
    metadata: dict

def analyze_image(
    model: torch.nn.Module,
    device: str,
    image_bgr: np.ndarray,
    metadata: dict = None,
) -> ImageAnalysisResult:
    
    transform = build_inference_transform()
    
    
    tensor, roi_rgb = preprocess_frame(image_bgr, transform)
    tensor = tensor.unsqueeze(0).to(device)
    
   
    target_layer = model.layer4[-1]
    cam_extractor = GradCAM(model, target_layer)
    
    model.eval()
    
    output = model(tensor)
    probability = torch.sigmoid(output).item()

    predicted_class = POSITIVE_CLASS if probability >= THRESHOLD else NEGATIVE_CLASS

    
    cam_mask = cam_extractor.generate(tensor)
    heatmap_rgb = generate_heatmap_overlay(roi_rgb, cam_mask)

    return ImageAnalysisResult(
        probability=probability,
        predicted_class=predicted_class,
        label=CLASS_LABELS[predicted_class],
        image_bgr=image_bgr,
        roi_rgb=roi_rgb,
        heatmap_rgb=heatmap_rgb,
        metadata=metadata or {},
    )