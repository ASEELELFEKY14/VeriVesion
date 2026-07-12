import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # For a single-node binary classifier (sigmoid output)
        score = output[0, 0]
        score.backward(retain_graph=True)
        
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Global average pooling on the gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU on the CAM, resize to match input, and normalize
        cam = np.maximum(cam, 0)
        cam = cv.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        return cam

def generate_heatmap_overlay(image_rgb: np.ndarray, cam_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays the Grad-CAM mask on the original RGB image."""
    heatmap = cv.applyColorMap(np.uint8(255 * cam_mask), cv.COLORMAP_JET)
    heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
    
    # Ensure image_rgb is uint8
    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb * 255).astype(np.uint8)
        
    overlay = cv.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay