"""
Grad-CAM Explainability
Generate visual explanations for fire detection decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, Tuple, Union


class FireDetectionGradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for fire detection.
    
    Generates heatmaps showing which regions of the image contributed most
    to the fire detection decision.
    
    Formula: L^c_GradCAM = ReLU(Σ α_k^c * A_k)
    where α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The fire detection model.
            target_layer: The layer to compute Grad-CAM on (typically last conv).
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image [1, 3, H, W] or sequence [1, T, 3, H, W]
            target_class: Target class index (None = use predicted class)
            
        Returns:
            Heatmap as numpy array [H, W] with values 0-1
        """
        self.model.eval()
        
        # Handle sequence input
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        
        # Enable gradients for input
        input_tensor = input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor, return_features=True)
        
        # Get target score
        score = output['confidence'][0, 0]
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Check if we have gradients
        if self.gradients is None or self.activations is None:
            # Fallback to features-based heatmap
            return self._generate_from_features(output, input_tensor.shape[-2:])
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])  # [C]
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        return cam.cpu().numpy()
    
    def _generate_from_features(
        self,
        output: dict,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Generate a simple heatmap from fused features when gradients unavailable.
        """
        if 'features' in output:
            features = output['features'][0].cpu().numpy()
            # Create a simple heatmap based on feature magnitudes
            heatmap = np.abs(features).sum()
            heatmap = np.full(target_size, heatmap / (features.size + 1e-8))
            return heatmap
        return np.zeros(target_size)
    
    def visualize(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: Original image [H, W, 3] (BGR, 0-255)
            heatmap: Grad-CAM heatmap [H, W] (0-1)
            alpha: Overlay transparency
            colormap: OpenCV colormap
            
        Returns:
            Visualization [H, W, 3] (BGR, 0-255)
        """
        # Resize heatmap to match image
        if heatmap.shape[:2] != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            colormap
        )
        
        # Overlay
        visualization = cv2.addWeighted(
            original_image.astype(np.uint8),
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return visualization
    
    def generate_and_visualize(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate heatmap and create visualization in one call.
        
        Returns:
            (heatmap, visualization)
        """
        heatmap = self.generate(input_tensor)
        visualization = self.visualize(original_image, heatmap)
        return heatmap, visualization


class AttentionVisualizer:
    """
    Visualize attention weights from multi-modal fusion.
    """
    
    @staticmethod
    def create_attention_bar(
        weights: torch.Tensor,
        modality_names: list = ['YOLO', 'ViT', 'Optical Flow'],
        bar_height: int = 50,
        bar_width: int = 300
    ) -> np.ndarray:
        """
        Create a bar chart visualization of attention weights.
        
        Args:
            weights: Attention weights [3]
            modality_names: Names of modalities
            bar_height: Height of each bar
            bar_width: Maximum width of bars
            
        Returns:
            Visualization image [H, W, 3] (BGR)
        """
        weights = weights.cpu().numpy() if torch.is_tensor(weights) else weights
        
        height = bar_height * len(modality_names) + 20
        width = bar_width + 150
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        colors = [
            (255, 100, 100),  # Blue for YOLO
            (100, 255, 100),  # Green for ViT
            (100, 100, 255)   # Red for Flow
        ]
        
        for i, (name, weight, color) in enumerate(zip(modality_names, weights, colors)):
            y = 10 + i * bar_height
            
            # Draw bar
            bar_len = int(weight * (bar_width - 50))
            cv2.rectangle(img, (100, y), (100 + bar_len, y + bar_height - 10), color, -1)
            cv2.rectangle(img, (100, y), (100 + bar_width - 50, y + bar_height - 10), (0, 0, 0), 1)
            
            # Draw label
            cv2.putText(img, name, (5, y + bar_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw percentage
            cv2.putText(img, f'{weight:.1%}', (bar_width + 60, y + bar_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img


class FireTypeVisualizer:
    """
    Visualize fire type classification results.
    """
    
    @staticmethod
    def create_fire_type_display(
        probs: torch.Tensor,
        fire_types: list = ['Class A', 'Class B', 'Class C'],
        descriptions: list = ['Combustibles', 'Flammable Liquids', 'Electrical'],
        size: Tuple[int, int] = (300, 150)
    ) -> np.ndarray:
        """
        Create a visualization of fire type classification.
        """
        if probs is None:
            img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
            cv2.putText(img, "No fire detected", (20, size[1] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            return img
        
        probs = probs.cpu().numpy() if torch.is_tensor(probs) else probs
        probs = probs.flatten()
        
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        
        # Find predicted class
        pred_idx = probs.argmax()
        
        for i, (ft, desc, prob) in enumerate(zip(fire_types, descriptions, probs)):
            y = 30 + i * 40
            
            color = (0, 200, 0) if i == pred_idx else (100, 100, 100)
            
            text = f"{ft} ({desc}): {prob:.1%}"
            cv2.putText(img, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 if i != pred_idx else 2)
        
        return img


if __name__ == '__main__':
    print("Grad-CAM module loaded successfully.")
    print("Use FireDetectionGradCAM for generating heatmaps.")
    print("Use AttentionVisualizer for modality attention visualization.")
