"""
YOLO Feature Extractor
Extract intermediate features from YOLO for multi-modal fusion
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, Any, Optional


class YOLOFeatureExtractor(nn.Module):
    """
    Extract intermediate features from YOLO for fusion with ViT and Optical Flow.
    
    Uses hooks to capture features from the backbone before the detection head.
    """
    
    def __init__(self, model_path: Optional[str] = None, pretrained: bool = True):
        """
        Initialize the YOLO feature extractor.
        
        Args:
            model_path: Path to trained YOLO weights. If None, uses pretrained weights.
            pretrained: Whether to use pretrained weights (only if model_path is None).
        """
        super().__init__()
        
        if model_path:
            self.yolo = YOLO(model_path)
        else:
            self.yolo = YOLO('yolov8n.pt' if pretrained else 'yolov8n.yaml')
        
        self.model = self.yolo.model
        # Freeze YOLO and keep in eval mode to prevent train() conflict
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.features = None
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook to extract features from backbone."""
        def hook(module, input, output):
            self.features = output
        
        # Register hook on layer 9 (SPPF) which contains high-level features
        # For YOLOv8n, Layer 9 outputs 256 channels
        try:
            self.model.model[9].register_forward_hook(hook)
            print("   ✅ Hook registered on Layer 9 (SPPF)")
        except IndexError:
            # Fallback
            print("   ⚠️ Layer 9 not found, falling back to last Conv2d")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            last_conv.register_forward_hook(hook)
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Extract features and detections from YOLO.
        
        Args:
            x: Input tensor [B, 3, H, W] normalized to [0, 1]
            
        Returns:
            Dictionary containing:
                - 'features': Intermediate features [B, C, H', W']
                - 'boxes': Detected bounding boxes (xyxy format)
                - 'conf': Detection confidences
                - 'cls': Class predictions
        """
        # Convert to numpy for YOLO inference
        if x.requires_grad:
            x_detached = x.detach()
        else:
            x_detached = x
            
        # Clear previous features
        self.features = None
        
        # Run YOLO inference
        # verbose=False prevents printing to stdout
        results = self.yolo(x_detached, verbose=False)
        
        # Get detections from first result
        result = results[0]
        
        return {
            'features': self.features,
            'boxes': result.boxes.xyxy if result.boxes else torch.tensor([]),
            'conf': result.boxes.conf if result.boxes else torch.tensor([]),
            'cls': result.boxes.cls if result.boxes else torch.tensor([])
        }
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return 256  # YOLOv8n SPPF output channels
    
    def train(self, mode: bool = True):
        """Override train() to prevent calling YOLO's train() method."""
        # Only set training mode on the module itself, not on YOLO submodule
        # YOLO stays in eval mode permanently
        if not mode:
            super().train(False)
        # Return self for backwards compatibility
        return self


class YOLOFeaturePooler(nn.Module):
    """
    Pool YOLO spatial features to a fixed-size vector for fusion.
    """
    
    def __init__(self, in_channels: int = 256, out_channels: int = 512):
        super().__init__()
        
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool spatial features to vector.
        
        Args:
            features: [B, C, H, W]
            
        Returns:
            Pooled features [B, out_channels]
        """
        return self.pool(features)


if __name__ == '__main__':
    # Test the feature extractor
    extractor = YOLOFeatureExtractor()
    pooler = YOLOFeaturePooler()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Extract features
    output = extractor(dummy_input)
    
    print(f"Features shape: {output['features'].shape}")
    print(f"Boxes: {output['boxes']}")
    print(f"Confidences: {output['conf']}")
    
    # Pool features
    pooled = pooler(output['features'])
    print(f"Pooled features shape: {pooled.shape}")
