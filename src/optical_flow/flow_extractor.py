"""
Optical Flow Extractor
Extract temporal motion features using Farneback optical flow
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class OpticalFlowExtractor(nn.Module):
    """
    Extract motion features using Farneback optical flow.
    
    Computes dense optical flow between consecutive frames and encodes
    the flow field into a feature vector for temporal analysis.
    """
    
    def __init__(self, output_dim: int = 256):
        """
        Initialize the optical flow extractor.
        
        Args:
            output_dim: Output feature dimension.
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # Flow CNN encoder to convert flow field to features
        self.flow_encoder = nn.Sequential(
            # Input: 2 channels (u, v motion vectors)
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True)
        )
        
        # Farneback parameters for optimal fire detection
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
    
    def compute_optical_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Farneback optical flow between two frames.
        
        Args:
            frame1: First frame [B, 3, H, W] (RGB, normalized 0-1)
            frame2: Second frame [B, 3, H, W] (RGB, normalized 0-1)
            
        Returns:
            Optical flow field [B, 2, H, W] containing (u, v) motion vectors
        """
        batch_size = frame1.shape[0]
        device = frame1.device
        height, width = frame1.shape[2], frame1.shape[3]
        
        flow_batch = []
        
        for i in range(batch_size):
            # Convert to numpy and grayscale
            img1 = frame1[i].permute(1, 2, 0).cpu().numpy()
            img2 = frame2[i].permute(1, 2, 0).cpu().numpy()
            
            # Convert from 0-1 to 0-255
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)
            
            # Convert RGB to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # Compute Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                **self.flow_params
            )
            
            # Convert to tensor [2, H, W]
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()
            flow_batch.append(flow_tensor)
        
        return torch.stack(flow_batch).to(device)
    
    def forward(
        self,
        frame_sequence: torch.Tensor,
        compute_between: Tuple[int, int] = (-2, -1)
    ) -> torch.Tensor:
        """
        Extract motion features from a sequence of frames.
        
        Args:
            frame_sequence: Tensor [B, T, 3, H, W] where T >= 2
            compute_between: Tuple of frame indices to compute flow between.
                           Default is last two frames.
            
        Returns:
            Motion features [B, output_dim]
        """
        # Get frames to compute flow between
        idx1, idx2 = compute_between
        frame1 = frame_sequence[:, idx1]
        frame2 = frame_sequence[:, idx2]
        
        # Compute optical flow
        flow = self.compute_optical_flow(frame1, frame2)
        
        # Encode flow to features
        flow_features = self.flow_encoder(flow)
        
        return flow_features
    
    def visualize_flow(
        self,
        flow: torch.Tensor,
        as_hsv: bool = True
    ) -> np.ndarray:
        """
        Visualize optical flow as a color image.
        
        Args:
            flow: Flow field [2, H, W] or [B, 2, H, W]
            as_hsv: If True, use HSV color coding. If False, use RGB arrows.
            
        Returns:
            Visualization as numpy array [H, W, 3] (BGR)
        """
        if flow.dim() == 4:
            flow = flow[0]  # Take first batch item
        
        flow_np = flow.permute(1, 2, 0).cpu().numpy()
        
        if as_hsv:
            # HSV visualization (direction = hue, magnitude = value)
            mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
            
            hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2  # Hue
            hsv[..., 1] = 255  # Saturation
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value
            
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # Simple RGB visualization
            flow_rgb = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
            flow_rgb[..., 0] = np.clip(flow_np[..., 0] * 10 + 128, 0, 255)
            flow_rgb[..., 2] = np.clip(flow_np[..., 1] * 10 + 128, 0, 255)
            flow_rgb[..., 1] = 128
            return flow_rgb


class TemporalFlowEncoder(nn.Module):
    """
    Encode multiple optical flow fields across time into a single feature.
    
    Useful for analyzing fire movement patterns over longer sequences.
    """
    
    def __init__(
        self,
        num_flows: int = 2,
        flow_dim: int = 256,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.flow_extractor = OpticalFlowExtractor(output_dim=flow_dim)
        
        self.temporal_fusion = nn.Sequential(
            nn.Linear(flow_dim * num_flows, flow_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(flow_dim, output_dim)
        )
    
    def forward(self, frame_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_sequence: [B, T, 3, H, W] where T >= 3
            
        Returns:
            [B, output_dim]
        """
        T = frame_sequence.shape[1]
        
        # Compute flow between consecutive frame pairs
        flows = []
        for t in range(T - 1):
            flow = self.flow_extractor(frame_sequence, compute_between=(t, t + 1))
            flows.append(flow)
        
        # Concatenate and fuse
        if len(flows) == 1:
            return flows[0]
        
        concat_flows = torch.cat(flows[-2:], dim=1)  # Use last 2 flows
        return self.temporal_fusion(concat_flows)


if __name__ == '__main__':
    # Test the optical flow extractor
    print("Testing OpticalFlowExtractor...")
    
    extractor = OpticalFlowExtractor()
    
    # Create a dummy sequence (3 frames)
    dummy_seq = torch.randn(2, 3, 3, 640, 640)  # [B, T, C, H, W]
    
    # Extract features
    features = extractor(dummy_seq)
    print(f"Flow features shape: {features.shape}")
    
    # Test flow computation
    flow = extractor.compute_optical_flow(dummy_seq[:, 0], dummy_seq[:, 1])
    print(f"Flow field shape: {flow.shape}")
    
    # Test temporal encoder
    print("\nTesting TemporalFlowEncoder...")
    temporal = TemporalFlowEncoder()
    temporal_features = temporal(dummy_seq)
    print(f"Temporal features shape: {temporal_features.shape}")
