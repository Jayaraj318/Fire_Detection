"""
Temporal Fusion Module
Smooth predictions across time using exponential moving average
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Tuple, Optional


class TemporalFusionModule(nn.Module):
    """
    Temporal fusion for smoothing fire detection across frames.
    
    Uses:
    1. Exponential Moving Average (EMA) for feature smoothing
    2. Temporal validation requiring N consecutive positive detections
    
    Formula: F_t = α * F(t) + (1-α) * F_{t-1}
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        buffer_size: int = 3,
        min_consecutive: int = 3,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize temporal fusion.
        
        Args:
            alpha: EMA smoothing factor (0-1). Higher = more weight on current frame.
            buffer_size: Size of confidence buffer for temporal analysis.
            min_consecutive: Minimum consecutive positive frames required.
            confidence_threshold: Threshold for positive detection.
        """
        super().__init__()
        
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.min_consecutive = min_consecutive
        self.confidence_threshold = confidence_threshold
        
        # State (not parameters, just runtime state)
        self.register_buffer('prev_fused', None, persistent=False)
        self.confidence_buffer = deque(maxlen=buffer_size)
        
        # Detection state counters
        self.consecutive_fire = 0
        self.consecutive_no_fire = 0
    
    def forward(
        self,
        current_features: torch.Tensor,
        current_confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        Apply temporal smoothing and validation.
        
        Args:
            current_features: Current frame features [B, D]
            current_confidence: Fire probability [B, 1] or [B]
            
        Returns:
            - Smoothed features [B, D]
            - Fire detection decision (bool)
        """
        # Ensure confidence is 2D
        if current_confidence.dim() == 1:
            current_confidence = current_confidence.unsqueeze(1)
        
        # Apply EMA smoothing to features
        if self.prev_fused is None:
            smoothed = current_features
        else:
            # Ensure dimensions match
            if self.prev_fused.shape != current_features.shape:
                smoothed = current_features
            else:
                smoothed = (self.alpha * current_features + 
                           (1 - self.alpha) * self.prev_fused)
        
        # Update state
        self.prev_fused = smoothed.detach()
        
        # Update confidence buffer
        mean_conf = current_confidence.mean().item()
        self.confidence_buffer.append(mean_conf)
        
        # Temporal validation
        fire_detected = self._validate_temporal(mean_conf)
        
        return smoothed, fire_detected
    
    def _validate_temporal(self, confidence: float) -> bool:
        """
        Validate detection with temporal consistency.
        
        Requires N consecutive frames above threshold to confirm fire.
        
        Args:
            confidence: Current frame confidence (0-1).
            
        Returns:
            True if fire is confirmed, False otherwise.
        """
        if confidence > self.confidence_threshold:
            self.consecutive_fire += 1
            self.consecutive_no_fire = 0
        else:
            self.consecutive_no_fire += 1
            self.consecutive_fire = 0
        
        # Fire is confirmed only after min_consecutive positive frames
        return self.consecutive_fire >= self.min_consecutive
    
    def reset(self):
        """Reset temporal state for new video/stream."""
        self.prev_fused = None
        self.confidence_buffer.clear()
        self.consecutive_fire = 0
        self.consecutive_no_fire = 0
    
    def get_temporal_stats(self) -> dict:
        """Get current temporal statistics."""
        return {
            'consecutive_fire': self.consecutive_fire,
            'consecutive_no_fire': self.consecutive_no_fire,
            'buffer_mean': sum(self.confidence_buffer) / len(self.confidence_buffer) if self.confidence_buffer else 0,
            'buffer_size': len(self.confidence_buffer)
        }


class AdaptiveTemporalFusion(nn.Module):
    """
    Adaptive temporal fusion that learns the smoothing factor.
    
    Instead of fixed alpha, learns to predict optimal smoothing
    based on feature similarity between frames.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        min_consecutive: int = 3,
        confidence_threshold: float = 0.5
    ):
        super().__init__()
        
        self.min_consecutive = min_consecutive
        self.confidence_threshold = confidence_threshold
        
        # Network to predict alpha based on frame similarity
        self.alpha_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.register_buffer('prev_fused', None, persistent=False)
        self.confidence_buffer = deque(maxlen=5)
        self.consecutive_fire = 0
        self.consecutive_no_fire = 0
    
    def forward(
        self,
        current_features: torch.Tensor,
        current_confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, bool, float]:
        """
        Apply adaptive temporal smoothing.
        
        Returns:
            - Smoothed features
            - Fire detection decision
            - Predicted alpha value
        """
        if current_confidence.dim() == 1:
            current_confidence = current_confidence.unsqueeze(1)
        
        if self.prev_fused is None:
            smoothed = current_features
            alpha = 1.0
        else:
            # Predict alpha based on frame similarity
            concat = torch.cat([current_features, self.prev_fused], dim=1)
            alpha = self.alpha_predictor(concat).squeeze(-1)
            
            # Apply learned EMA
            alpha_expanded = alpha.unsqueeze(-1)
            smoothed = alpha_expanded * current_features + (1 - alpha_expanded) * self.prev_fused
            alpha = alpha.mean().item()
        
        self.prev_fused = smoothed.detach()
        
        # Temporal validation
        mean_conf = current_confidence.mean().item()
        self.confidence_buffer.append(mean_conf)
        fire_detected = self._validate_temporal(mean_conf)
        
        return smoothed, fire_detected, alpha
    
    def _validate_temporal(self, confidence: float) -> bool:
        if confidence > self.confidence_threshold:
            self.consecutive_fire += 1
            self.consecutive_no_fire = 0
        else:
            self.consecutive_no_fire += 1
            self.consecutive_fire = 0
        
        return self.consecutive_fire >= self.min_consecutive
    
    def reset(self):
        self.prev_fused = None
        self.confidence_buffer.clear()
        self.consecutive_fire = 0
        self.consecutive_no_fire = 0


class ConfidenceAggregator(nn.Module):
    """
    Aggregate confidence scores over a sliding window.
    
    Useful for computing moving average confidence and detecting trends.
    """
    
    def __init__(self, window_size: int = 5):
        super().__init__()
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    
    def update(self, confidence: float) -> float:
        """Add confidence and return moving average."""
        self.buffer.append(confidence)
        return self.get_average()
    
    def get_average(self) -> float:
        """Get current moving average."""
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)
    
    def get_trend(self) -> str:
        """Get confidence trend (increasing/decreasing/stable)."""
        if len(self.buffer) < 3:
            return "stable"
        
        recent = list(self.buffer)[-3:]
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_diff = sum(diffs) / len(diffs)
        
        if avg_diff > 0.05:
            return "increasing"
        elif avg_diff < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def reset(self):
        self.buffer.clear()


if __name__ == '__main__':
    # Test Temporal Fusion
    print("Testing TemporalFusionModule...")
    
    temporal = TemporalFusionModule(min_consecutive=3)
    
    # Simulate a sequence with fire appearing
    features = torch.randn(1, 256)
    confidences = [0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]
    
    print("Simulating fire detection sequence:")
    for i, conf in enumerate(confidences):
        conf_tensor = torch.tensor([[conf]])
        smoothed, detected = temporal(features, conf_tensor)
        print(f"  Frame {i+1}: conf={conf:.2f}, detected={detected}, "
              f"consecutive={temporal.consecutive_fire}")
    
    # Test adaptive temporal fusion
    print("\nTesting AdaptiveTemporalFusion...")
    adaptive = AdaptiveTemporalFusion()
    
    for i, conf in enumerate(confidences):
        conf_tensor = torch.tensor([[conf]])
        features = torch.randn(1, 256)
        smoothed, detected, alpha = adaptive(features, conf_tensor)
        print(f"  Frame {i+1}: conf={conf:.2f}, alpha={alpha:.3f}, detected={detected}")
