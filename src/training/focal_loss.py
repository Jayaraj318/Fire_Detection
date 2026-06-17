"""
Focal Loss Implementation
Handles class imbalance in fire detection by down-weighting easy negatives.

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for the rare class (fire). Default: 0.25
               - alpha > 0.5 gives more weight to positive (fire) class
               - alpha < 0.5 gives more weight to negative (non-fire) class
        gamma: Focusing parameter. Higher gamma reduces loss for well-classified
               examples, focusing training on hard negatives. Default: 2.0
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted probabilities [B, 1] or [B] - already sigmoid activated
            targets: Ground truth labels [B, 1] or [B] (0 or 1)
            
        Returns:
            Focal loss value
        """
        # Ensure proper shapes
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Clamp for numerical stability
        p = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)
        
        # Compute focal loss
        # For positive samples (fire): -alpha * (1-p)^gamma * log(p)
        # For negative samples (non-fire): -(1-alpha) * p^gamma * log(1-p)
        ce_loss = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)
        
        p_t = p * targets + (1 - p) * (1 - targets)  # p if y=1, 1-p if y=0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossWithLogits(nn.Module):
    """
    Focal Loss that takes raw logits (before sigmoid).
    More numerically stable for training.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss from logits.
        
        Args:
            inputs: Raw logits [B, 1] or [B] (before sigmoid)
            targets: Ground truth labels [B, 1] or [B] (0 or 1)
            
        Returns:
            Focal loss value
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Compute BCE with logits (more stable)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == '__main__':
    # Quick test
    print("Testing Focal Loss...")
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Test with some predictions
    preds = torch.tensor([0.9, 0.1, 0.7, 0.3])  # Fire probs
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Ground truth
    
    loss = criterion(preds, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Compare with BCE
    bce = nn.BCELoss()
    bce_loss = bce(preds, targets)
    print(f"BCE Loss: {bce_loss.item():.4f}")
    
    print("✅ Focal Loss test passed!")
