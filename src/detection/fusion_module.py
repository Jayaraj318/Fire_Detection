"""
Multi-Modal Fusion Module
Fuse YOLO, ViT, and Optical Flow features using learned attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for multi-modal features.
    
    Learns to weight different modalities based on their relevance
    for fire detection in the current frame.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_modalities: int = 3,
        hidden_dim: int = 128
    ):
        """
        Args:
            input_dim: Dimension of each modality's features.
            num_modalities: Number of modalities to fuse.
            hidden_dim: Hidden dimension for attention computation.
        """
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(input_dim * num_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted fusion.
        
        Args:
            features: [B, num_modalities, input_dim]
            
        Returns:
            - Fused features [B, input_dim]
            - Attention weights [B, num_modalities]
        """
        B = features.shape[0]
        
        # Flatten for attention computation
        flat = features.view(B, -1)  # [B, num_modalities * input_dim]
        
        # Compute attention weights
        weights = self.attention(flat)  # [B, num_modalities]
        
        # Apply weights
        weights_expanded = weights.unsqueeze(-1)  # [B, num_modalities, 1]
        weighted = features * weights_expanded  # [B, num_modalities, input_dim]
        
        # Sum across modalities
        fused = weighted.sum(dim=1)  # [B, input_dim]
        
        return fused, weights


class MultiModalFusion(nn.Module):
    """
    Multi-Modal Fusion for YOLO, ViT, and Optical Flow features.
    
    This is a key component that combines:
    - YOLO: Local spatial features (object detection)
    - ViT: Global context features (scene understanding)
    - Optical Flow: Temporal motion features (flame dynamics)
    """
    
    def __init__(
        self,
        yolo_dim: int = 512,
        vit_dim: int = 512,
        flow_dim: int = 256,
        fusion_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        Initialize the fusion module.
        
        Args:
            yolo_dim: YOLO feature dimension.
            vit_dim: ViT feature dimension.
            flow_dim: Optical flow feature dimension.
            fusion_dim: Internal fusion dimension.
            output_dim: Output feature dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        # Project all modalities to same dimension
        self.yolo_proj = nn.Sequential(
            nn.Linear(yolo_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        self.flow_proj = nn.Sequential(
            nn.Linear(flow_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True)
        )
        
        # Attention-based fusion
        self.attention_fusion = AttentionFusion(
            input_dim=fusion_dim,
            num_modalities=3,
            hidden_dim=256
        )
        
        # Final fusion layers
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(
        self,
        yolo_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        flow_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multi-modal features.
        
        Args:
            yolo_feat: YOLO features [B, yolo_dim]
            vit_feat: ViT features [B, vit_dim]
            flow_feat: Optical flow features [B, flow_dim]
            
        Returns:
            - Fused features [B, output_dim]
            - Attention weights [B, 3] (YOLO, ViT, Flow)
        """
        # Project to common dimension
        yolo_proj = self.yolo_proj(yolo_feat)  # [B, fusion_dim]
        vit_proj = self.vit_proj(vit_feat)     # [B, fusion_dim]
        flow_proj = self.flow_proj(flow_feat)  # [B, fusion_dim]
        
        # Stack modalities
        stacked = torch.stack([yolo_proj, vit_proj, flow_proj], dim=1)  # [B, 3, fusion_dim]
        
        # Attention-weighted fusion
        fused, attention_weights = self.attention_fusion(stacked)  # [B, fusion_dim], [B, 3]
        
        # Final fusion
        output = self.fusion_head(fused)  # [B, output_dim]
        
        return output, attention_weights
    
    def get_modality_contributions(
        self,
        yolo_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        flow_feat: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get the contribution of each modality for interpretability.
        
        Returns:
            Dictionary with modality names and their contribution percentages.
        """
        _, weights = self.forward(yolo_feat, vit_feat, flow_feat)
        
        # Average across batch
        avg_weights = weights.mean(dim=0)
        
        return {
            'yolo': float(avg_weights[0]),
            'vit': float(avg_weights[1]),
            'optical_flow': float(avg_weights[2])
        }


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for more sophisticated fusion.
    
    Allows each modality to attend to features from other modalities.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-modal attention.
        
        Args:
            query: [B, 1, embed_dim] - Query modality
            key: [B, N, embed_dim] - Key modalities
            value: [B, N, embed_dim] - Value modalities
            
        Returns:
            Attended features [B, embed_dim]
        """
        # Cross attention
        attn_out, _ = self.cross_attn(query, key, value)
        query = self.norm1(query + attn_out)
        
        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        
        return query.squeeze(1)


if __name__ == '__main__':
    # Test Multi-Modal Fusion
    print("Testing MultiModalFusion...")
    
    fusion = MultiModalFusion()
    
    # Create dummy features
    yolo_feat = torch.randn(4, 512)
    vit_feat = torch.randn(4, 512)
    flow_feat = torch.randn(4, 256)
    
    # Fuse
    fused, weights = fusion(yolo_feat, vit_feat, flow_feat)
    
    print(f"Fused features shape: {fused.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Weights: {weights[0].tolist()}")
    
    # Get contributions
    contributions = fusion.get_modality_contributions(yolo_feat, vit_feat, flow_feat)
    print(f"\nModality contributions:")
    for modality, contrib in contributions.items():
        print(f"  {modality}: {contrib:.2%}")
