"""
Vision Transformer Branch
Extract global context features using ViT for fire detection
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTImageProcessor
from typing import Optional
import torch.nn.functional as F


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer for extracting global context features.
    
    Uses pretrained ViT-base and extracts the [CLS] token for global representation.
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        img_size: int = 640,
        embed_dim: int = 768,
        output_dim: int = 512,
        freeze_early_layers: int = 50
    ):
        """
        Initialize the ViT feature extractor.
        
        Args:
            model_name: HuggingFace model name for pretrained ViT.
            img_size: Input image size.
            embed_dim: ViT embedding dimension.
            output_dim: Output projection dimension for fusion.
            freeze_early_layers: Number of early parameters to freeze.
        """
        super().__init__()
        
        self.img_size = img_size
        self.vit_input_size = 224  # ViT expects 224x224
        
        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained(model_name)
        
        # Freeze early layers for transfer learning
        params = list(self.vit.parameters())
        for param in params[:freeze_early_layers]:
            param.requires_grad = False
        
        # Projection layer to match fusion dimensions
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global context features from image.
        
        Args:
            x: Input tensor [B, 3, H, W] normalized to [0, 1]
            
        Returns:
            Global context features [B, output_dim]
        """
        # Resize to ViT input size if necessary
        if x.shape[-1] != self.vit_input_size or x.shape[-2] != self.vit_input_size:
            x = F.interpolate(
                x, 
                size=(self.vit_input_size, self.vit_input_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Get ViT outputs
        outputs = self.vit(pixel_values=x)
        
        # Extract [CLS] token (global representation)
        cls_token = outputs.last_hidden_state[:, 0]  # [B, embed_dim]
        
        # Project to fusion dimension
        features = self.projection(cls_token)  # [B, output_dim]
        
        return features
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps from ViT for visualization.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Attention maps from last layer [B, num_heads, num_patches, num_patches]
        """
        if x.shape[-1] != self.vit_input_size:
            x = F.interpolate(x, size=(self.vit_input_size, self.vit_input_size))
        
        outputs = self.vit(pixel_values=x, output_attentions=True)
        
        # Return attention from last layer
        return outputs.attentions[-1]


class LightweightViT(nn.Module):
    """
    Lightweight ViT alternative for edge deployment.
    Uses a simpler transformer architecture with fewer parameters.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        output_dim: int = 512
    ):
        super().__init__()
        
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, 3, H, W]
            
        Returns:
            [B, output_dim]
        """
        B = x.shape[0]
        
        # Resize if necessary
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Extract CLS token
        x = self.norm(x[:, 0])
        x = self.projection(x)
        
        return x


if __name__ == '__main__':
    # Test the ViT extractor
    print("Testing ViTFeatureExtractor...")
    vit = ViTFeatureExtractor()
    
    dummy_input = torch.randn(2, 3, 640, 640)
    features = vit(dummy_input)
    print(f"ViT features shape: {features.shape}")
    
    # Test lightweight ViT
    print("\nTesting LightweightViT...")
    lightweight_vit = LightweightViT()
    features = lightweight_vit(dummy_input)
    print(f"Lightweight ViT features shape: {features.shape}")
