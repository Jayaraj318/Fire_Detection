"""
Full System Training Script
Train the complete YOLO-ViT hybrid fire detection system

Features:
- Focal Loss for class imbalance handling
- Learning rate warmup (5 epochs)
- Early stopping with configurable patience
- Comprehensive validation metrics (Precision, Recall, F1, ROC-AUC)
- Complete checkpoint saving with config and optimizer state
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional, Dict, Any

# Local imports
from .focal_loss import FocalLoss
from .metrics import MetricsTracker, format_confusion_matrix

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not installed. Logging will be local only.")


class FireDetectionDataset(Dataset):
    """
    Dataset for fire detection training.
    
    Expected structure:
    data/splits/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ...
    """
    
    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 3,
        img_size: int = 640,
        augment: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.augment = augment
        
        # Find all images
        self.images = sorted(list((self.root_dir / 'images').glob('*.jpg')))
        self.images.extend(sorted(list((self.root_dir / 'images').glob('*.png'))))
        
        print(f"Found {len(self.images)} images in {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def get_label(self, idx: int) -> int:
        """Get label for sample (1=fire, 0=non-fire)."""
        img_path = self.images[idx]
        label_path = self.root_dir / 'labels' / f'{img_path.stem}.txt'
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == 0:
                        return 1  # Fire detected
        return 0  # No fire
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Load label
        label_path = self.root_dir / 'labels' / f'{img_path.stem}.txt'
        has_fire = 0
        fire_type = 0
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    # Check if any line has fire class (0)
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5 and int(parts[0]) == 0:
                            has_fire = 1
                            break
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Create sequence (replicate for single images)
        # TODO: Replace with actual temporal sequence loading for video data
        sequence = img_tensor.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
        
        return {
            'frames': sequence,  # [T, 3, H, W]
            'labels': torch.tensor([has_fire]),
            'fire_types': torch.tensor(fire_type)
        }


def get_weighted_sampler(dataset: FireDetectionDataset) -> WeightedRandomSampler:
    """
    Create a weighted sampler for balanced training.
    
    Args:
        dataset: Training dataset
        
    Returns:
        WeightedRandomSampler for balanced sampling
    """
    labels = [dataset.get_label(i) for i in range(len(dataset))]
    fire_count = sum(labels)
    non_fire_count = len(labels) - fire_count
    
    if fire_count == 0 or non_fire_count == 0:
        print("⚠️ Warning: Single-class dataset detected, skipping weighted sampling")
        return None
    
    # Compute weights (inverse frequency)
    total = len(labels)
    fire_weight = total / (2 * fire_count)
    non_fire_weight = total / (2 * non_fire_count)
    
    weights = [fire_weight if label == 1 else non_fire_weight for label in labels]
    
    print(f"⚖️ Weighted Sampling:")
    print(f"   Fire: {fire_count} samples (weight: {fire_weight:.3f})")
    print(f"   Non-fire: {non_fire_count} samples (weight: {non_fire_weight:.3f})")
    
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    metrics: Dict,
    config: Dict,
    path: Path
):
    """
    Save comprehensive checkpoint with all training state.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        # Normalization stats (for inference)
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    torch.save(checkpoint, path)


def train_model(
    data_dir: str = 'data/splits',
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.0001,
    device: str = 'auto',
    checkpoint_dir: str = 'models/checkpoints',
    use_wandb: bool = False,
    patience: int = 15,
    warmup_epochs: int = 5,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    use_weighted_sampling: bool = True
):
    """
    Train the complete fire detection system.
    
    Args:
        data_dir: Path to data splits
        epochs: Number of training epochs
        batch_size: Batch size (recommend 16-32 for stable BatchNorm)
        learning_rate: Initial learning rate
        device: 'auto', 'cuda', or 'cpu'
        checkpoint_dir: Where to save checkpoints
        use_wandb: Enable W&B logging
        patience: Early stopping patience
        warmup_epochs: Number of warmup epochs for LR scheduler
        use_focal_loss: Use Focal Loss instead of BCE
        focal_alpha: Focal Loss alpha parameter
        focal_gamma: Focal Loss gamma parameter
        use_weighted_sampling: Enable weighted sampling for class balance
    """
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Store config for checkpointing
    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'patience': patience,
        'warmup_epochs': warmup_epochs,
        'use_focal_loss': use_focal_loss,
        'focal_alpha': focal_alpha,
        'focal_gamma': focal_gamma,
        'use_weighted_sampling': use_weighted_sampling,
        'device': str(device),
    }
    
    print(f"🔥 Starting Fire Detection System Training")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Warmup Epochs: {warmup_epochs}")
    print(f"   Early Stopping Patience: {patience}")
    print(f"   Loss Function: {'Focal Loss' if use_focal_loss else 'BCE Loss'}")
    if use_focal_loss:
        print(f"   Focal Alpha: {focal_alpha}, Gamma: {focal_gamma}")
    
    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project='fire-detection-full', name='yolo-vit-hybrid-v2', config=config)
    
    # Import model (use absolute import for module execution)
    try:
        from src.detection.fire_detector import FireDetectionSystem
    except ImportError:
        from ..detection.fire_detector import FireDetectionSystem
    
    # Create model
    model = FireDetectionSystem().to(device)
    
    # Create datasets
    train_dataset = FireDetectionDataset(f'{data_dir}/train', sequence_length=3)
    val_dataset = FireDetectionDataset(f'{data_dir}/val', sequence_length=3)
    
    # Create sampler for balanced training
    train_sampler = None
    shuffle = True
    if use_weighted_sampling:
        train_sampler = get_weighted_sampler(train_dataset)
        if train_sampler is not None:
            shuffle = False  # Cannot use both sampler and shuffle
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print("✅ Using Focal Loss for class imbalance handling")
    else:
        criterion = nn.BCELoss()
    
    fire_type_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0001
    )
    
    # Learning rate scheduler with warmup
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[warmup_epochs]
        )
        print(f"✅ Using LR warmup for {warmup_epochs} epochs")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training state
    best_val_loss = float('inf')
    best_f1 = 0.0
    patience_counter = 0
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    training_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_metrics = MetricsTracker(threshold=0.5)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            frames = batch['frames'].to(device)  # [B, T, 3, H, W]
            labels = batch['labels'].to(device).float()  # [B, 1]
            fire_types = batch['fire_types'].to(device)  # [B]
            
            # Reset temporal state for each batch
            model.reset_temporal_state()
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(frames)
            
            # Detection loss
            loss = criterion(outputs['confidence'], labels)
            
            # Fire type loss (only for fire samples)
            if outputs['fire_type_probs'] is not None:
                fire_mask = labels.squeeze() == 1
                if fire_mask.sum() > 0:
                    type_loss = fire_type_criterion(
                        outputs['fire_type_probs'][fire_mask],
                        fire_types[fire_mask]
                    )
                    loss = loss + 0.3 * type_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_metrics.update(outputs['confidence'], labels)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        val_loss, val_metrics_result = validate(model, val_loader, criterion, device)
        
        # Compute training metrics
        train_metrics_result = train_metrics.compute()
        
        # Logging
        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"   Val Metrics - P: {val_metrics_result.precision:.4f}, "
              f"R: {val_metrics_result.recall:.4f}, F1: {val_metrics_result.f1_score:.4f}")
        if val_metrics_result.roc_auc:
            print(f"   ROC-AUC: {val_metrics_result.roc_auc:.4f}")
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # Store history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'precision': val_metrics_result.precision,
            'recall': val_metrics_result.recall,
            'f1_score': val_metrics_result.f1_score,
            'roc_auc': val_metrics_result.roc_auc,
            'learning_rate': current_lr,
        }
        training_history.append(epoch_data)
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'precision': val_metrics_result.precision,
                'recall': val_metrics_result.recall,
                'f1_score': val_metrics_result.f1_score,
                'roc_auc': val_metrics_result.roc_auc or 0,
                'learning_rate': current_lr
            })
        
        # Save best model (by val_loss or F1)
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
        
        if val_metrics_result.f1_score > best_f1:
            best_f1 = val_metrics_result.f1_score
            improved = True
        
        if improved:
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss, val_loss,
                val_metrics_result.to_dict(), config,
                checkpoint_path / 'best_model.pth'
            )
            print(f"   ✅ Best model saved (val_loss: {val_loss:.4f}, F1: {val_metrics_result.f1_score:.4f})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter}/{patience} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
        
        # Save periodic checkpoint with attention/feature maps
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss, val_loss,
                val_metrics_result.to_dict(), config,
                checkpoint_path / f'checkpoint_epoch_{epoch+1}.pth'
            )
            print(f"   💾 Periodic checkpoint saved")
        
        scheduler.step()
    
    # Save training history
    with open(checkpoint_path / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\n🎉 Training Complete!")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Best F1 Score: {best_f1:.4f}")
    print(f"   Model saved to: {checkpoint_path}/best_model.pth")
    print(f"   Training history saved to: {checkpoint_path}/training_history.json")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model


def validate(model, val_loader, criterion, device):
    """
    Run validation with comprehensive metrics.
    
    Returns:
        Tuple of (val_loss, MetricsResult)
    """
    model.eval()
    val_loss = 0.0
    metrics_tracker = MetricsTracker(threshold=0.5)
    
    with torch.no_grad():
        for batch in val_loader:
            frames = batch['frames'].to(device)
            labels = batch['labels'].to(device).float()
            
            model.reset_temporal_state()
            outputs = model(frames)
            loss = criterion(outputs['confidence'], labels)
            val_loss += loss.item()
            
            metrics_tracker.update(outputs['confidence'], labels)
    
    metrics_result = metrics_tracker.compute()
    
    return val_loss / len(val_loader), metrics_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train fire detection system')
    parser.add_argument('--data', type=str, default='data/splits', help='Data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size (recommend 16-32)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--warmup', type=int, default=5, help='LR warmup epochs')
    parser.add_argument('--no-focal-loss', action='store_true', help='Use BCE instead of Focal Loss')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal Loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal Loss gamma')
    parser.add_argument('--no-weighted-sampling', action='store_true', help='Disable weighted sampling')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        device=args.device,
        use_wandb=args.wandb,
        patience=args.patience,
        warmup_epochs=args.warmup,
        use_focal_loss=not args.no_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_weighted_sampling=not args.no_weighted_sampling
    )
