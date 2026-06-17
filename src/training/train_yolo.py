"""
YOLO Training Script
Train YOLOv8 for fire detection
"""

from ultralytics import YOLO
import argparse
from pathlib import Path
import yaml


def train_yolo(
    data_config: str = 'configs/data.yaml',
    model: str = 'yolov8n.pt',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    project: str = 'models/yolo',
    name: str = 'fire_detector',
    device: str = '0'
):
    """
    Train YOLOv8 for fire detection.
    
    Args:
        data_config: Path to data.yaml file
        model: Model variant (yolov8n/s/m/l/x.pt)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        project: Project directory
        name: Run name
        device: Device to train on ('0' for GPU, 'cpu' for CPU)
    """
    print("🔥 Starting YOLO Training for Fire Detection")
    print(f"   Model: {model}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    
    # Load model
    yolo = YOLO(model)
    
    # Training configuration
    results = yolo.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=20,
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        # Validation
        val=True,
        save=True,
        save_period=10,
        
        # Output
        project=project,
        name=name,
        exist_ok=True,
        
        # Logging
        verbose=True,
        plots=True
    )
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    metrics = yolo.val(split='test')
    
    print(f"\n✅ Training Complete!")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Best model saved to: {project}/{name}/weights/best.pt")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 for fire detection')
    parser.add_argument('--data', type=str, default='configs/data.yaml', help='Data config path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model variant')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device')
    
    args = parser.parse_args()
    
    train_yolo(
        data_config=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device
    )
