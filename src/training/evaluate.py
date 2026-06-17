"""
Model Evaluation Script
Evaluate trained model on test set with comprehensive analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional

# Local imports
from .metrics import MetricsTracker, compute_roc_curve, format_confusion_matrix
from .train import FireDetectionDataset


def load_checkpoint(checkpoint_path: str, device: str = 'auto') -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Dict with model, config, and other checkpoint data
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Import model
    try:
        from src.detection.fire_detector import FireDetectionSystem
    except ImportError:
        from ..detection.fire_detector import FireDetectionSystem
    
    # Create model and load weights
    model = FireDetectionSystem().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return {
        'model': model,
        'device': device,
        'config': checkpoint.get('config', {}),
        'epoch': checkpoint.get('epoch', -1),
        'train_metrics': checkpoint.get('metrics', {}),
        'normalization': checkpoint.get('normalization', {}),
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        threshold: Detection threshold
        
    Returns:
        Evaluation results dict
    """
    model.eval()
    metrics_tracker = MetricsTracker(threshold=threshold)
    
    all_confidences = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            frames = batch['frames'].to(device)
            labels = batch['labels'].to(device).float()
            
            model.reset_temporal_state()
            outputs = model(frames)
            
            confidences = outputs['confidence'].cpu()
            predictions = (confidences >= threshold).float()
            
            metrics_tracker.update(confidences, labels.cpu())
            
            all_confidences.extend(confidences.view(-1).tolist())
            all_labels.extend(labels.cpu().view(-1).tolist())
            all_predictions.extend(predictions.view(-1).tolist())
    
    # Compute metrics
    metrics = metrics_tracker.compute()
    
    # Compute ROC curve for threshold calibration
    import numpy as np
    roc_data = compute_roc_curve(
        np.array(all_confidences),
        np.array(all_labels)
    )
    
    return {
        'metrics': metrics.to_dict(),
        'confusion_matrix': metrics.confusion_matrix.tolist(),
        'roc_curve': roc_data,
        'threshold': threshold,
        'n_samples': len(all_labels),
        'n_fire': sum(all_labels),
        'n_non_fire': len(all_labels) - sum(all_labels),
    }


def analyze_errors(
    model: nn.Module,
    test_dataset: FireDetectionDataset,
    device: torch.device,
    threshold: float = 0.5,
    max_samples: int = 20
) -> Dict:
    """
    Analyze false positives and false negatives.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device
        threshold: Detection threshold
        max_samples: Max errors to report
        
    Returns:
        Dict with FP and FN analysis
    """
    model.eval()
    
    false_positives = []
    false_negatives = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            frames = sample['frames'].unsqueeze(0).to(device)
            label = sample['labels'].item()
            
            model.reset_temporal_state()
            outputs = model(frames)
            confidence = outputs['confidence'].item()
            prediction = 1 if confidence >= threshold else 0
            
            img_path = str(test_dataset.images[idx])
            
            if prediction == 1 and label == 0:
                # False positive
                false_positives.append({
                    'image': img_path,
                    'confidence': confidence,
                    'index': idx
                })
            elif prediction == 0 and label == 1:
                # False negative
                false_negatives.append({
                    'image': img_path,
                    'confidence': confidence,
                    'index': idx
                })
    
    # Sort by confidence
    false_positives.sort(key=lambda x: -x['confidence'])  # Highest confidence first
    false_negatives.sort(key=lambda x: x['confidence'])   # Lowest confidence first
    
    return {
        'false_positives': {
            'count': len(false_positives),
            'samples': false_positives[:max_samples]
        },
        'false_negatives': {
            'count': len(false_negatives),
            'samples': false_negatives[:max_samples]
        }
    }


def generate_report(
    eval_results: Dict,
    error_analysis: Dict,
    checkpoint_info: Dict,
    output_path: Path
) -> str:
    """
    Generate comprehensive evaluation report.
    """
    lines = [
        "=" * 60,
        "FIRE DETECTION MODEL EVALUATION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "MODEL INFORMATION",
        "-" * 40,
        f"Checkpoint Epoch: {checkpoint_info.get('epoch', 'N/A')}",
        f"Training Config: {json.dumps(checkpoint_info.get('config', {}), indent=2)}",
        "",
        "TEST SET STATISTICS",
        "-" * 40,
        f"Total Samples: {eval_results['n_samples']}",
        f"Fire Samples: {eval_results['n_fire']}",
        f"Non-Fire Samples: {eval_results['n_non_fire']}",
        f"Detection Threshold: {eval_results['threshold']}",
        "",
        "PERFORMANCE METRICS",
        "-" * 40,
        f"Accuracy:  {eval_results['metrics']['accuracy']:.4f}",
        f"Precision: {eval_results['metrics']['precision']:.4f}",
        f"Recall:    {eval_results['metrics']['recall']:.4f}",
        f"F1 Score:  {eval_results['metrics']['f1_score']:.4f}",
        f"ROC-AUC:   {eval_results['metrics'].get('roc_auc', 'N/A')}",
        "",
        "CONFUSION MATRIX",
        "-" * 40,
        f"True Negatives:  {eval_results['metrics']['true_negatives']}",
        f"False Positives: {eval_results['metrics']['false_positives']}",
        f"False Negatives: {eval_results['metrics']['false_negatives']}",
        f"True Positives:  {eval_results['metrics']['true_positives']}",
        "",
        "ERROR ANALYSIS",
        "-" * 40,
        f"Total False Positives: {error_analysis['false_positives']['count']}",
        f"Total False Negatives: {error_analysis['false_negatives']['count']}",
        "",
        "THRESHOLD CALIBRATION (FROM ROC CURVE)",
        "-" * 40,
        f"Optimal Threshold: {eval_results['roc_curve']['optimal_threshold']:.4f}",
        f"At Optimal - TPR: {eval_results['roc_curve']['optimal_tpr']:.4f}, "
        f"FPR: {eval_results['roc_curve']['optimal_fpr']:.4f}",
        "",
    ]
    
    if error_analysis['false_positives']['samples']:
        lines.append("TOP FALSE POSITIVES (High confidence non-fire classified as fire)")
        lines.append("-" * 40)
        for fp in error_analysis['false_positives']['samples'][:5]:
            lines.append(f"  {Path(fp['image']).name}: {fp['confidence']:.4f}")
        lines.append("")
    
    if error_analysis['false_negatives']['samples']:
        lines.append("TOP FALSE NEGATIVES (Low confidence fire classified as non-fire)")
        lines.append("-" * 40)
        for fn in error_analysis['false_negatives']['samples'][:5]:
            lines.append(f"  {Path(fn['image']).name}: {fn['confidence']:.4f}")
        lines.append("")
    
    lines.extend([
        "=" * 60,
        "RECOMMENDATIONS",
        "=" * 60,
    ])
    
    # Generate recommendations
    metrics = eval_results['metrics']
    if metrics['precision'] < 0.8:
        lines.append("⚠️ Low precision - Model has many false positives")
        lines.append("   Consider: Increasing detection threshold, more negative training samples")
    
    if metrics['recall'] < 0.8:
        lines.append("⚠️ Low recall - Model misses fire detections")
        lines.append("   Consider: Decreasing detection threshold, more positive training samples")
    
    if metrics['f1_score'] < 0.7:
        lines.append("⚠️ Low F1 score - Overall performance needs improvement")
        lines.append("   Consider: More training data, data augmentation, longer training")
    
    optimal = eval_results['roc_curve']['optimal_threshold']
    current = eval_results['threshold']
    if abs(optimal - current) > 0.1:
        lines.append(f"💡 Consider using optimal threshold {optimal:.2f} instead of {current:.2f}")
    
    report = '\n'.join(lines)
    
    # Save report
    with open(output_path / 'evaluation_report.txt', 'w') as f:
        f.write(report)
    
    # Save detailed results as JSON
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump({
            'eval_results': eval_results,
            'error_analysis': error_analysis,
            'checkpoint_info': {k: v for k, v in checkpoint_info.items() if k != 'model'}
        }, f, indent=2)
    
    return report


def evaluate(
    checkpoint_path: str,
    data_dir: str = 'data/splits',
    output_dir: str = 'models/checkpoints',
    threshold: float = 0.5,
    device: str = 'auto',
    batch_size: int = 16
):
    """
    Run full evaluation pipeline.
    """
    print("🔍 Fire Detection Model Evaluation")
    print("=" * 50)
    
    # Load checkpoint
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    checkpoint_info = load_checkpoint(checkpoint_path, device)
    model = checkpoint_info['model']
    device = checkpoint_info['device']
    
    print(f"   Device: {device}")
    print(f"   Trained for {checkpoint_info['epoch'] + 1} epochs")
    
    # Load test dataset
    test_dir = f'{data_dir}/test'
    print(f"\n📂 Loading test data: {test_dir}")
    
    test_dataset = FireDetectionDataset(test_dir, sequence_length=3, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    print(f"\n⚙️ Evaluating on {len(test_dataset)} samples...")
    eval_results = evaluate_model(model, test_loader, device, threshold)
    
    # Error analysis
    print("\n🔬 Analyzing errors...")
    error_analysis = analyze_errors(model, test_dataset, device, threshold)
    
    # Generate report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n📝 Generating report...")
    report = generate_report(eval_results, error_analysis, checkpoint_info, output_path)
    
    print("\n" + report)
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate fire detection model')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/splits',
                       help='Path to data splits directory')
    parser.add_argument('--output', type=str, default='models/checkpoints',
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cuda/cpu)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        output_dir=args.output,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch
    )
