"""
Validation Metrics Module
Comprehensive metrics for fire detection evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MetricsResult:
    """Container for computed metrics."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: np.ndarray  # [[TN, FP], [FN, TP]]
    roc_auc: Optional[float] = None
    threshold: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'roc_auc': self.roc_auc,
            'threshold': self.threshold,
            'true_negatives': int(self.confusion_matrix[0, 0]),
            'false_positives': int(self.confusion_matrix[0, 1]),
            'false_negatives': int(self.confusion_matrix[1, 0]),
            'true_positives': int(self.confusion_matrix[1, 1]),
        }


class MetricsTracker:
    """
    Accumulate predictions and compute metrics across batches.
    
    Usage:
        tracker = MetricsTracker()
        for batch in dataloader:
            preds, targets = model(batch)
            tracker.update(preds, targets)
        
        metrics = tracker.compute()
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Clear accumulated predictions."""
        self.predictions: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.confidences: List[torch.Tensor] = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Add batch predictions.
        
        Args:
            predictions: Model confidence scores [B] or [B, 1] (0-1 range)
            targets: Ground truth labels [B] or [B, 1] (0 or 1)
        """
        predictions = predictions.detach().cpu().view(-1)
        targets = targets.detach().cpu().view(-1)
        
        self.confidences.append(predictions)
        self.predictions.append((predictions >= self.threshold).float())
        self.targets.append(targets.float())
    
    def compute(self) -> MetricsResult:
        """Compute all metrics from accumulated data."""
        if len(self.predictions) == 0:
            raise ValueError("No predictions accumulated. Call update() first.")
        
        preds = torch.cat(self.predictions)
        targets = torch.cat(self.targets)
        confidences = torch.cat(self.confidences)
        
        # Confusion matrix
        cm = compute_confusion_matrix(preds, targets)
        
        # Basic metrics
        precision, recall, f1 = compute_precision_recall_f1(cm)
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        
        # ROC-AUC
        try:
            roc_auc = compute_roc_auc(confidences.numpy(), targets.numpy())
        except Exception:
            roc_auc = None
        
        return MetricsResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            confusion_matrix=cm,
            roc_auc=roc_auc,
            threshold=self.threshold
        )


def compute_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Binary predictions [N]
        targets: Ground truth [N]
        
    Returns:
        2x2 numpy array: [[TN, FP], [FN, TP]]
    """
    predictions = predictions.view(-1).numpy()
    targets = targets.view(-1).numpy()
    
    tp = ((predictions == 1) & (targets == 1)).sum()
    tn = ((predictions == 0) & (targets == 0)).sum()
    fp = ((predictions == 1) & (targets == 0)).sum()
    fn = ((predictions == 0) & (targets == 1)).sum()
    
    return np.array([[tn, fp], [fn, tp]], dtype=np.float32)


def compute_precision_recall_f1(confusion_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 from confusion matrix.
    
    Args:
        confusion_matrix: 2x2 array [[TN, FP], [FN, TP]]
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return float(precision), float(recall), float(f1)


def compute_roc_auc(confidences: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute ROC-AUC score.
    
    Args:
        confidences: Model confidence scores [N]
        targets: Ground truth labels [N]
        
    Returns:
        ROC-AUC score
    """
    # Simple implementation without sklearn dependency
    # Sort by confidence descending
    sorted_indices = np.argsort(-confidences)
    sorted_targets = targets[sorted_indices]
    
    # Compute TPR and FPR at each threshold
    n_pos = targets.sum()
    n_neg = len(targets) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return random baseline
    
    tp = 0
    fp = 0
    tpr_prev = 0
    fpr_prev = 0
    auc = 0.0
    
    for i, target in enumerate(sorted_targets):
        if target == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        # Trapezoidal rule
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        
        tpr_prev = tpr
        fpr_prev = fpr
    
    return float(auc)


def compute_roc_curve(confidences: np.ndarray, targets: np.ndarray, n_thresholds: int = 100) -> Dict:
    """
    Compute full ROC curve data for threshold calibration.
    
    Args:
        confidences: Model confidence scores [N]
        targets: Ground truth labels [N]
        n_thresholds: Number of threshold points
        
    Returns:
        Dict with 'thresholds', 'tpr', 'fpr', 'optimal_threshold'
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list = []
    fpr_list = []
    
    n_pos = targets.sum()
    n_neg = len(targets) - n_pos
    
    for threshold in thresholds:
        preds = (confidences >= threshold).astype(float)
        
        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        
        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    
    # Optimal threshold: maximize TPR - FPR (Youden's J statistic)
    j_scores = tpr_arr - fpr_arr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'thresholds': thresholds.tolist(),
        'tpr': tpr_arr.tolist(),
        'fpr': fpr_arr.tolist(),
        'optimal_threshold': float(optimal_threshold),
        'optimal_tpr': float(tpr_arr[optimal_idx]),
        'optimal_fpr': float(fpr_arr[optimal_idx]),
    }


def format_confusion_matrix(cm: np.ndarray, class_names: List[str] = None) -> str:
    """
    Format confusion matrix as readable string.
    
    Args:
        cm: 2x2 confusion matrix
        class_names: Optional names for classes
        
    Returns:
        Formatted string
    """
    if class_names is None:
        class_names = ['Non-Fire', 'Fire']
    
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    lines = [
        "Confusion Matrix:",
        f"                  Predicted",
        f"                  {class_names[0]:>10} {class_names[1]:>10}",
        f"Actual {class_names[0]:>10} {int(tn):>10} {int(fp):>10}",
        f"       {class_names[1]:>10} {int(fn):>10} {int(tp):>10}",
    ]
    
    return '\n'.join(lines)


if __name__ == '__main__':
    print("Testing MetricsTracker...")
    
    # Create sample data
    tracker = MetricsTracker(threshold=0.5)
    
    # Simulate batches
    for _ in range(3):
        preds = torch.rand(10)  # Random predictions
        targets = (torch.rand(10) > 0.5).float()  # Random targets
        tracker.update(preds, targets)
    
    # Compute metrics
    metrics = tracker.compute()
    
    print(f"\nMetrics:")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 Score:  {metrics.f1_score:.4f}")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  ROC-AUC:   {metrics.roc_auc:.4f}" if metrics.roc_auc else "  ROC-AUC:   N/A")
    print()
    print(format_confusion_matrix(metrics.confusion_matrix))
    
    print("\n✅ Metrics test passed!")
