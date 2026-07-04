import torch
from typing import Dict, Any

def compute_classification_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 2) -> Dict[str, float]:
    """
    Computes standard classification metrics using logits and targets.
    Supports binary (num_classes=2) and multi-class.
    """
    is_binary = num_classes <= 2 and (logits.ndim == 1 or logits.shape[-1] == 1)
    
    if is_binary:
        probs = torch.sigmoid(logits).squeeze()
        preds = (probs >= 0.5).long()
        targets = targets.squeeze().long()
        
        # Simple computation for binary (batch-level)
        tp = ((preds == 1) & (targets == 1)).sum().float()
        tn = ((preds == 0) & (targets == 0)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # We would typically use sklearn.metrics or torchmetrics for AUC
        # This is a stub for the benchmark registry
        return {
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "f1_score": f1.item(),
            "roc_auc": 0.0, # requires full dataset evaluation for true AUC
            "pr_auc": 0.0,
        }
    else:
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        targets = targets.long()
        
        accuracy = (preds == targets).float().mean()
        
        return {
            "accuracy": accuracy.item(),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
        }
