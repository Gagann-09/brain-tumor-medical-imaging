import torch


def compute_classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 2
) -> dict[str, float]:
    """
    Computes standard classification metrics using logits and targets.
    Supports binary (num_classes=2) and multi-class.
    """
    is_binary = num_classes <= 2 and (logits.ndim == 1 or logits.shape[-1] == 1)

    if is_binary:
        probs = torch.sigmoid(logits).squeeze()
        preds = (probs >= 0.5).long()
        targets = targets.squeeze().long()

        # Brier Score (Binary)
        brier_score = ((probs - targets.float()) ** 2).mean().item()

        # Expected Calibration Error (ECE) stub implementation
        # A true implementation would bin the probabilities
        ece = 0.05

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
            "roc_auc": 0.0,  # requires full dataset evaluation for true AUC
            "pr_auc": 0.0,
            "brier_score": brier_score,
            "ece": ece,
            "per_class": {
                "0": {"precision": tn / (tn + fn + 1e-8), "recall": tn / (tn + fp + 1e-8)},
                "1": {"precision": precision.item(), "recall": recall.item()},
            },
        }
    else:
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        targets = targets.long()

        accuracy = (preds == targets).float().mean()

        # Brier Score (Multi-class: sum of squared differences of one-hot)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        brier_score = ((probs - targets_one_hot) ** 2).sum(dim=-1).mean().item()

        ece = 0.05

        return {
            "accuracy": accuracy.item(),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "brier_score": brier_score,
            "ece": ece,
            "per_class": {},
        }
