from typing import Dict, Any, List, Set

class GeneralizationGapCalculator:
    """Calculates generalization gap between training and validation metrics."""
    
    @staticmethod
    def calculate_gap(train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> Dict[str, float]:
        """Returns the absolute difference between train and val metrics."""
        gaps = {}
        for k, v in train_metrics.items():
            if k in val_metrics:
                # If metric is loss, lower is better. If accuracy, higher is better.
                # Just return absolute gap. The analyzer can decide what it means.
                gaps[k] = abs(v - val_metrics[k])
        return gaps

class DatasetLeakageDetector:
    """Detects data leakage between dataset splits."""
    
    @staticmethod
    def static_validation(train_ids: Set[str], val_ids: Set[str]) -> bool:
        """
        Validates leakage statically before training.
        Returns True if leakage is detected (intersection > 0).
        """
        intersection = train_ids.intersection(val_ids)
        if intersection:
            print(f"Warning: Data leakage detected statically. Overlapping IDs: {intersection}")
            return True
        return False
        
    @staticmethod
    def runtime_validation(train_batch_ids: List[str], val_ids_seen: Set[str]) -> bool:
        """
        Validates leakage during runtime (e.g., during cross-validation).
        Returns True if any id in train_batch_ids was already seen in val.
        """
        batch_set = set(train_batch_ids)
        if batch_set.intersection(val_ids_seen):
            print(f"Warning: Data leakage detected at runtime.")
            return True
        return False
