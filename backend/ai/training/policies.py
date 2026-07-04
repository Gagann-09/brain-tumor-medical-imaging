from typing import Dict, Any

class AugmentationPolicy:
    """Adjusts augmentations based on underfitting/overfitting."""
    @staticmethod
    def adjust(is_overfitting: bool, is_underfitting: bool, current_strength: float) -> float:
        if is_overfitting:
            return min(1.0, current_strength + 0.1) # Increase augmentation
        elif is_underfitting:
            return max(0.0, current_strength - 0.1) # Decrease augmentation
        return current_strength

class RegularizationPolicy:
    """Adjusts weight decay and dropout based on generalization gap."""
    @staticmethod
    def adjust(generalization_gap: float, gap_threshold: float, current_weight_decay: float) -> float:
        if generalization_gap > gap_threshold:
            return current_weight_decay * 1.5 # Increase regularization
        return current_weight_decay

class EarlyStoppingPolicy:
    """Stops training when OverfittingMonitor triggers consistently."""
    @staticmethod
    def should_stop(is_overfitting: bool, is_underfitting: bool) -> bool:
        return is_overfitting # Stop if overfitting

class LearningRatePolicy:
    """Wraps LR schedulers with rules for stability."""
    @staticmethod
    def adjust_for_stability(current_lr: float, is_oscillating: bool, is_diverging: bool) -> float:
        if is_diverging:
            return current_lr * 0.1 # Drastically reduce LR
        elif is_oscillating:
            return current_lr * 0.5 # Reduce LR
        return current_lr
