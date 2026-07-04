from typing import List

class OverfittingMonitor:
    """Monitors if the model is overfitting by comparing train/val metrics over time."""
    
    def __init__(self, patience: int = 3, threshold: float = 0.05):
        self.patience = patience
        self.threshold = threshold # Gap threshold
        self.overfitting_count = 0
        self.is_overfitting = False
        
    def update(self, train_loss: float, val_loss: float):
        # Simplistic logic: if val_loss is higher than train_loss by threshold
        if (val_loss - train_loss) > self.threshold:
            self.overfitting_count += 1
        else:
            self.overfitting_count = 0
            
        if self.overfitting_count >= self.patience:
            self.is_overfitting = True

class UnderfittingMonitor:
    """Monitors if the model is underfitting (both train/val loss plateau at high values)."""
    def __init__(self, high_loss_threshold: float = 0.5, patience: int = 3):
        self.high_loss_threshold = high_loss_threshold
        self.patience = patience
        self.underfitting_count = 0
        self.is_underfitting = False
        
    def update(self, train_loss: float, val_loss: float):
        if train_loss > self.high_loss_threshold and val_loss > self.high_loss_threshold:
            self.underfitting_count += 1
        else:
            self.underfitting_count = 0
            
        if self.underfitting_count >= self.patience:
            self.is_underfitting = True
