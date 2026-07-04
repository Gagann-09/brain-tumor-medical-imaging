from typing import List, Dict, Any
import numpy as np

class LearningCurveAnalyzer:
    """Analyzes learning curves for plateaus, divergence, oscillation, etc."""
    
    @staticmethod
    def analyze_curve(train_history: List[float], val_history: List[float]) -> Dict[str, bool]:
        """
        Analyzes the training and validation history (e.g. losses).
        Returns flags for various states.
        """
        states = {
            "plateau": False,
            "divergence": False,
            "oscillation": False,
            "exploding_gradients": False,
            "vanishing_gradients": False
        }
        
        if not train_history or not val_history:
            return states
            
        train_arr = np.array(train_history)
        val_arr = np.array(val_history)
        
        # Exploding gradients: check for NaNs, Infs, or massive sudden spikes
        if np.isnan(train_arr).any() or np.isinf(train_arr).any() or np.max(train_arr) > 1e4:
            states["exploding_gradients"] = True
            
        # Vanishing gradients: loss barely moves over many epochs, but not near 0
        if len(train_arr) > 5:
            recent_train = train_arr[-5:]
            if np.std(recent_train) < 1e-5 and np.mean(recent_train) > 0.5:
                states["vanishing_gradients"] = True
                
        # Plateau: Validation loss hasn't improved much recently
        if len(val_arr) > 5:
            recent_val = val_arr[-5:]
            if np.std(recent_val) < 1e-4:
                states["plateau"] = True
                
        # Divergence: Training loss goes down, validation loss goes up
        if len(train_arr) > 3 and len(val_arr) > 3:
            train_trend = np.polyfit(np.arange(3), train_arr[-3:], 1)[0]
            val_trend = np.polyfit(np.arange(3), val_arr[-3:], 1)[0]
            if train_trend < 0 and val_trend > 0:
                states["divergence"] = True
                
        # Oscillation: Frequent changes in direction
        if len(val_arr) > 4:
            diffs = np.diff(val_arr)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes >= 3:
                states["oscillation"] = True
                
        return states

class ClassDistributionAnalyzer:
    @staticmethod
    def analyze(class_counts: Dict[str, int]) -> Dict[str, float]:
        """Returns the percentage distribution of each class."""
        total = sum(class_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in class_counts.items()}
