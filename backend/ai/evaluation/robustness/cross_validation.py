from typing import List, Dict, Any
import numpy as np

class CrossValidationManager:
    """Generates folds and aggregates results. Training execution is left to the TrainingManager."""
    
    def __init__(self, num_folds: int = 5):
        self.num_folds = num_folds
        self.fold_results: List[Dict[str, float]] = []
        
    def generate_folds(self, dataset_size: int) -> List[Dict[str, List[int]]]:
        """Returns a list of dicts with 'train' and 'val' indices."""
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        fold_sizes = np.full(self.num_folds, dataset_size // self.num_folds, dtype=int)
        fold_sizes[:dataset_size % self.num_folds] += 1
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop].tolist()
            train_idx = np.concatenate([indices[:start], indices[stop:]]).tolist()
            folds.append({"train": train_idx, "val": val_idx})
            current = stop
        return folds
        
    def add_result(self, metrics: Dict[str, float]):
        self.fold_results.append(metrics)
        
    def aggregate_results(self) -> Dict[str, float]:
        """Returns mean and std of metrics across folds."""
        if not self.fold_results:
            return {}
            
        aggregated = {}
        for key in self.fold_results[0].keys():
            values = [res[key] for res in self.fold_results if key in res]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            
        return aggregated
