import time
from typing import Dict, Any, List
import torch

from ai.classification.metrics.metrics import compute_classification_metrics
from ai.classification.inference.inference import ClassificationInference
from ai.classification.datasets.sample import ClassificationSample

class ClassificationBenchmark:
    """
    Evaluates a classification model over a dataset and compiles standard metrics:
    Accuracy, Precision, Recall, F1, ROC AUC, PR AUC, Confusion Matrix, and latency.
    """
    def __init__(self, inference_engine: ClassificationInference):
        self.inference_engine = inference_engine
        
    def evaluate(self, dataset: List[ClassificationSample], num_classes: int = 2) -> Dict[str, Any]:
        all_logits = []
        all_targets = []
        latencies = []
        
        for sample in dataset:
            start_time = time.time()
            result = self.inference_engine(sample)
            latencies.append(time.time() - start_time)
            
            all_logits.append(result["logits"])
            # Assuming 'target' is stored in labels dictionary
            target = sample.labels.get("target", 0)
            all_targets.append(target)
            
        logits_tensor = torch.stack(all_logits)
        targets_tensor = torch.tensor(all_targets)
        
        metrics = compute_classification_metrics(logits_tensor, targets_tensor, num_classes)
        metrics["avg_inference_latency_sec"] = sum(latencies) / len(latencies) if latencies else 0.0
        
        return metrics
