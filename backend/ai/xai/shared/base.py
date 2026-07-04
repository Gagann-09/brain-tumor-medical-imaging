import abc
from typing import Dict, Any, Optional
import torch
import numpy as np

from ai.interfaces.features import FeatureProvider

class BaseExplainer(abc.ABC):
    """
    Abstract base class for all explainability methods.
    """
    
    def __init__(self, model: FeatureProvider, device: Optional[torch.device] = None):
        if not isinstance(model, FeatureProvider):
            # We don't strictly enforce it here because standard nn.Module might be passed
            # but ideally it should implement FeatureProvider for feature-based methods.
            pass
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abc.abstractmethod
    def generate_explanation(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generates an explanation heatmap/map for the given input.
        
        Args:
            input_tensor (torch.Tensor): The input image tensor.
            target_class (Optional[int]): The index of the target class to explain. 
                                          If None, the model's top prediction is used.
                                          
        Returns:
            np.ndarray: A NumPy array representing the explanation map.
        """
        pass
