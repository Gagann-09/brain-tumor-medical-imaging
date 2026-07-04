from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch

@dataclass
class ClassificationSample:
    """
    Domain object representing a single sample for classification.
    Can contain the raw volume, pre-extracted ROI tensor, metadata, and labels.
    """
    sample_id: str
    
    # Input Data
    image_tensor: torch.Tensor
    mask_tensor: Optional[torch.Tensor] = None
    
    # Classification Labels
    labels: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
