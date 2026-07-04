from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class DatasetStatistics:
    """Statistics about the training and validation datasets."""
    total_samples: int = 0
    modality_distribution: Dict[str, int] = field(default_factory=dict)
    missing_modality_statistics: Dict[str, int] = field(default_factory=dict)
    tumor_class_distribution: Dict[str, int] = field(default_factory=dict)
    image_dimension_distribution: Dict[str, int] = field(default_factory=dict)
    voxel_spacing_distribution: Dict[str, int] = field(default_factory=dict)
