from dataclasses import dataclass, field


@dataclass
class DatasetStatistics:
    """Statistics about the training and validation datasets."""

    total_samples: int = 0
    modality_distribution: dict[str, int] = field(default_factory=dict)
    missing_modality_statistics: dict[str, int] = field(default_factory=dict)
    tumor_class_distribution: dict[str, int] = field(default_factory=dict)
    image_dimension_distribution: dict[str, int] = field(default_factory=dict)
    voxel_spacing_distribution: dict[str, int] = field(default_factory=dict)
