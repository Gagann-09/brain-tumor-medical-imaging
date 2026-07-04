import numpy as np

from medical.domain import MRIStudy, SegmentationAnnotation


class DatasetHarmonizer:
    """Normalizes modalities, orientations, voxel spacing, and label schemas across datasets."""

    def __init__(self):
        # A mapping of common naming conventions to standard internal names
        self.modality_mapping = {
            "t1": "T1",
            "t1w": "T1",
            "t1-weighted": "T1",
            "t1ce": "T1ce",
            "t1c": "T1ce",
            "t1-gd": "T1ce",
            "t2": "T2",
            "t2w": "T2",
            "t2-weighted": "T2",
            "flair": "FLAIR",
            "t2-flair": "FLAIR"
        }

    def harmonize_modality_name(self, raw_name: str) -> str:
        """Standardize a modality string."""
        return self.modality_mapping.get(raw_name.lower(), raw_name)

    def harmonize_label_schema(self, annotation: SegmentationAnnotation, target_map: dict[int, str], value_mapping: dict[int, int]) -> SegmentationAnnotation:
        """
        Harmonize a segmentation mask's values to a target schema.
        
        Args:
            annotation: The input SegmentationAnnotation.
            target_map: The new label map (e.g., {1: 'Core', 2: 'Edema'}).
            value_mapping: How to map old values to new values (e.g., {4: 1, 1: 1, 2: 2}).
        """
        new_mask = np.zeros_like(annotation.mask)
        for old_val, new_val in value_mapping.items():
            new_mask[annotation.mask == old_val] = new_val

        return SegmentationAnnotation(
            mask=new_mask,
            label_map=target_map,
            metadata=annotation.metadata
        )

    def harmonize_study(self, study: MRIStudy) -> MRIStudy:
        """
        Apply harmonization in-place or return a new harmonized study.
        Here we standardize modality keys in the primary image.
        """
        image = study.primary_image
        new_volumes = {}
        for mod, vol in image.volumes.items():
            std_mod = self.harmonize_modality_name(mod)
            new_volumes[std_mod] = vol

        image.volumes = new_volumes
        return study
