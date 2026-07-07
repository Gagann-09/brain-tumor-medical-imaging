from pathlib import Path
from typing import Any

import numpy as np

from ai.config.profiles import DEVELOPMENT_PROFILE, DatasetProfile
from medical.domain import MRIImage, MRIStudy, SegmentationAnnotation

from .base import DatasetAdapter


class BraTSDataset(DatasetAdapter):
    """
    Adapter for the Brain Tumor Segmentation (BraTS) dataset.
    Resolves dataset path via dataset profile or explicitly passed directory.
    """

    def __init__(self, data_dir: str | None = None, profile: DatasetProfile | None = None):
        self.profile = profile or DEVELOPMENT_PROFILE

        # Resolve dataset location: explicit data_dir > profile.data_dir
        self.root_dir = data_dir if data_dir else self.profile.data_dir

        if not self.root_dir or not Path(self.root_dir).exists():
            raise ValueError(
                f"Validation Error: BraTS dataset path not found at '{self.root_dir}'.\n"
                f"Configured via profile '{self.profile.name}'.\n"
                "Please configure the dataset path correctly in the environment or configuration."
            )

        self.items = []
        self.load()

    def load(self) -> None:
        """Load BraTS dataset index."""
        root_path = Path(self.root_dir)
        patient_dirs = [d for d in root_path.iterdir() if d.is_dir()]
        for p_dir in patient_dirs:
            # Mock discovering files to populate `files` dict for validation
            # In reality, this would search for actual .nii.gz files inside p_dir
            mock_files = {
                mod: f"{p_dir.name}_{mod}.nii.gz" for mod in self.profile.expected_modalities
            }
            mock_paths = [str(p_dir / v) for v in mock_files.values()]

            self.items.append(
                {
                    "patient_id": p_dir.name,
                    "path": str(p_dir),
                    "files": mock_files,
                    "paths": mock_paths,
                }
            )

    def get_metadata(self) -> dict[str, Any]:
        """Expose profile metadata through the adapter."""
        return self.profile.model_dump()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        patient_id = item["patient_id"]

        # Mock volumes based on expected modalities
        volumes = {}
        for mod in self.profile.expected_modalities:
            if mod != "SEG":
                volumes[mod] = np.zeros((1, 1, 1), dtype=np.float32)

        primary_image = (
            MRIImage(volumes=volumes)
            if volumes
            else MRIImage(volumes={"T1": np.zeros((1, 1, 1), dtype=np.float32)})
        )

        annotations = []
        if "SEG" in self.profile.expected_modalities:
            mask = np.zeros((1, 1, 1), dtype=np.float32)
            annotations.append(SegmentationAnnotation(mask=mask, label_map={"tumor": 1}))

        study = MRIStudy(
            primary_image=primary_image,
            study_id=patient_id,
            patient_id=patient_id,
            annotations=annotations,
        )
        return {"study": study}

    def preprocess(self, item: dict[str, Any]) -> dict[str, Any]:
        return item
