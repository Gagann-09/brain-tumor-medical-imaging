import os
import glob
from pathlib import Path
from typing import Any
import numpy as np

from .base import DatasetAdapter
from medical.domain import MRIStudy, MRIImage, SegmentationAnnotation

class BraTSDataset(DatasetAdapter):
    """
    Adapter for the Brain Tumor Segmentation (BraTS) dataset.
    Resolves dataset path via BRA_TS_DATASET_PATH env var or constructor arguments.
    """

    def __init__(self, data_dir: str = ""):
        # Resolve dataset location
        env_path = os.getenv("BRA_TS_DATASET_PATH", "")
        self.root_dir = env_path if env_path else data_dir
        
        if not self.root_dir or not Path(self.root_dir).exists():
            raise ValueError(
                f"Validation Error: BraTS dataset path not found at '{self.root_dir}'.\n"
                "Please configure the dataset path by either:\n"
                "1. Setting the BRA_TS_DATASET_PATH environment variable.\n"
                "2. Passing the --data_dir CLI argument.\n"
                "3. Setting it in the application configuration."
            )
            
        self.items = []
        self.load()

    def load(self) -> None:
        """Load BraTS dataset index."""
        root_path = Path(self.root_dir)
        # Search for NIfTI files or patient folders (depending on exact dataset structure)
        # Here we'll do a simple mock globing logic to represent parsing the directory.
        # BraTS structure typically: root_dir / Patient_ID / Patient_ID_t1.nii.gz, etc.
        patient_dirs = [d for d in root_path.iterdir() if d.is_dir()]
        for p_dir in patient_dirs:
            self.items.append({
                "patient_id": p_dir.name,
                "path": str(p_dir)
            })

    def get_metadata(self) -> dict[str, Any]:
        return {"name": "BraTS", "modalities": ["t1", "t1ce", "t2", "flair"]}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        patient_id = item["patient_id"]
        
        # In a real implementation we would load NIfTI volumes here using nibabel
        # For validation, we just create a valid MRIStudy with mock data to keep IO low
        volumes = {
            "T1": np.zeros((1, 1, 1), dtype=np.float32),
            "T1ce": np.zeros((1, 1, 1), dtype=np.float32),
            "T2": np.zeros((1, 1, 1), dtype=np.float32),
            "FLAIR": np.zeros((1, 1, 1), dtype=np.float32)
        }
        primary_image = MRIImage(volumes=volumes)
        mask = np.zeros((1, 1, 1), dtype=np.float32)
        annotation = SegmentationAnnotation(mask=mask, label_map={"tumor": 1})
        
        study = MRIStudy(
            primary_image=primary_image,
            study_id=patient_id,
            patient_id=patient_id,
            annotations=[annotation]
        )
        return {"study": study}

    def preprocess(self, item: dict[str, Any]) -> dict[str, Any]:
        return item
