import csv
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from medical.domain import ClassificationAnnotation, MRIImage, MRIStudy

from .base import BaseDatasetAdapter


class KaggleDatasetAdapter(BaseDatasetAdapter):
    """
    Adapter for Kaggle supplementary classification datasets.
    Assumes a CSV file maps study_id to a class label, and folders contain 
    either 2D slices (e.g. .npy or .png mapped to numpy arrays) or 3D NIfTIs.
    """

    def __init__(self, root_dir: str | Path, csv_path: str | Path):
        super().__init__(root_dir)
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self._load_labels()

    def _load_labels(self):
        self.labels = {}
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            # Assuming columns 'study_id' and 'label'
            for row in reader:
                study_id = row.get("study_id", row.get("BraTS21ID"))  # Example fallback for RSNA Kaggle
                label = row.get("label", row.get("MGMT_value"))
                if study_id and label is not None:
                    # Pad to standard BraTS format if it's the RSNA dataset (e.g., "00001")
                    if study_id.isdigit():
                        study_id = study_id.zfill(5)
                    self.labels[study_id] = str(label)

    def load_studies(self) -> Iterator[MRIStudy]:
        for patient_dir in self.root_dir.iterdir():
            if patient_dir.is_dir():
                yield self.load_study(patient_dir.name)

    def load_study(self, study_id: str) -> MRIStudy:
        patient_dir = self.root_dir / study_id
        if not patient_dir.exists():
            raise FileNotFoundError(f"Study directory not found: {patient_dir}")

        # Mocking the load logic for a Kaggle dataset where images might be grouped
        # by modality folders (e.g., RSNA MGMT Kaggle competition)
        volumes = {}
        # In reality, we'd use pydicom to load DCMs or PIL for PNGs, then stack them.
        # For framework-agnostic skeleton, we create a dummy volume.
        # This implementation requires specific image loaders depending on Kaggle competition format.

        # Example dummy extraction
        volumes["UNKNOWN"] = np.zeros((240, 240, 155), dtype=np.float32)

        image = MRIImage(volumes=volumes)

        annotations = []
        if study_id in self.labels:
            class_name = self.labels[study_id]
            annotations.append(ClassificationAnnotation(class_name=class_name))

        return MRIStudy(
            primary_image=image,
            study_id=study_id,
            patient_id=study_id,
            annotations=annotations
        )
