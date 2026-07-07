import json
import random
from abc import ABC, abstractmethod
from typing import Any

from medical.domain import MRIStudy
from medical.exceptions import MedicalImagingError


class SplitStrategy(ABC):
    """Abstract strategy for splitting a dataset into train/val/test."""

    @abstractmethod
    def split(
        self, studies: list[MRIStudy], train_ratio: float, val_ratio: float
    ) -> tuple[list[MRIStudy], list[MRIStudy], list[MRIStudy]]:
        pass


class ManifestSplitStrategy(SplitStrategy):
    """
    Splits dataset strictly based on a provided JSON manifest file containing lists of patient identifiers.  # noqa: E501
    """

    def __init__(self, manifest_path: str, fold: str | None = None):
        self.manifest_path = manifest_path
        self.fold = fold
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        with open(self.manifest_path) as f:
            return json.load(f)

    def split(
        self, studies: list[MRIStudy], train_ratio: float, val_ratio: float
    ) -> tuple[list[MRIStudy], list[MRIStudy], list[MRIStudy]]:
        splits = self.manifest.get("splits", {})
        if self.fold:
            splits = splits.get(self.fold, {})
            if not splits:
                raise MedicalImagingError(
                    f"Fold '{self.fold}' not found in manifest '{self.manifest_path}'."
                )

        train_ids = set(splits.get("train", []))
        val_ids = set(splits.get("val", []))
        test_ids = set(splits.get("test", []))

        # Ensure no overlap
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise MedicalImagingError("Split manifest contains overlapping patient IDs.")

        train_set, val_set, test_set = [], [], []

        assigned_ids = set()
        for study in studies:
            pid = study.patient_id
            if not pid:
                raise MedicalImagingError(
                    f"Study {study.study_id} missing patient ID for ManifestSplitStrategy"
                )

            if pid in assigned_ids:
                continue

            assigned_ids.add(pid)

            if pid in train_ids:
                train_set.append(study)
            elif pid in val_ids:
                val_set.append(study)
            elif pid in test_ids:
                test_set.append(study)
            else:
                raise MedicalImagingError(
                    f"Patient ID '{pid}' not found in any split in manifest '{self.manifest_path}'."
                )

        # Optionally check if all manifest IDs were found
        missing_ids = (train_ids | val_ids | test_ids) - assigned_ids
        if missing_ids:
            # Just logging a warning or throwing an error depending on strictness, but for this implementation:
            pass

        return train_set, val_set, test_set


class PatientSplitStrategy(SplitStrategy):
    """
    Splits dataset at the patient level to ensure no data leakage.
    All studies belonging to the same patient will be in the same split.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split(
        self, studies: list[MRIStudy], train_ratio: float, val_ratio: float
    ) -> tuple[list[MRIStudy], list[MRIStudy], list[MRIStudy]]:
        if train_ratio + val_ratio > 1.0:
            raise ValueError("train_ratio + val_ratio cannot exceed 1.0")

        patient_groups: dict[str, list[MRIStudy]] = {}
        for study in studies:
            if not study.patient_id:
                raise MedicalImagingError(
                    f"Study {study.study_id} missing patient ID for PatientSplitStrategy"
                )
            if study.patient_id not in patient_groups:
                patient_groups[study.patient_id] = []
            patient_groups[study.patient_id].append(study)

        patient_ids = list(patient_groups.keys())
        random.seed(self.seed)
        random.shuffle(patient_ids)

        n_patients = len(patient_ids)
        n_train = int(n_patients * train_ratio)
        n_val = int(n_patients * val_ratio)

        train_ids = set(patient_ids[:n_train])
        val_ids = set(patient_ids[n_train : n_train + n_val])
        test_ids = set(patient_ids[n_train + n_val :])

        train_set, val_set, test_set = [], [], []

        for pid in train_ids:
            train_set.extend(patient_groups[pid])
        for pid in val_ids:
            val_set.extend(patient_groups[pid])
        for pid in test_ids:
            test_set.extend(patient_groups[pid])

        return train_set, val_set, test_set


class DatasetSplitManager:
    """Manages the splitting of datasets using a configurable strategy."""

    def __init__(self, strategy: SplitStrategy):
        self.strategy = strategy

    def create_splits(
        self, studies: list[MRIStudy], train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> tuple[list[MRIStudy], list[MRIStudy], list[MRIStudy]]:
        return self.strategy.split(studies, train_ratio, val_ratio)
