import random
from abc import ABC, abstractmethod

from medical.domain import MRIStudy
from medical.exceptions import MedicalImagingError


class SplitStrategy(ABC):
    """Abstract strategy for splitting a dataset into train/val/test."""

    @abstractmethod
    def split(self, studies: list[MRIStudy], train_ratio: float, val_ratio: float) -> tuple[list[MRIStudy], list[MRIStudy], list[MRIStudy]]:
        """
        Split a list of studies into train, validation, and test sets.
        """
        pass

class PatientSplitStrategy(SplitStrategy):
    """
    Splits dataset at the patient level to ensure no data leakage.
    All studies belonging to the same patient will be in the same split.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def split(self, studies: list[MRIStudy], train_ratio: float, val_ratio: float) -> tuple[list[MRIStudy], list[MRIStudy], list[MRIStudy]]:
        if train_ratio + val_ratio > 1.0:
            raise ValueError("train_ratio + val_ratio cannot exceed 1.0")

        # Group studies by patient ID
        patient_groups: dict[str, list[MRIStudy]] = {}
        for study in studies:
            if not study.patient_id:
                raise MedicalImagingError(f"Study {study.study_id} missing patient ID for PatientSplitStrategy")
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
        val_ids = set(patient_ids[n_train:n_train + n_val])
        test_ids = set(patient_ids[n_train + n_val:])

        train_set = []
        val_set = []
        test_set = []

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

    def create_splits(self, studies: list[MRIStudy], train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[list[MRIStudy], list[MRIStudy], list[MRIStudy]]:
        return self.strategy.split(studies, train_ratio, val_ratio)
