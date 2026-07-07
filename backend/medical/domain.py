from typing import Any

import numpy as np

from .metadata.models import ImageMetadata, PatientMetadata, StudyMetadata


class MRIImage:
    """
    Unified domain model representing loaded MRI imaging data and metadata.
    Supports multiple MRI modalities (e.g., T1, T1ce, T2, FLAIR) within a single object.
    """

    def __init__(
        self,
        volumes: dict[str, np.ndarray],
        affine: np.ndarray | None = None,
        voxel_sizes: tuple[float, ...] | None = None,
        patient_metadata: PatientMetadata | None = None,
        study_metadata: StudyMetadata | None = None,
        image_metadata: ImageMetadata | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize the MRIImage.

        Args:
            volumes: A dictionary mapping modality names (e.g., 'T1', 'T1ce', 'T2', 'FLAIR', 'mask')
                     to their corresponding 3D numpy arrays.
            affine: The 4x4 affine transformation matrix relating voxel coordinates to world coordinates.  # noqa: E501
            voxel_sizes: The physical size of the voxels in mm (e.g., (1.0, 1.0, 1.0)).
            patient_metadata: Strongly typed patient metadata.
            study_metadata: Strongly typed study metadata.
            image_metadata: Strongly typed image/series metadata.
            custom_metadata: Any additional unstructured metadata.
        """
        # Validate that all volumes have the same shape
        shapes = [v.shape for v in volumes.values()]
        if shapes and not all(s == shapes[0] for s in shapes):
            raise ValueError(f"All modality volumes must have the same shape. Got: {shapes}")

        self.volumes = volumes
        self.affine = affine if affine is not None else np.eye(4)
        self.voxel_sizes = voxel_sizes

        self.patient_metadata = patient_metadata
        self.study_metadata = study_metadata
        self.image_metadata = image_metadata
        self.custom_metadata = custom_metadata or {}

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the spatial shape of the image volumes."""
        if not self.volumes:
            return ()
        return next(iter(self.volumes.values())).shape

    @property
    def modalities(self) -> list[str]:
        """Get the list of available modalities in this image."""
        return list(self.volumes.keys())

    def get_volume(self, modality: str) -> np.ndarray:
        """Get the volume for a specific modality."""
        if modality not in self.volumes:
            raise KeyError(
                f"Modality '{modality}' not found. Available modalities: {self.modalities}"
            )
        return self.volumes[modality]

    def add_volume(self, modality: str, volume: np.ndarray) -> None:
        """Add or overwrite a volume for a specific modality."""
        if self.volumes and volume.shape != self.shape:
            raise ValueError(
                f"Volume shape {volume.shape} does not match existing shape {self.shape}"
            )
        self.volumes[modality] = volume

    def remove_volume(self, modality: str) -> None:
        """Remove a volume for a specific modality."""
        if modality in self.volumes:
            del self.volumes[modality]


class Annotation:
    """Base class for all dataset annotations/labels."""

    def __init__(self, annotation_type: str, metadata: dict[str, Any] | None = None):
        self.annotation_type = annotation_type
        self.metadata = metadata or {}


class SegmentationAnnotation(Annotation):
    """
    Segmentation label.
    Contains the segmentation mask (numpy array) and a mapping of label values to names.
    """

    def __init__(
        self,
        mask: np.ndarray,
        label_map: dict[int, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(annotation_type="segmentation", metadata=metadata)
        self.mask = mask
        self.label_map = label_map or {}


class ClassificationAnnotation(Annotation):
    """
    Classification label.
    Contains a predicted or ground truth class name and an optional probability/confidence.
    """

    def __init__(
        self,
        class_name: str,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(annotation_type="classification", metadata=metadata)
        self.class_name = class_name
        self.confidence = confidence


class MRIStudy:
    """
    Container representing a single MRI examination/study.
    Holds the primary multimodal MRIImage and any associated annotations.
    Extensible for future longitudinal studies by allowing multiple images if needed.
    """

    def __init__(
        self,
        primary_image: MRIImage,
        study_id: str,
        patient_id: str,
        annotations: list[Annotation] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.primary_image = primary_image
        self.study_id = study_id
        self.patient_id = patient_id
        self.annotations = annotations or []
        self.metadata = metadata or {}
        # Extensible placeholder for future longitudinal support
        self._additional_images: list[MRIImage] = []

    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation to the study."""
        self.annotations.append(annotation)

    def get_segmentations(self) -> list[SegmentationAnnotation]:
        """Retrieve all segmentation annotations for this study."""
        return [a for a in self.annotations if isinstance(a, SegmentationAnnotation)]

    def get_classifications(self) -> list[ClassificationAnnotation]:
        """Retrieve all classification annotations for this study."""
        return [a for a in self.annotations if isinstance(a, ClassificationAnnotation)]
