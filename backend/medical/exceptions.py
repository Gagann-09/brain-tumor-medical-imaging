class MedicalImagingError(Exception):
    """Base class for all medical imaging related errors."""

    pass


class InvalidImageFormatError(MedicalImagingError):
    """Raised when an image format is unsupported or invalid."""

    pass


class CorruptedDataError(MedicalImagingError):
    """Raised when image data is corrupted or cannot be read."""

    pass


class MissingModalityError(MedicalImagingError):
    """Raised when a required MRI modality is missing."""

    pass


class MetadataExtractionError(MedicalImagingError):
    """Raised when metadata cannot be extracted or parsed."""

    pass


class DICOMValidationError(MedicalImagingError):
    """Raised when DICOM validation fails."""

    pass
