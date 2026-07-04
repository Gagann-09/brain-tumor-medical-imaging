from datetime import date, datetime

from pydantic import BaseModel, Field


class PatientMetadata(BaseModel):
    """Metadata related to the patient."""

    patient_id: str = Field(..., description="Unique identifier for the patient")
    patient_name: str | None = Field(None, description="Patient's name")
    patient_birth_date: date | None = Field(None, description="Patient's birth date")
    patient_sex: str | None = Field(None, description="Patient's sex (e.g., M, F, O)")
    patient_age: str | None = Field(None, description="Patient's age at the time of study")
    patient_weight: float | None = Field(None, description="Patient's weight in kg")


class StudyMetadata(BaseModel):
    """Metadata related to the medical study."""

    study_instance_uid: str = Field(..., description="Unique identifier for the study")
    study_id: str | None = Field(None, description="Study ID")
    study_date: date | None = Field(None, description="Date the study started")
    study_time: datetime | None = Field(None, description="Time the study started")
    accession_number: str | None = Field(None, description="Accession number")
    study_description: str | None = Field(None, description="Description of the study")
    referring_physician: str | None = Field(None, description="Name of referring physician")


class ImageMetadata(BaseModel):
    """Metadata specific to the image or series."""

    series_instance_uid: str = Field(..., description="Unique identifier for the series")
    series_number: int | None = Field(None, description="Number of the series")
    modality: str = Field(..., description="Modality of the image (e.g., MR, CT)")
    series_description: str | None = Field(None, description="Description of the series")
    manufacturer: str | None = Field(None, description="Manufacturer of the equipment")
    manufacturer_model_name: str | None = Field(None, description="Model name of the equipment")
    magnetic_field_strength: float | None = Field(None, description="Magnetic field strength (for MR)")
    spacing_between_slices: float | None = Field(None, description="Spacing between slices in mm")
    slice_thickness: float | None = Field(None, description="Slice thickness in mm")
    pixel_spacing: list[float] | None = Field(None, description="Physical distance between pixel centers [row, col] in mm")
