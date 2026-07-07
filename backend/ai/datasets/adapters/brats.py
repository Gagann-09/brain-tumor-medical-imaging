from collections.abc import Iterator

import nibabel as nib
import numpy as np

from medical.domain import MRIImage, MRIStudy, SegmentationAnnotation
from medical.exceptions import MedicalImagingError

from .base import BaseDatasetAdapter


class BraTSAdapter(BaseDatasetAdapter):
    """
    Adapter for the BraTS (Brain Tumor Segmentation) dataset.
    Expects a folder structure where each patient has a subdirectory containing:
    - {ID}_t1.nii.gz
    - {ID}_t1ce.nii.gz
    - {ID}_t2.nii.gz
    - {ID}_flair.nii.gz
    - {ID}_seg.nii.gz
    """

    # Typical mapping in BraTS
    BRATS_MODALITIES = ["t1", "t1ce", "t2", "flair"]  # type: ignore  # type: ignore
    # 1: Necrotic, 2: Edema, 4: Enhancing (often mapped internally as needed)
    BRATS_LABEL_MAP = {  # type: ignore  # type: ignore
        1: "Necrotic and Non-Enhancing Tumor Core",
        2: "Peritumoral Edema",
        4: "GD-Enhancing Tumor",
    }

    def load_studies(self) -> Iterator[MRIStudy]:
        for patient_dir in self.root_dir.iterdir():
            if patient_dir.is_dir():
                yield self.load_study(patient_dir.name)

    def load_study(self, study_id: str) -> MRIStudy:
        patient_dir = self.root_dir / study_id
        if not patient_dir.exists():
            raise FileNotFoundError(f"Study directory not found: {patient_dir}")

        volumes = {}
        affine = None
        voxel_sizes = None

        for mod in self.BRATS_MODALITIES:
            # Look for the file ending with _{mod}.nii.gz or .nii
            mod_files = list(patient_dir.glob(f"*{mod}.nii*"))
            if not mod_files:
                # Modality missing; depends on config if we throw error here or let validator handle it.
                # Since BraTS expects all 4, but we want to leave strict checking to validator, we just skip.
                continue

            mod_file = mod_files[0]
            try:
                nii = nib.load(str(mod_file))
                volume = np.array(nii.get_fdata(dtype=np.float32))
                volumes[mod.upper()] = volume
                if affine is None:
                    affine = nii.affine
                    zooms = nii.header.get_zooms()
                    voxel_sizes = tuple(float(z) for z in zooms[:3]) if zooms else None
            except Exception as e:
                raise MedicalImagingError(f"Failed to load {mod_file}: {e!s}") from e

        if not volumes:
            raise MedicalImagingError(f"No valid modalities found in {patient_dir}")

        image = MRIImage(volumes=volumes, affine=affine, voxel_sizes=voxel_sizes)

        annotations = []
        seg_files = list(patient_dir.glob("*seg.nii*"))
        if seg_files:
            try:
                nii_seg = nib.load(str(seg_files[0]))
                mask = np.array(nii_seg.get_fdata(dtype=np.uint8))
                seg_annotation = SegmentationAnnotation(
                    mask=mask,
                    label_map=self.BRATS_LABEL_MAP,
                    metadata={"source": seg_files[0].name},
                )
                annotations.append(seg_annotation)
            except Exception as e:
                raise MedicalImagingError(f"Failed to load segmentation {seg_files[0]}: {e!s}") from e

        return MRIStudy(
            primary_image=image,
            study_id=study_id,
            patient_id=study_id,  # In BraTS, study_id is usually patient_id
            annotations=annotations,
        )
