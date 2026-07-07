from pathlib import Path

import pydicom

from medical.exceptions import CorruptedDataError


def load_dicom_file(filepath: str | Path) -> pydicom.dataset.FileDataset:
    """
    Load a single DICOM file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    try:
        dataset = pydicom.dcmread(str(path))
        return dataset
    except Exception as e:
        raise CorruptedDataError(f"Failed to read DICOM file {path}: {e!s}") from e


def load_dicom_series(directory: str | Path) -> list[pydicom.dataset.FileDataset]:
    """
    Load all DICOM files from a given directory.
    Note: It does not group by series, it just loads everything in the directory.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    datasets = []
    for file_path in dir_path.iterdir():
        if file_path.is_file():
            try:
                # We attempt to read every file. If it fails, we ignore it (might not be DICOM)
                ds = pydicom.dcmread(str(file_path))
                datasets.append(ds)
            except pydicom.errors.InvalidDicomError:
                continue
            except Exception:  # noqa: S112
                continue

    if not datasets:
        raise CorruptedDataError(f"No valid DICOM files found in directory: {dir_path}")

    return datasets
