
import numpy as np
import pydicom


def extract_volume_and_affine(datasets: list[pydicom.dataset.FileDataset]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of DICOM datasets (representing a series) into a 3D numpy array
    and compute its affine transformation matrix.
    """
    # Sort datasets by slice location or instance number to ensure correct z-axis ordering
    # Usually sorted by ImagePositionPatient Z-coordinate if available
    try:
        datasets.sort(key=lambda ds: ds.ImagePositionPatient[2])
    except AttributeError:
        datasets.sort(key=lambda ds: ds.InstanceNumber)

    slices = []
    for ds in datasets:
        # Simple extraction. Real-world needs rescaling (RescaleSlope, RescaleIntercept)
        pixel_array = ds.pixel_array
        if "RescaleSlope" in ds and "RescaleIntercept" in ds:
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        slices.append(pixel_array)

    volume = np.stack(slices, axis=-1)

    # Compute a basic affine matrix (Identity for now, actual math involves ImageOrientationPatient)
    # This is a placeholder for the actual affine computation which requires cross products
    affine = np.eye(4)

    return volume, affine
