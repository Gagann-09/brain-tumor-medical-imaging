import numpy as np


def normalize_min_max(volume: np.ndarray) -> np.ndarray:
    """
    Normalize a volume to [0, 1] range using Min-Max scaling.
    """
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val == min_val:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - min_val) / (max_val - min_val)


def normalize_z_score(volume: np.ndarray) -> np.ndarray:
    """
    Normalize a volume using Z-score (zero mean, unit variance).
    """
    mean = np.mean(volume)
    std = np.std(volume)
    if std == 0:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - mean) / std


def crop_to_non_zero(volume: np.ndarray) -> np.ndarray:
    """
    Crop the volume to its non-zero bounding box.
    """
    non_zero_indices = np.nonzero(volume)
    if len(non_zero_indices[0]) == 0:
        return volume

    min_indices = [np.min(idx) for idx in non_zero_indices]
    max_indices = [np.max(idx) for idx in non_zero_indices]

    slices = tuple(
        slice(min_idx, max_idx + 1) for min_idx, max_idx in zip(min_indices, max_indices, strict=False)
    )
    return volume[slices]
