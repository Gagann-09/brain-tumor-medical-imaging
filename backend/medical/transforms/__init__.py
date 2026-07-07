"""
Transforms package for affine and coordinate mappings.
"""



import numpy as np


def apply_affine(affine: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 affine transformation to a set of 3D coordinates.

    Args:
        affine: 4x4 numpy array representing the affine matrix.
        coords: Nx3 numpy array of coordinates.

    Returns:
        Nx3 numpy array of transformed coordinates.
    """
    if affine.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine matrix, got {affine.shape}")

    # Convert to homogeneous coordinates
    homogenous_coords = np.hstack([coords, np.ones((coords.shape[0], 1))])

    # Apply affine
    transformed = homogenous_coords @ affine.T

    return transformed[:, :3]


def get_center_coordinate(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Get the center coordinate of an image given its shape.
    """
    return tuple(s // 2 for s in shape)
