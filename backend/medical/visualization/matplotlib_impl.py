import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:
    plt = None

from typing import Any

from medical.domain import MRIImage
from medical.exceptions import MedicalImagingError
from medical.visualization.interfaces import VisualizationEngine


class MatplotlibVisualizationEngine(VisualizationEngine):
    """Matplotlib implementation of the visualization engine."""

    def __init__(self):
        if plt is None:
            raise MedicalImagingError("matplotlib is required for MatplotlibVisualizationEngine")

    def _get_slice(self, volume: np.ndarray, axis: int, slice_idx: int) -> np.ndarray:
        if slice_idx == -1:
            slice_idx = volume.shape[axis] // 2

        if axis == 0:
            return volume[slice_idx, :, :]
        elif axis == 1:
            return volume[:, slice_idx, :]
        elif axis == 2:
            return volume[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid axis {axis}. Must be 0, 1, or 2.")

    def plot_slice(
        self,
        image: MRIImage,
        modality: str,
        axis: int = 2,
        slice_idx: int = -1,
        cmap: str = 'gray',
        **kwargs: Any
    ) -> "Figure":
        volume = image.get_volume(modality)
        slice_data = self._get_slice(volume, axis, slice_idx)

        fig, ax = plt.subplots(**kwargs)
        ax.imshow(slice_data.T, cmap=cmap, origin='lower')
        ax.axis('off')
        return fig

    def plot_overlay(
        self,
        base_image: MRIImage,
        overlay_image: MRIImage,
        base_modality: str,
        overlay_modality: str,
        axis: int = 2,
        slice_idx: int = -1,
        alpha: float = 0.5,
        base_cmap: str = 'gray',
        overlay_cmap: str = 'jet',
        **kwargs: Any
    ) -> "Figure":
        base_volume = base_image.get_volume(base_modality)
        overlay_volume = overlay_image.get_volume(overlay_modality)

        if base_volume.shape != overlay_volume.shape:
            raise ValueError(f"Shape mismatch: {base_volume.shape} vs {overlay_volume.shape}")

        base_slice = self._get_slice(base_volume, axis, slice_idx)
        overlay_slice = self._get_slice(overlay_volume, axis, slice_idx)

        # Mask out zero/background in the overlay to be transparent
        overlay_masked = np.ma.masked_where(overlay_slice == 0, overlay_slice)

        fig, ax = plt.subplots(**kwargs)
        ax.imshow(base_slice.T, cmap=base_cmap, origin='lower')
        ax.imshow(overlay_masked.T, cmap=overlay_cmap, alpha=alpha, origin='lower')
        ax.axis('off')
        return fig
