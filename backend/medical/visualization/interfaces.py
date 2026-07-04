from abc import ABC, abstractmethod
from typing import Any

from medical.domain import MRIImage


class VisualizationEngine(ABC):
    """Abstract interface for medical image visualization."""

    @abstractmethod
    def plot_slice(
        self,
        image: MRIImage,
        modality: str,
        axis: int = 2,
        slice_idx: int = -1,
        **kwargs: Any
    ) -> Any:
        """
        Plot a single 2D slice from a 3D volume.
        
        Args:
            image: MRIImage instance.
            modality: The modality to plot (e.g., 'T1').
            axis: The axis to slice along (0=sagittal, 1=coronal, 2=axial).
            slice_idx: The index of the slice. If -1, plot the middle slice.
        """
        pass

    @abstractmethod
    def plot_overlay(
        self,
        base_image: MRIImage,
        overlay_image: MRIImage,
        base_modality: str,
        overlay_modality: str,
        axis: int = 2,
        slice_idx: int = -1,
        alpha: float = 0.5,
        **kwargs: Any
    ) -> Any:
        """
        Plot a slice with an overlay (e.g., segmentation mask or heatmap).
        """
        pass
