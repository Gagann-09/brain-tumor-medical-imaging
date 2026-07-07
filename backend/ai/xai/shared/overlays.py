from typing import Any

import cv2
import nibabel as nib
import numpy as np


class OverlayLayers:
    """Holds independent layers for overlay rendering."""

    def __init__(
        self,
        mri: np.ndarray,
        heatmap: np.ndarray,
        mask: np.ndarray | None = None,
        boundaries: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.mri = mri
        self.heatmap = heatmap
        self.mask = mask
        self.boundaries = boundaries
        self.metadata = metadata or {}


class OverlayGenerator:
    """Generates overlaid explainability visuals."""

    @staticmethod
    def apply_colormap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Applies a colormap to a [0, 1] normalized heatmap."""
        # Ensure heatmap is between 0 and 1
        heatmap_norm = np.clip(heatmap, 0, 1)
        heatmap_uint8 = np.uint8(255 * heatmap_norm)
        return cv2.applyColorMap(heatmap_uint8, colormap)

    @staticmethod
    def render_2d(layers: OverlayLayers, alpha: float = 0.5) -> np.ndarray:
        """Renders layers into a single 2D RGB numpy array."""
        mri_norm = (layers.mri - layers.mri.min()) / (layers.mri.max() - layers.mri.min() + 1e-8)
        mri_uint8 = np.uint8(255 * mri_norm)

        if len(mri_uint8.shape) == 2:
            base_img = cv2.cvtColor(mri_uint8, cv2.COLOR_GRAY2RGB)
        else:
            base_img = mri_uint8

        color_heatmap = OverlayGenerator.apply_colormap(layers.heatmap)

        # Combine base and heatmap
        overlaid = cv2.addWeighted(base_img, 1 - alpha, color_heatmap, alpha, 0)

        # Add mask if present (e.g. green tint)
        if layers.mask is not None:
            mask_rgb = np.zeros_like(base_img)
            mask_rgb[layers.mask > 0] = [0, 255, 0]
            overlaid = np.where(
                layers.mask[..., None] > 0,
                cv2.addWeighted(overlaid, 0.7, mask_rgb, 0.3, 0),
                overlaid,
            )

        # Add boundaries if present
        if layers.boundaries is not None:
            overlaid[layers.boundaries > 0] = [255, 0, 0]  # Red boundaries

        return overlaid

    @staticmethod
    def render_2d_with_metadata(layers: OverlayLayers, alpha: float = 0.5) -> np.ndarray:
        """Renders 2D overlay and burns metadata into it."""
        img_array = OverlayGenerator.render_2d(layers, alpha)

        if not layers.metadata:
            return img_array

        y_offset = 20
        for k, v in layers.metadata.items():
            text = f"{k}: {v}"
            cv2.putText(
                img_array,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += 20

        return img_array

    @staticmethod
    def export_numpy(overlaid: np.ndarray, filepath: str):
        np.save(filepath, overlaid)

    @staticmethod
    def export_png(overlaid: np.ndarray, filepath: str):
        # Convert RGB to BGR for cv2 saving
        cv2.imwrite(filepath, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))

    @staticmethod
    def export_nifti(overlaid: np.ndarray, filepath: str):
        img = nib.Nifti1Image(overlaid, np.eye(4))
        nib.save(img, filepath)
