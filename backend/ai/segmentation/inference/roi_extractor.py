from abc import ABC, abstractmethod

import torch


class ROIStrategy(ABC):
    """Base strategy for extracting a Region of Interest (ROI) from a segmentation mask."""

    @abstractmethod
    def extract(
        self, image: torch.Tensor, mask: torch.Tensor, padding: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts ROI from the image based on the mask.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The cropped image and the cropped mask.
        """
        pass


class BoundingBoxStrategy(ROIStrategy):
    """Crops the image to the 3D bounding box of the non-zero mask elements, with optional padding."""

    def extract(
        self, image: torch.Tensor, mask: torch.Tensor, padding: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Assume mask shape is (C, D, H, W) or (C, H, W). We find non-zero indices.
        non_zero_indices = torch.nonzero(mask)
        if len(non_zero_indices) == 0:
            # If empty mask, return the original or a center crop (returning original for safety)
            return image, mask

        # Get min and max across spatial dimensions (skipping channel dim 0)
        mins = non_zero_indices[:, 1:].min(dim=0)[0]
        maxs = non_zero_indices[:, 1:].max(dim=0)[0]

        # Apply padding and clamp to image boundaries
        spatial_dims = image.shape[1:]
        mins = torch.clamp(mins - padding, min=0)
        maxs = torch.clamp(maxs + padding + 1, max=torch.tensor(spatial_dims, device=mins.device))

        if image.ndim == 4:  # (C, D, H, W)
            return (
                image[:, mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]],
                mask[:, mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]],
            )
        elif image.ndim == 3:  # (C, H, W)
            return (
                image[:, mins[0] : maxs[0], mins[1] : maxs[1]],
                mask[:, mins[0] : maxs[0], mins[1] : maxs[1]],
            )
        return image, mask


class MaskCropStrategy(BoundingBoxStrategy):
    """
    Crops the image to the bounding box and masks out the background (sets non-tumor pixels to zero).  # noqa: E501
    """

    def extract(
        self, image: torch.Tensor, mask: torch.Tensor, padding: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cropped_image, cropped_mask = super().extract(image, mask, padding)
        # Mask out background
        masked_image = cropped_image * (cropped_mask > 0).float()
        return masked_image, cropped_mask


class LargestComponentStrategy(ROIStrategy):
    """
    Finds the largest connected component in the mask, and crops to its bounding box.
    Useful for removing small false-positive segmented regions before classification.
    """

    def extract(
        self, image: torch.Tensor, mask: torch.Tensor, padding: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # In a full implementation, we'd use scipy.ndimage.label or MONAI's KeepLargestConnectedComponent.
        # This is a stub that falls back to BoundingBoxStrategy for now.
        # Future implementation will isolate the largest component.
        return BoundingBoxStrategy().extract(image, mask, padding)


class ROIExtractor:
    """
    Transforms whole-volume segmentation outputs into localized ROIs for classification.
    """

    def __init__(self, strategy: ROIStrategy = BoundingBoxStrategy(), padding: int = 0):
        self.strategy = strategy
        self.padding = padding

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.strategy.extract(image, mask, self.padding)
