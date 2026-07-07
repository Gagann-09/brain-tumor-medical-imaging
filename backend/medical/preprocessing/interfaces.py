from abc import ABC, abstractmethod
from typing import Any

from medical.domain import MRIImage


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessing steps."""

    @abstractmethod
    def process(self, image: MRIImage, **kwargs: Any) -> MRIImage:
        """
        Apply the preprocessing step to the given MRIImage.

        Args:
            image: The input MRIImage.
            **kwargs: Additional parameters for the preprocessing step.

        Returns:
            A new MRIImage instance with the processed data.
        """
        pass


class SkullStripper(BasePreprocessor):
    """Interface for skull stripping algorithms."""

    pass


class Registrator(BasePreprocessor):
    """Interface for image registration algorithms."""

    @abstractmethod
    def register(self, moving_image: MRIImage, fixed_image: MRIImage, **kwargs: Any) -> MRIImage:
        """
        Register a moving image to a fixed (reference) image.
        """
        pass

    def process(self, image: MRIImage, **kwargs: Any) -> MRIImage:
        """
        Default process method might not be directly applicable without a fixed image,
        but implementations can provide a default atlas if fixed_image is not provided in kwargs.
        """
        fixed_image = kwargs.get("fixed_image")
        if not fixed_image:
            raise ValueError(
                "Registrator requires a 'fixed_image' in kwargs for default process method."
            )
        return self.register(image, fixed_image, **kwargs)


class BiasCorrector(BasePreprocessor):
    """Interface for bias field correction algorithms (e.g., N4)."""

    pass


class Resampler(BasePreprocessor):
    """Interface for image resampling algorithms."""

    @abstractmethod
    def resample_to_spacing(
        self, image: MRIImage, target_spacing: tuple[float, ...], **kwargs: Any
    ) -> MRIImage:
        """
        Resample the image to a specific voxel spacing.
        """
        pass


class IntensityNormalizer(BasePreprocessor):
    """Interface for intensity normalization algorithms (e.g., Z-score, Min-Max)."""

    pass


class Denoiser(BasePreprocessor):
    """Interface for denoising algorithms."""

    pass
