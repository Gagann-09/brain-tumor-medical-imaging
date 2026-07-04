import torch
import torch.nn.functional as F

class Resize3D:
    """Resizes a 3D tensor to a fixed target shape (D, H, W)."""
    def __init__(self, target_shape: tuple[int, int, int]):
        self.target_shape = target_shape
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x expected to be (C, D, H, W)
        if x.ndim == 4:
            x = x.unsqueeze(0) # (1, C, D, H, W)
            x = F.interpolate(x, size=self.target_shape, mode='trilinear', align_corners=False)
            return x.squeeze(0)
        return x

class NormalizeIntensity:
    """Z-score normalizes the non-zero intensity values of the tensor."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mask = x > 0
        if not mask.any():
            return x
        mean = x[mask].mean()
        std = x[mask].std()
        if std > 1e-8:
            x[mask] = (x[mask] - mean) / std
        return x
