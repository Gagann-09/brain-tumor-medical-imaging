from abc import ABC, abstractmethod
from typing import Any


class HardwareManager(ABC):
    """Abstract interface for hardware resource management."""

    @abstractmethod
    def set_device(self, device_ids: list[int]) -> None:
        """Set the active devices."""
        pass

    @abstractmethod
    def to_device(self, tensor: Any) -> Any:
        """Move a tensor to the primary device."""
        pass

    @abstractmethod
    def empty_cache(self) -> None:
        """Clear hardware caches (e.g., CUDA cache)."""
        pass

class PyTorchHardwareManager(HardwareManager):
    """Implementation of HardwareManager for PyTorch."""

    def __init__(self, device_type: str = "cuda"):
        import torch
        self.device_type = device_type
        self.device = torch.device("cpu")
        self.device_ids = []

        if self.device_type == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.device_type == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif self.device_type == "rocm":
            # ROCm is accessed via the 'cuda' device type in PyTorch
            # if compiled with ROCm, but we keep the logical distinction
            if torch.cuda.is_available():
                self.device = torch.device("cuda")

    def set_device(self, device_ids: list[int]) -> None:
        self.device_ids = device_ids
        # In a real setup, we might set CUDA_VISIBLE_DEVICES or use torch.cuda.set_device
        if self.device.type == "cuda" and len(device_ids) > 0:
            import torch
            torch.cuda.set_device(device_ids[0])

    def to_device(self, tensor: Any) -> Any:
        # Assuming tensor is a torch.Tensor
        if hasattr(tensor, "to"):
            return tensor.to(self.device)
        return tensor

    def empty_cache(self) -> None:
        import torch
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
