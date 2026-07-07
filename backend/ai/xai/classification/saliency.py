import numpy as np
import torch

from ai.xai.shared.base import BaseExplainer
from ai.xai.shared.registry import ExplainerRegistry


@ExplainerRegistry.register("saliency", ["classification", "hybrid_cnn", "segmentation"])
class SaliencyMap(BaseExplainer):
    def __init__(self, model, device: torch.device | None = None):
        super().__init__(model, device)

    def generate_explanation(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()

        output = self.model(input_tensor)

        if target_class is None:
            if len(output.shape) == 2:  # Classification
                target_class = output.argmax(dim=1).item()
                score = output[0, target_class]
            else:  # Segmentation (or other)
                score = output.max(dim=1)[0].sum()
        else:
            if len(output.shape) == 2:
                score = output[0, target_class]
            else:
                score = output[0, target_class].sum()

        score.backward()

        saliency = input_tensor.grad.abs().squeeze().detach().cpu().numpy()

        # If multi-channel (e.g. RGB), take the max across channels
        if len(saliency.shape) == 3:
            saliency = np.max(saliency, axis=0)

        saliency = saliency - np.min(saliency)
        saliency = saliency / (np.max(saliency) + 1e-8)

        return saliency
