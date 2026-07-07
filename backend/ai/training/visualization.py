from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ai.training.callbacks import Callback
from ai.training.events import Event


class ValidationVisualizationCallback(Callback):
    priority = 5

    def __init__(self, output_dir: str, num_cases: int = 5, seed: int = 42):
        self.output_dir = Path(output_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_cases = num_cases
        self.seed = seed
        self.epoch_predictions = []

    def on_evaluation_start(self, event: Event) -> None:
        self.epoch_predictions = []

    def on_batch_end(self, event: Event) -> None:
        if event.data.get("mode") != "val":
            return
        batch_cases = event.data.get("batch_cases", [])
        self.epoch_predictions.extend(batch_cases)

    def on_evaluation_end(self, event: Event) -> None:
        if not self.epoch_predictions:
            return

        sorted_cases = sorted(self.epoch_predictions, key=lambda x: x.get("dice", 0.0))

        worst_cases = sorted_cases[: self.num_cases]
        best_cases = sorted_cases[-self.num_cases :]

        np.random.seed(self.seed)
        if len(sorted_cases) > self.num_cases:
            random_cases = np.random.choice(sorted_cases, self.num_cases, replace=False).tolist()
        else:
            random_cases = sorted_cases

        epoch = event.data.get("epoch", 0)

        self._plot_subset(worst_cases, f"epoch_{epoch}_worst")
        self._plot_subset(best_cases, f"epoch_{epoch}_best")
        self._plot_subset(random_cases, f"epoch_{epoch}_random")

    def _plot_subset(self, cases: list, prefix: str):
        for idx, case in enumerate(cases):
            case_id = case.get("case_id", f"case_{idx}")

            image = case.get("image")
            gt = case.get("gt")
            pred = case.get("pred")

            if image is None or gt is None or pred is None:
                continue

            if image.ndim > 2:
                slice_idx = image.shape[-1] // 2
                image = image[..., slice_idx]
                gt = gt[..., slice_idx]
                pred = pred[..., slice_idx]

            _fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(image, cmap="gray")
            axes[0].set_title("Original MRI")
            axes[0].axis("off")

            axes[1].imshow(image, cmap="gray")
            axes[1].imshow(gt, alpha=0.5, cmap="Reds")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(image, cmap="gray")
            axes[2].imshow(pred, alpha=0.5, cmap="Greens")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            diff = np.abs(gt - pred)
            axes[3].imshow(image, cmap="gray")
            axes[3].imshow(diff, alpha=0.5, cmap="hot")
            axes[3].set_title("Difference Overlay")
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(self.output_dir / f"{prefix}_{case_id}.png")
            plt.close()
