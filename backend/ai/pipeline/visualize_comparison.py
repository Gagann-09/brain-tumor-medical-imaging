from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


class VisualizationManager:
    """Manages generation of visual comparisons for experimental validation."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_comparison_grid(
        self,
        image: torch.Tensor,
        ground_truth: torch.Tensor,
        unet_pred: torch.Tensor,
        armt_pred: torch.Tensor,
        sample_idx: int,
        slice_idx: int = -1,
    ) -> str:
        """
        Generates a 1x5 image grid for the given tensors (C, D, H, W) or (C, H, W).
        Returns the path to the saved image.
        """
        # Assume inputs from DataLoader are (B, C, D, H, W) or (B, C, H, W)
        if image.ndim == 5:  # (B, C, D, H, W)
            if slice_idx == -1:
                slice_idx = image.shape[2] // 2
            img_slice = image[0, 0, slice_idx, :, :].cpu().numpy()
            gt_slice = ground_truth[0, 0, slice_idx, :, :].cpu().numpy()
            unet_slice = unet_pred[0, 0, slice_idx, :, :].cpu().numpy()
            armt_slice = armt_pred[0, 0, slice_idx, :, :].cpu().numpy()
        elif image.ndim == 4:  # (B, C, H, W) or (C, D, H, W)
            # Assuming (B, C, H, W) for 2D inference
            img_slice = image[0, 0, :, :].cpu().numpy()
            gt_slice = ground_truth[0, 0, :, :].cpu().numpy()
            unet_slice = unet_pred[0, 0, :, :].cpu().numpy()
            armt_slice = armt_pred[0, 0, :, :].cpu().numpy()

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        # 1. Original
        axes[0].imshow(img_slice, cmap="gray")
        axes[0].set_title("Original MRI (T1ce)")
        axes[0].axis("off")

        # 2. Ground Truth
        axes[1].imshow(img_slice, cmap="gray")
        axes[1].imshow(gt_slice, cmap="Reds", alpha=0.5)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        # 3. U-Net
        axes[2].imshow(img_slice, cmap="gray")
        axes[2].imshow(unet_slice, cmap="Blues", alpha=0.5)
        axes[2].set_title("U-Net Prediction")
        axes[2].axis("off")

        # 4. ARMT-GAN
        axes[3].imshow(img_slice, cmap="gray")
        axes[3].imshow(armt_slice, cmap="Greens", alpha=0.5)
        axes[3].set_title("ARMT-GAN Prediction")
        axes[3].axis("off")

        # 5. Overlay Difference (ARMT vs GT)
        # Red = False Negative, Green = True Positive, Blue = False Positive
        rgb_overlay = np.zeros((*gt_slice.shape, 3))
        rgb_overlay[..., 0] = gt_slice * (1 - armt_slice)  # FN (Red)
        rgb_overlay[..., 1] = gt_slice * armt_slice  # TP (Green)
        rgb_overlay[..., 2] = armt_slice * (1 - gt_slice)  # FP (Blue)

        axes[4].imshow(img_slice, cmap="gray")
        axes[4].imshow(rgb_overlay, alpha=0.6)
        axes[4].set_title("Overlay (G=TP, R=FN, B=FP)")
        axes[4].axis("off")

        plt.tight_layout()
        out_path = self.output_dir / f"comparison_{sample_idx}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        return str(out_path)
