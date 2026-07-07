import time
from typing import Any

import numpy as np
import torch

from ai.models.benchmark_registry import BenchmarkRecord, BenchmarkRegistry


# Simplified metrics for benchmarking
def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = (torch.sigmoid(preds) > 0.5).float()

    # Flatten
    p = preds.view(-1)
    t = targets.view(-1)

    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    tn = ((1 - p) * (1 - t)).sum().item()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Dummy Hausdorff for speed here (would use monai.metrics.compute_hausdorff_distance)
    hausdorff_95 = 0.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "hausdorff_95": hausdorff_95,
    }


def benchmark_model(
    model: torch.nn.Module, data_loader: Any, device: torch.device
) -> dict[str, Any]:
    model.to(device)
    model.eval()

    all_metrics = []
    latencies = []

    has_nan = False

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            targets = batch.get("label", None)

            # Latency
            start_time = time.perf_counter()
            if hasattr(model, "prediction_step"):
                preds = model.prediction_step(batch, 0).predictions.to(device)
            else:
                preds = model(images)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.perf_counter()

            latencies.append(end_time - start_time)

            # NaN check
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                has_nan = True

            if targets is not None:
                targets = targets.to(device)
                all_metrics.append(calculate_metrics(preds, targets))

    # Aggregate
    aggregated = (
        {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        if all_metrics
        else {}
    )

    # System metrics
    avg_latency = np.mean(latencies) if latencies else 0.0
    fps = 1.0 / avg_latency if avg_latency > 0 else 0.0

    param_count = sum(p.numel() for p in model.parameters())

    mem_alloc = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )

    aggregated.update(
        {
            "latency_sec": avg_latency,
            "fps": fps,
            "param_count": param_count,
            "gpu_mem_mb": mem_alloc,
            "has_nan": float(has_nan),
        }
    )

    return aggregated


def run_comparison(unet_model, armt_model, test_loader, registry_dir, repro_meta=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet_results = benchmark_model(unet_model, test_loader, device)

    armt_results = benchmark_model(armt_model, test_loader, device)

    registry = BenchmarkRegistry(registry_dir)

    outcome = (
        "ARMT-GAN Outperforms UNet"
        if armt_results.get("dice", 0) > unet_results.get("dice", 0)
        else "UNet Retains Baseline"
    )
    if armt_results.get("has_nan"):
        outcome = "ARMT-GAN Failed (NaN detected)"

    record = BenchmarkRecord(
        benchmark_id=f"bench_{int(time.time())}",
        models_compared=["UNetBaseline", "ARMTGANModel"],
        dataset="BraTS_Test",
        metrics={"UNetBaseline": unet_results, "ARMTGANModel": armt_results},
        hardware=device.type,
        configuration={},
        reproducibility=repro_meta or {},
        outcome=outcome,
    )

    registry.add_benchmark(record)
    return record, unet_results, armt_results
