import json
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

import matplotlib.pyplot as plt
import torch
from torch.profiler import ProfilerActivity, profile

from ai.evaluation.robustness.analyzers import LearningCurveAnalyzer
from ai.evaluation.robustness.health_score import ExperimentHealthScoreCalculator
from ai.evaluation.robustness.report import GeneralizationReport


class ExperimentProfiler:
    """
    Tracks and logs performance metrics.
    Supports two modes: 'development' (high-level) and 'research' (full torch.profiler).
    """

    def __init__(self, mode: str = "development", output_dir: str = "./outputs_val/profiler"):
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: dict[str, Any] = {"epochs": [], "overall": {}}
        self.epoch_start_time = 0.0
        self.torch_profiler: profile | None = None

        if self.mode == "research":
            # Initialize PyTorch Profiler for deep tracing
            self.torch_profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(self.output_dir / "tb_traces")
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

    def start_epoch(self, epoch: int):
        self.epoch_start_time = time.time()

    def end_epoch(self, epoch: int, num_samples: int):
        epoch_time = time.time() - self.epoch_start_time
        throughput = num_samples / epoch_time if epoch_time > 0 else 0

        epoch_metrics = {
            "epoch": epoch,
            "time_sec": epoch_time,
            "throughput_samples_per_sec": throughput,
            "cpu_percent": psutil.cpu_percent(interval=None) if psutil else 0.0,
            "cpu_memory_mb": psutil.virtual_memory().used / (1024**2) if psutil else 0.0,
        }

        if torch.cuda.is_available():
            epoch_metrics["gpu_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
            torch.cuda.reset_peak_memory_stats()

        self.metrics["epochs"].append(epoch_metrics)

    def step_profiler(self):
        """Called per batch to step the torch.profiler in research mode."""
        if self.torch_profiler:
            self.torch_profiler.step()

    def start_training(self):
        if self.torch_profiler:
            self.torch_profiler.start()

    def stop_training(self):
        if self.torch_profiler:
            self.torch_profiler.stop()

        # Compute overall averages
        if self.metrics["epochs"]:
            avg_time = sum(e["time_sec"] for e in self.metrics["epochs"]) / len(
                self.metrics["epochs"]
            )
            avg_throughput = sum(
                e["throughput_samples_per_sec"] for e in self.metrics["epochs"]
            ) / len(self.metrics["epochs"])
            self.metrics["overall"]["avg_epoch_time_sec"] = avg_time
            self.metrics["overall"]["avg_throughput"] = avg_throughput

    def save_report(self, filename: str = "performance_report.json"):
        with open(self.output_dir / filename, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def generate_generalization_report(
        self,
        experiment_id: str,
        train_history: list,
        val_history: list,
        has_leakage: bool,
        failed_cv: bool,
        failed_calibration: bool,
        performance_score: float,
        generalization_score: float,
        stability_score: float,
        calibration_score: float,
        reproducibility_score: float,
        data_quality_score: float,
        excessive_gap: bool,
        has_nan_inf: bool,
    ) -> GeneralizationReport:

        # 1. Analyze learning curves
        curve_analysis = LearningCurveAnalyzer.analyze_curve(train_history, val_history)

        # 2. Compute Health Score
        health_score = ExperimentHealthScoreCalculator.calculate(
            performance_score,
            generalization_score,
            stability_score,
            calibration_score,
            reproducibility_score,
            data_quality_score,
            has_leakage,
            has_nan_inf,
            excessive_gap,
            failed_cv,
            failed_calibration,
        )

        # 3. Generate learning curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_history, label="Train Loss")
        plt.plot(val_history, label="Validation Loss")
        plt.title("Learning Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = self.output_dir / f"{experiment_id}_learning_curve.png"
        plt.savefig(plot_path)
        plt.close()

        # 4. Create report
        report = GeneralizationReport(
            experiment_id=experiment_id,
            overfitting_status=curve_analysis.get("divergence", False),
            underfitting_status=curve_analysis.get("plateau", False),
            leakage_status=has_leakage,
            generalization_gap={},  # Can be populated from calculator output if needed
            cross_validation_summary={},
            calibration_summary={},
            experiment_health_score=health_score,
            promotion_recommendation="Promote"
            if health_score >= 70.0
            and not has_leakage
            and not failed_cv
            and not failed_calibration
            and not excessive_gap
            and not has_nan_inf
            else "Do Not Promote",
            curve_analysis=curve_analysis,
        )

        # 5. Save report
        with open(self.output_dir / f"{experiment_id}_generalization.json", "w") as f:
            f.write(report.to_json())

        return report
