import time
import json
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

import torch
from torch.profiler import profile, record_function, ProfilerActivity

class ExperimentProfiler:
    """
    Tracks and logs performance metrics.
    Supports two modes: 'development' (high-level) and 'research' (full torch.profiler).
    """
    
    def __init__(self, mode: str = "development", output_dir: str = "./outputs_val/profiler"):
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, Any] = {
            "epochs": [],
            "overall": {}
        }
        self.epoch_start_time = 0.0
        self.torch_profiler: Optional[profile] = None
        
        if self.mode == "research":
            # Initialize PyTorch Profiler for deep tracing
            self.torch_profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir / "tb_traces")),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
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
            avg_time = sum(e["time_sec"] for e in self.metrics["epochs"]) / len(self.metrics["epochs"])
            avg_throughput = sum(e["throughput_samples_per_sec"] for e in self.metrics["epochs"]) / len(self.metrics["epochs"])
            self.metrics["overall"]["avg_epoch_time_sec"] = avg_time
            self.metrics["overall"]["avg_throughput"] = avg_throughput
            
    def save_report(self, filename: str = "performance_report.json"):
        with open(self.output_dir / filename, "w") as f:
            json.dump(self.metrics, f, indent=4)
