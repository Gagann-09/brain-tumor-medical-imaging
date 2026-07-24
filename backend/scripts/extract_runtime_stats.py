import json
import os
import platform

import torch


def get_env_info():
    info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    if torch.cuda.is_available():
        info["gpu_available"] = True
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(info["gpu_count"])]
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
    else:
        info["gpu_available"] = False

    return info

def get_experiment_stats(exp_dir):
    metrics_file = os.path.join(exp_dir, "metrics", "step_metrics.jsonl")
    if not os.path.exists(metrics_file):
        return {"error": "metrics file not found"}

    latencies = []
    with open(metrics_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                if "batch_latency" in data:
                    latencies.append(data["batch_latency"])
                elif "batch_time" in data:
                    latencies.append(data["batch_time"])
            except Exception:  # noqa: S110
                pass

    stats = {}
    if latencies:
        stats["avg_batch_latency_seconds"] = sum(latencies) / len(latencies)
        stats["total_steps"] = len(latencies)
        stats["total_training_time_seconds"] = sum(latencies)
    else:
        stats["error"] = "no batch latency data found"

    return stats

def main():
    unet_exp_dir = "outputs/experiments/exp_20260724_205306_ab8d72ea"
    armt_gan_exp_dir = "outputs/experiments/exp_20260724_205440_c079e47a"

    stats = {
        "environment": get_env_info(),
        "models": {
            "unet_baseline": get_experiment_stats(unet_exp_dir),
            "armt_gan": get_experiment_stats(armt_gan_exp_dir)
        }
    }

    with open("runtime_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("Runtime statistics extracted to runtime_statistics.json")

if __name__ == "__main__":
    main()
