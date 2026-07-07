import argparse

import torch

from ai.classification.benchmark.benchmark import ClassificationBenchmark
from ai.classification.datasets.sample import ClassificationSample
from ai.classification.inference.inference import ClassificationInference
from ai.classification.models.baseline_cnn import BaselineCNNClassifier, BaselineCNNConfig
from ai.classification.models.hybrid_cnn import HybridCNNClassifier, HybridCNNConfig
from ai.classification.pipeline.input_pipeline import ClassificationInputPipeline
from ai.models.promotion import ModelPromotionManager
from ai.pipeline.runner import BaseValidationRunner


class ClassificationValidationRunner(BaseValidationRunner):
    def generate_mock_dataset(self):
        """Generates mock ClassificationSamples for testing infrastructure."""
        samples = []
        for i in range(10):
            x = torch.randn(4, 32, 32, 32)
            label = {"target": torch.randint(0, 2, (1,)).item()}
            samples.append(
                ClassificationSample(sample_id=f"mock_{i}", image_tensor=x, labels=label)
            )
        return samples

    def run(self):

        # 1. Dataset setup
        if self.profile_name == "mock":
            studies = self.generate_mock_dataset()
        else:
            raise NotImplementedError(
                "Real dataset loading not yet implemented for classification mock run."
            )

        train_studies = studies[:6]
        val_studies = studies[6:]

        # 2. Pipeline setup
        pipeline = ClassificationInputPipeline()
        promo_manager = ModelPromotionManager(registry_dir=str(self.output_dir))

        # 3. Train & Eval Baseline CNN
        base_cfg = BaselineCNNConfig(in_channels=4)
        base_model = BaselineCNNClassifier(base_cfg)

        self.profiler.start_training()
        self.profiler.start_epoch(1)
        # Mock training sleep
        import time

        time.sleep(0.5)
        self.profiler.end_epoch(1, len(train_studies))
        self.profiler.stop_training()

        base_infer = ClassificationInference(base_model, pipeline)
        benchmark = ClassificationBenchmark(base_infer)
        base_res = benchmark.evaluate(val_studies, num_classes=base_cfg.num_classes)

        promo_manager.register_model("BaselineCNN", "v1.0")
        promo_manager.mark_as_candidate("BaselineCNN", "v1.0")

        self.generate_model_card(
            model_name="BaselineCNN",
            architecture="3D CNN",
            description="Baseline 3D CNN for tumor classification.",
            training_config=base_cfg.model_dump(mode="json"),
            metrics=base_res,
            benchmark_record={"status": "baseline_established"},
            hardware="CPU",
            task="classification",
        )

        # 4. Train & Eval Hybrid CNN
        hybrid_cfg = HybridCNNConfig()
        hybrid_model = HybridCNNClassifier(hybrid_cfg)

        self.profiler.start_training()
        self.profiler.start_epoch(1)
        time.sleep(0.5)
        self.profiler.end_epoch(1, len(train_studies))
        self.profiler.stop_training()
        self.profiler.save_report("performance_report.json")

        ClassificationInference(hybrid_model, pipeline)
        hybrid_res = benchmark.evaluate(val_studies, num_classes=hybrid_cfg.num_classes)

        promo_manager.register_model("HybridCNN", "v1.0")
        promo_manager.mark_as_candidate("HybridCNN", "v1.0")

        success = promo_manager.evaluate_classification_candidate(
            candidate_model="HybridCNN",
            candidate_version="v1.0",
            baseline_metrics=base_res,
            candidate_metrics=hybrid_res,
            benchmark_id="mock_classification_bench",
        )

        if success:
            pass
        else:
            pass

        self.generate_model_card(
            model_name="HybridCNN",
            architecture="Hybrid CNN (Multimodal Early Fusion)",
            description="Advanced Multimodal 3D CNN for tumor classification.",
            training_config=hybrid_cfg.model_dump(mode="json"),
            metrics=hybrid_res,
            benchmark_record={"status": "candidate_evaluated", "promoted": success},
            hardware="CPU",
            task="classification",
        )

        # Generate Structured Comparison Artifact
        comp_md = "# Classification Models Comparison\n\n"
        comp_md += "## Baseline CNN Metrics\n"
        for k, v in base_res.items():
            comp_md += f"- **{k}**: {v}\n"
        comp_md += "\n## Hybrid CNN Metrics\n"
        for k, v in hybrid_res.items():
            comp_md += f"- **{k}**: {v}\n"
        comp_md += "\n## Promotion Recommendation\n"
        comp_md += (
            "Promoted"
            if success
            else f"Failed: {promo_manager.records['HybridCNN_v1.0'].promotion_reason}\n"
        )

        with open(self.output_dir / "classification_comparison.md", "w") as f:
            f.write(comp_md)


def main():
    parser = argparse.ArgumentParser(description="Run Classification Validation Pipeline")
    parser.add_argument(
        "--profile",
        type=str,
        default="mock",
        choices=["mock", "development", "research", "production"],
    )
    parser.add_argument("--output_dir", type=str, default="./outputs_val/classification_run")
    args = parser.parse_args()

    runner = ClassificationValidationRunner(output_dir=args.output_dir, profile_name=args.profile)
    runner.run()


if __name__ == "__main__":
    main()
