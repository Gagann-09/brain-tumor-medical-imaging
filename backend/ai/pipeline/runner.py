import abc
from typing import Any, Dict
from pathlib import Path
from datetime import datetime

from ai.pipeline.profiler import ExperimentProfiler
from ai.models.model_card import ModelCard, ModelCardConfig
from ai.model_registry.registry import ModelRegistry, ModelRegistration

class BaseValidationRunner(abc.ABC):
    """
    Abstract validation runner providing shared infrastructure for benchmarking, 
    profiling, model card generation, and promotion logic.
    """
    def __init__(self, output_dir: str, profile_name: str = "mock"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile_name = profile_name
        
        profiler_mode = "research" if profile_name == "research" else "development"
        self.profiler = ExperimentProfiler(mode=profiler_mode, output_dir=str(self.output_dir / "profiler"))
        
    @abc.abstractmethod
    def run(self) -> None:
        """Executes the specific validation workflow (segmentation or classification)."""
        pass
        
    def generate_model_card(self, model_name: str, architecture: str, description: str, 
                            training_config: Dict[str, Any], metrics: Dict[str, float], 
                            benchmark_record: Dict[str, Any], hardware: str, task: str) -> None:
        """Generates and registers the model card."""
        mc_config = ModelCardConfig(
            model_name=model_name,
            architecture=architecture,
            description=description,
        )
        mc = ModelCard(
            model_details=mc_config,
            training_config=training_config,
            configuration_checksum="checksum_placeholder",
            dataset_info={"name": "BraTS Mock" if self.profile_name == "mock" else "BraTS"},
            dataset_version="v1.0",
            metrics=metrics,
            benchmark_summary=benchmark_record,
            software_versions={"runner": "1.0"},
            hardware_information=hardware
        )
        
        # Save locally
        json_path = self.output_dir / f"{model_name.lower()}_model_card.json"
        md_path = self.output_dir / f"{model_name.lower()}_model_card.md"
        mc.save(json_path)
        mc.export_markdown(md_path)
        
        # Register in Model Registry
        reg = ModelRegistration(
            name=model_name,
            version="v1.0",
            task=task,
            dataset="BraTS",
            metrics=metrics,
            checksum="checksum_placeholder",
            framework="PyTorch",
            artifact_location=str(json_path)
        )
        ModelRegistry.register(reg)
