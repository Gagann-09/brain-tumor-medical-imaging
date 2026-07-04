from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelCardConfig(BaseModel):
    model_name: str
    model_version: str = "v1.0"
    architecture: str
    description: str
    limitations: list[str] = Field(default_factory=list)
    intended_use: str = "Research use only. Not for clinical diagnosis."
    preprocessing_requirements: list[str] = Field(default_factory=list)
    author: str = "AI Team"

class ModelCard(BaseModel):
    """Automatically generated documentation for trained models."""

    model_details: ModelCardConfig
    training_config: dict[str, Any]
    configuration_checksum: str = "unknown"
    dataset_info: dict[str, Any]
    dataset_version: str = "unknown"
    metrics: dict[str, Any]
    benchmark_summary: dict[str, Any] = Field(default_factory=dict)
    experiment_health_score: float = 0.0
    generalization_report_ref: str = ""
    
    software_versions: dict[str, str] = Field(default_factory=dict)
    hardware_information: str = "unknown"
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    git_hash: str = "unknown"

    def save(self, path: str | Path) -> None:
        """Saves the model card as a JSON file."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    def export_markdown(self, path: str | Path) -> None:
        """Exports the model card as a human-readable Markdown file."""
        md = f"# Model Card: {self.model_details.model_name}\n\n"
        md += f"**Architecture:** {self.model_details.architecture}\n"
        md += f"**Author:** {self.model_details.author}\n"
        md += f"**Date:** {self.timestamp.isoformat()}\n\n"

        md += "## Description\n"
        md += f"{self.model_details.description}\n\n"
        
        md += "## Experiment Health\n"
        md += f"- **Health Score**: {self.experiment_health_score:.2f}\n"
        if self.generalization_report_ref:
            md += f"- **Generalization Report**: {self.generalization_report_ref}\n\n"

        md += "## Metrics\n"
        for k, v in self.metrics.items():
            if isinstance(v, dict):
                md += f"- **{k}**:\n"
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        md += f"  - **{sub_k}**: {sub_v}\n"
                    else:
                        md += f"  - **{sub_k}**: {sub_v:.4f}\n"
            else:
                md += f"- **{k}**: {v:.4f}\n"
        md += "\n"

        md += "## Limitations\n"
        for lim in self.model_details.limitations:
            md += f"- {lim}\n"
        md += "\n"
        
        md += "## Preprocessing Requirements\n"
        for req in self.model_details.preprocessing_requirements:
            md += f"- {req}\n"
        md += "\n"
        
        md += "## Environment\n"
        md += f"**Hardware:** {self.hardware_information}\n"
        for k, v in self.software_versions.items():
            md += f"- **{k}**: {v}\n"
        md += "\n"

        with open(path, "w") as f:
            f.write(md)
