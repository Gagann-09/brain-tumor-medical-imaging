import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BenchmarkInfo:
    generation_time_ms: float
    peak_memory_mb: float | None = None


@dataclass
class ExplainabilityReport:
    prediction: dict[str, Any]
    confidence: float
    model_version: str
    dataset_version: str
    explanation_method: str
    feature_layer: str
    generated_artifacts: dict[str, str]  # e.g., {"overlay_png": "/path/to/overlay.png"}
    benchmark: BenchmarkInfo
    software_versions: dict[str, str]
    runtime: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        md = "# Explainability Report\n\n"
        md += f"**Method**: {self.explanation_method}\n"
        md += f"**Runtime**: {self.runtime}\n\n"

        md += "## Prediction Details\n"
        md += f"- **Prediction**: {self.prediction}\n"
        md += f"- **Confidence**: {self.confidence:.4f}\n\n"

        md += "## Model & Dataset\n"
        md += f"- **Model Version**: {self.model_version}\n"
        md += f"- **Dataset Version**: {self.dataset_version}\n"
        md += f"- **Feature Layer**: {self.feature_layer}\n\n"

        md += "## Benchmark\n"
        md += f"- **Generation Time**: {self.benchmark.generation_time_ms:.2f} ms\n"
        if self.benchmark.peak_memory_mb:
            md += f"- **Peak Memory**: {self.benchmark.peak_memory_mb:.2f} MB\n\n"

        md += "## Generated Artifacts\n"
        for name, path in self.generated_artifacts.items():
            md += f"- **{name}**: `{path}`\n"

        md += "\n## Software Versions\n"
        for name, version in self.software_versions.items():
            md += f"- **{name}**: {version}\n"

        return md
