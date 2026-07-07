from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai.xai.shared.report import ExplainabilityReport


@dataclass
class ExplainabilitySession:
    """Groups all artifacts and metadata from a single explainability run."""

    session_id: str
    original_input: Any
    prediction_result: dict[str, Any]
    artifacts: dict[str, Any]  # In-memory arrays or other representations
    report: ExplainabilityReport

    def export(self, output_dir: str):
        """Exports the session's report to the specified directory."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        with open(path / f"{self.session_id}_report.json", "w") as f:
            f.write(self.report.to_json())

        # Save Markdown report
        with open(path / f"{self.session_id}_report.md", "w") as f:
            f.write(self.report.to_markdown())
