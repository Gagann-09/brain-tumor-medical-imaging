import json
from pathlib import Path


class ArtifactRegistry:
    """Automatically records all generated artifacts from every inference stage."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts: dict[str, str] = {}

    def record_artifact(self, name: str, path: str):
        """Records the path of a generated artifact."""
        self.artifacts[name] = str(path)

    def save_manifest(self, filename: str = "artifact_manifest.json"):
        """Saves the artifact manifest to the output directory."""
        with open(self.output_dir / filename, "w") as f:
            json.dump(self.artifacts, f, indent=4)
