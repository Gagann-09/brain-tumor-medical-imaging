import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkRecord(BaseModel):
    benchmark_id: str
    models_compared: list[str]
    dataset: str
    metrics: dict[str, dict[str, float]]
    hardware: str
    configuration: dict[str, Any]
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    outcome: str


class BenchmarkRegistry:
    """Registry to store and query model comparison benchmarks."""

    def __init__(self, registry_dir: str):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "benchmarks.json"
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
                self.records = [BenchmarkRecord(**r) for r in data]
        else:
            self.records = []

    def _save_registry(self) -> None:
        with open(self.registry_file, "w") as f:
            json.dump([r.model_dump(mode="json") for r in self.records], f, indent=4)

    def add_benchmark(self, record: BenchmarkRecord) -> None:
        self.records.append(record)
        self._save_registry()

    def get_benchmarks(self) -> list[BenchmarkRecord]:
        return self.records
