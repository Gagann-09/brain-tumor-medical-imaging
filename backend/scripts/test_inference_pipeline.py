import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.inference.context import InferenceContext, StageStatus
from ai.inference.manifest import PipelineManifest
from ai.inference.orchestrator import InferenceOrchestrator
from ai.inference.policy import ExecutionMode, InferencePolicy
from ai.inference.registry import ArtifactRegistry
from ai.inference.stages import STAGE_CLASSES
from ai.pipeline.profiler import ExperimentProfiler
from ai.training.events import Event, EventBus, EventType


def test_inference_pipeline():
    print("Setting up Inference Pipeline...")
    event_bus = EventBus()

    events_recorded = []

    def record_event(event: Event):
        events_recorded.append((event.type, event.data))

    for evt_type in EventType:
        event_bus.subscribe(evt_type, record_event)

    artifact_registry = ArtifactRegistry(output_dir="test_inference_output/artifacts")
    policy = InferencePolicy.get_default_policy(ExecutionMode.CLINICAL_REVIEW)
    manifest = PipelineManifest()
    profiler = ExperimentProfiler(output_dir="test_inference_output/profiler")

    orchestrator = InferenceOrchestrator(
        event_bus=event_bus,
        artifact_registry=artifact_registry,
        policy=policy,
        manifest=manifest,
        stage_classes=STAGE_CLASSES,
        profiler=profiler,
    )

    # 1. Test Successful Run
    print("Running Successful Pipeline...")
    initial_context = InferenceContext(
        study_metadata={
            "patient_id": "PT001",
            "study_id": "ST001",
            "segmentation_model": "v1",
            "classification_model": "v1",
        }
    )

    result = orchestrator.run(initial_context)

    assert result.status == "success"
    assert "InputValidationStage" in result.diagnostics
    assert result.diagnostics["InputValidationStage"].status == StageStatus.SUCCESS
    assert result.segmentation is not None
    assert result.classification is not None
    assert result.report is not None

    # Ensure events were fired
    event_types = [e[0] for e in events_recorded]
    assert EventType.INFERENCE_STARTED in event_types
    assert EventType.STAGE_STARTED in event_types
    assert EventType.STAGE_FINISHED in event_types
    assert EventType.INFERENCE_FINISHED in event_types

    # 2. Test Failed Run (Validation Failure)
    print("Running Failed Pipeline (Metadata missing)...")
    events_recorded.clear()

    initial_context_fail = InferenceContext(
        study_metadata={}
    )  # Missing metadata should fail validation
    result_fail = orchestrator.run(initial_context_fail)

    assert result_fail.status == "failed"
    assert result_fail.diagnostics["InputValidationStage"].status == StageStatus.FAILED
    assert "SegmentationStage" not in result_fail.diagnostics  # Should short-circuit

    event_types_fail = [e[0] for e in events_recorded]
    assert EventType.INFERENCE_FAILED in event_types_fail

    print("Inference Pipeline Tests Passed!")


if __name__ == "__main__":
    test_inference_pipeline()
