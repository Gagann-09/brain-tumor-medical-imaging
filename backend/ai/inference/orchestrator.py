import time
from typing import Dict, List, Type
from ai.training.events import EventBus, Event, EventType
from ai.inference.context import InferenceContext, InferenceResult, StageStatus, StageDiagnostics
from ai.inference.manifest import PipelineManifest
from ai.inference.policy import InferencePolicy, ExecutionMode
from ai.inference.registry import ArtifactRegistry
from ai.inference.stages.base import BaseStage
from ai.pipeline.profiler import ExperimentProfiler

class InferenceOrchestrator:
    def __init__(
        self, 
        event_bus: EventBus, 
        artifact_registry: ArtifactRegistry,
        policy: InferencePolicy,
        manifest: PipelineManifest,
        stage_classes: Dict[str, Type[BaseStage]],
        profiler: ExperimentProfiler,
        progress_callback = None
    ):
        self.event_bus = event_bus
        self.artifact_registry = artifact_registry
        self.policy = policy
        self.manifest = manifest
        self.stages = [stage_classes[name](name=name) for name in manifest.stages if name in stage_classes]
        self.profiler = profiler
        self.progress_callback = progress_callback

    def run(self, initial_context: InferenceContext) -> InferenceResult:
        self.event_bus.publish(Event(EventType.INFERENCE_STARTED))
        self.profiler.start_epoch(0) # Using existing profiler API creatively
        
        context = initial_context
        diagnostics_map: Dict[str, StageDiagnostics] = {}
        pipeline_status = "success"
        start_time = time.time()
        
        total_stages = len(self.stages)
        for i, stage in enumerate(self.stages):
            if self.progress_callback:
                self.progress_callback(stage.name, int((i / total_stages) * 100))
                
            self.event_bus.publish(Event(EventType.STAGE_STARTED, {"stage": stage.name}))
            
            context, stage_diag = stage.run(context)
            diagnostics_map[stage.name] = stage_diag
            
            self.event_bus.publish(Event(EventType.STAGE_FINISHED, {"stage": stage.name, "status": stage_diag.status.value}))
            
            if stage_diag.status == StageStatus.FAILED:
                pipeline_status = "failed"
                break
        
        total_latency_ms = (time.time() - start_time) * 1000
        
        if pipeline_status == "failed":
            self.event_bus.publish(Event(EventType.INFERENCE_FAILED))
        else:
            self.event_bus.publish(Event(EventType.INFERENCE_FINISHED))
            
        return InferenceResult(
            status=pipeline_status,
            diagnostics=diagnostics_map,
            segmentation=context.segmentation_results,
            classification=context.classification_results,
            explainability=context.explainability_artifacts,
            report=context.generated_reports,
            total_latency_ms=total_latency_ms
        )
