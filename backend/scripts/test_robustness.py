import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.evaluation.robustness.health_score import ExperimentHealthScoreCalculator
from ai.evaluation.robustness.calculators import GeneralizationGapCalculator, DatasetLeakageDetector
from ai.evaluation.robustness.analyzers import LearningCurveAnalyzer
from ai.models.promotion import ModelPromotionManager
from ai.pipeline.profiler import ExperimentProfiler

def test_robustness():
    print("Testing Dataset Leakage Detector...")
    has_leakage = DatasetLeakageDetector.static_validation(set(["pt1", "pt2"]), set(["pt2", "pt3"]))
    assert has_leakage == True
    
    print("Testing Learning Curve Analyzer...")
    train_hist = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]
    val_hist = [1.0, 0.85, 0.7, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
    curve_analysis = LearningCurveAnalyzer.analyze_curve(train_hist, val_hist)
    assert curve_analysis["divergence"] == True
    
    print("Testing Health Score Calculator...")
    score = ExperimentHealthScoreCalculator.calculate(
        performance_score=90,
        generalization_score=80,
        stability_score=70,
        calibration_score=85,
        reproducibility_score=100,
        data_quality_score=90,
        has_leakage=False,
        failed_cv=False
    )
    print(f"Health Score: {score}")
    assert score > 0
    
    score_with_leakage = ExperimentHealthScoreCalculator.calculate(
        performance_score=90, generalization_score=80, stability_score=70, 
        calibration_score=85, reproducibility_score=100, data_quality_score=90,
        has_leakage=True
    )
    assert score_with_leakage == 0.0
    
    print("Testing Profiler Report Generation...")
    profiler = ExperimentProfiler(output_dir="test_robustness_output")
    report = profiler.generate_generalization_report(
        experiment_id="exp_001",
        train_history=train_hist,
        val_history=val_hist,
        has_leakage=False,
        failed_cv=False,
        failed_calibration=False,
        performance_score=90,
        generalization_score=40,
        stability_score=50,
        calibration_score=80,
        reproducibility_score=100,
        data_quality_score=90,
        excessive_gap=True,
        has_nan_inf=False
    )
    print(f"Report recommendation: {report.promotion_recommendation}")
    
    print("Testing Model Promotion Manager...")
    pm = ModelPromotionManager(registry_dir="test_robustness_output/registry")
    pm.register_model("model_A", "v1")
    pm.mark_as_candidate("model_A", "v1")
    
    promoted = pm.evaluate_promotion_robust("model_A", "v1", score_with_leakage, ["Excessive Generalization Gap"], "bench_1")
    assert promoted == False
    print("Promotion correctly rejected due to critical failures.")

    print("Robustness Framework Tests Passed!")

if __name__ == "__main__":
    test_robustness()
