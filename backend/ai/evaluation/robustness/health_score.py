class ExperimentHealthScoreCalculator:
    """Calculates the Experiment Health Score using weights and mandatory failure gates."""

    # Weights based on refinement
    WEIGHTS = {  # type: ignore  # type: ignore
        "performance": 0.30,
        "generalization": 0.25,
        "stability": 0.15,
        "calibration": 0.15,
        "reproducibility": 0.10,
        "data_quality": 0.05,
    }

    @staticmethod
    def calculate(
        performance_score: float,  # 0-100
        generalization_score: float,  # 0-100
        stability_score: float,  # 0-100
        calibration_score: float,  # 0-100
        reproducibility_score: float,  # 0-100
        data_quality_score: float,  # 0-100
        has_leakage: bool = False,
        has_nan_inf: bool = False,
        excessive_gap: bool = False,
        failed_cv: bool = False,
        failed_calibration: bool = False,
    ) -> float:
        """
        Calculates the health score.
        If any mandatory failure gate is triggered, the score is penalized to 0.
        """
        if has_leakage or has_nan_inf or excessive_gap or failed_cv or failed_calibration:
            return 0.0

        score = (
            performance_score * ExperimentHealthScoreCalculator.WEIGHTS["performance"]
            + generalization_score * ExperimentHealthScoreCalculator.WEIGHTS["generalization"]
            + stability_score * ExperimentHealthScoreCalculator.WEIGHTS["stability"]
            + calibration_score * ExperimentHealthScoreCalculator.WEIGHTS["calibration"]
            + reproducibility_score * ExperimentHealthScoreCalculator.WEIGHTS["reproducibility"]
            + data_quality_score * ExperimentHealthScoreCalculator.WEIGHTS["data_quality"]
        )
        return score
