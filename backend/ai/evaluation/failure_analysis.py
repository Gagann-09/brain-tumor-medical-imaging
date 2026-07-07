from pathlib import Path

import pandas as pd

from ai.training.callbacks import Callback
from ai.training.events import Event


class FailureAnalyzerCallback(Callback):
    priority = 6

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def on_evaluation_end(self, event: Event) -> None:
        self.results = event.data.get("all_case_metrics", [])
        self.analyze_and_export()

    def analyze_and_export(self):
        if not self.results:
            return

        df = pd.DataFrame(self.results)

        analysis = {}

        if "dice" in df.columns:
            analysis["lowest_dice"] = df.nsmallest(5, "dice").to_dict(orient="records")
            analysis["highest_dice"] = df.nlargest(5, "dice").to_dict(orient="records")

        if "hausdorff" in df.columns:
            analysis["highest_hausdorff"] = df.nlargest(5, "hausdorff").to_dict(orient="records")

        if "pred_sum" in df.columns and "gt_sum" in df.columns:
            fn_df = df[(df["pred_sum"] == 0) & (df["gt_sum"] > 0)]
            analysis["empty_predictions_fn"] = fn_df.to_dict(orient="records")

            fp_df = df[(df["pred_sum"] > 0) & (df["gt_sum"] == 0)]
            analysis["false_positives"] = fp_df.to_dict(orient="records")

            df["volume_diff"] = df["pred_sum"] - df["gt_sum"]
            analysis["largest_over_segmentation"] = df.nlargest(5, "volume_diff").to_dict(
                orient="records"
            )
            analysis["largest_under_segmentation"] = df.nsmallest(5, "volume_diff").to_dict(
                orient="records"
            )

        csv_path = self.output_dir / "failure_analysis.csv"
        df.to_csv(csv_path, index=False)

        md_path = self.output_dir / "failure_analysis.md"
        with open(md_path, "w") as f:
            f.write("# Automated Failure Analysis Report\n\n")
            for category, cases in analysis.items():
                f.write(f"## {category.replace('_', ' ').title()}\n")
                if not cases:
                    f.write("No cases found in this category.\n\n")
                    continue

                f.write("| Case ID | Metrics |\n")
                f.write("|---------|---------|\n")
                for c in cases:
                    c_id = c.pop("case_id", "unknown")
                    metrics_str = ", ".join(
                        [
                            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in c.items()
                        ]
                    )
                    f.write(f"| {c_id} | {metrics_str} |\n")
                f.write("\n")

