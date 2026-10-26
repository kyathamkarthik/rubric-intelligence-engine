"""
Active Learning Retraining Pipeline with Sample Weighting on Mentor Overrides.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLS = [
    "code_complexity_score",
    "test_coverage_ratio",
    "docstring_coverage",
    "lint_warning_count",
    "submission_text_embedding_0",
    "submission_text_embedding_1",
    "submission_text_embedding_2",
]


def retrain_with_mentor_feedback(
    raw_data_path: Path,
    feedback_log_path: Path,
    model_dir: Path = Path("models/checkpoints"),
) -> dict:
    """
    Merges human-calibrated scores with historical training data and retrains
    rubric models using sample weighting to correct systematic model errors.
    """
    df_all = pd.read_csv(raw_data_path)
    base_train = df_all[df_all["dataset_split"] == "historical_train"].copy()
    base_train["sample_weight"] = 1.0

    if feedback_log_path.exists():
        fb_df = pd.read_csv(feedback_log_path)
        # Keep most recent calibration per submission_id
        fb_latest = fb_df.sort_values("timestamp_utc").drop_duplicates(
            subset=["submission_id"], keep="last"
        )

        # Join feature columns from the original submission pool
        calibrated = df_all.merge(
            fb_latest[
                [
                    "submission_id",
                    "action",
                    "final_technical_correctness",
                    "final_completeness_score",
                ]
            ],
            on="submission_id",
            how="inner",
        )
        calibrated["technical_correctness"] = calibrated["final_technical_correctness"]
        calibrated["completeness_score"] = calibrated["final_completeness_score"]

        # Weight human overrides 3x higher than baseline historical data
        calibrated["sample_weight"] = np.where(
            calibrated["action"] == "Overridden", 3.0, 1.5
        )

        combined_df = pd.concat(
            [base_train, calibrated[base_train.columns]], ignore_index=True
        )
    else:
        combined_df = base_train

    model_dir.mkdir(parents=True, exist_ok=True)
    X_train = combined_df[FEATURE_COLS]
    weights = combined_df["sample_weight"].to_numpy()

    metrics_summary = {"total_training_rows": len(combined_df)}

    for target in ["technical_correctness", "completeness_score"]:
        y_train = combined_df[target]
        model = RandomForestRegressor(
            n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train, sample_weight=weights)
        joblib.dump(model, model_dir / f"rf_{target}.joblib")
        metrics_summary[f"{target}_r2_in_sample"] = round(
            model.score(X_train, y_train, sample_weight=weights), 4
        )

    return metrics_summary