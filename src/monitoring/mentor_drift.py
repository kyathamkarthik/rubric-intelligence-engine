"""
Mentor Drift & Inter-Rater Reliability Monitor.
Tracks grading offsets (harshness bias) and Intraclass Correlation Coefficient (ICC).
"""

import pandas as pd
import pingouin as pg

def calculate_mentor_offsets(feedback_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the mean grading bias per mentor relative to the AI's initial prediction.
    Negative values indicate a harsher mentor; positive indicate a lenient mentor.
    """
    if feedback_df.empty or "mentor_id" not in feedback_df.columns:
        return pd.DataFrame()

    df = feedback_df.copy()
    
    # Calculate delta: (Human Score - AI Score)
    df["tech_delta"] = df["final_technical_correctness"] - df["pred_technical_correctness"]
    df["comp_delta"] = df["final_completeness_score"] - df["pred_completeness_score"]
    
    # Aggregate by mentor
    drift_summary = df.groupby("mentor_id").agg(
        reviews_completed=("submission_id", "count"),
        mean_tech_offset=("tech_delta", "mean"),
        mean_comp_offset=("comp_delta", "mean"),
    ).reset_index()
    
    # Round to 1 decimal place
    drift_summary["mean_tech_offset"] = drift_summary["mean_tech_offset"].round(1)
    drift_summary["mean_comp_offset"] = drift_summary["mean_comp_offset"].round(1)
    
    return drift_summary


def calculate_golden_icc(golden_scores_df: pd.DataFrame, score_column: str = "score") -> dict:
    """
    Computes ICC (Intraclass Correlation Coefficient) for Golden Anchor submissions.
    Requires a 'long' format DataFrame with: [submission_id, mentor_id, score]
    
    Returns ICC2 (Two-way random effects, absolute agreement) which evaluates both 
    correlation and systematic offsets between raters.
    """
    if golden_scores_df.empty or golden_scores_df["mentor_id"].nunique() < 2:
        return {"icc_score": None, "interpretation": "Need at least 2 mentors for ICC."}

    # Calculate ICC using pingouin
    icc_results = pg.intraclass_corr(
        data=golden_scores_df, 
        targets="submission_id", 
        raters="mentor_id", 
        ratings=score_column
    )
    
    # Extract ICC2 (Single fixed raters, absolute agreement)
    icc2_row = icc_results[icc_results["Type"] == "ICC2"].iloc[0]
    icc_val = round(icc2_row["ICC"], 3)
    
    # Standard interpretation bands
    if icc_val >= 0.90:
        interpretation = "Excellent (Highly Aligned)"
    elif icc_val >= 0.75:
        interpretation = "Good (Minor Drift)"
    elif icc_val >= 0.50:
        interpretation = "Moderate (High Drift - Calibration Needed)"
    else:
        interpretation = "Poor (Severe Disagreement)"
        
    return {
        "icc_score": icc_val,
        "ci_95": f"[{round(icc2_row['CI95%'][0], 2)}, {round(icc2_row['CI95%'][1], 2)}]",
        "interpretation": interpretation
    }