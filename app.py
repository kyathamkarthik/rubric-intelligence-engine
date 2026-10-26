"""
Streamlit UI: Rubric-Aware Project Quality Intelligence & Mentor Calibration Engine
(Now with Closed-Loop Active Learning & Sample-Weighted Retraining)
"""

from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from scripts.generate_data import generate_synthetic_dataset

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Rubric Quality Intelligence & Mentor Calibration",
    page_icon="🎯",
    layout="wide",
)

DATA_PATH = Path("data/raw/synthetic_submissions.csv")
FEEDBACK_LOG_PATH = Path("data/feedback/mentor_calibration_log.csv")

FEATURE_COLS = [
    "code_complexity_score",
    "test_coverage_ratio",
    "docstring_coverage",
    "lint_warning_count",
    "submission_text_embedding_0",
    "submission_text_embedding_1",
    "submission_text_embedding_2",
]

RUBRIC_TARGETS = ["technical_correctness", "completeness_score"]


# ---------------------------------------------------------------------------
# Data & Model Caching Layer (Upgraded with Active Learning)
# ---------------------------------------------------------------------------
@st.cache_data
def load_or_create_data() -> pd.DataFrame:
    """Loads synthetic dataset from disk or generates it automatically if missing."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_dataset(n_samples=150)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Retraining AI Models with Mentor Feedback...")
def train_rubric_models(df: pd.DataFrame):
    """
    Trains RandomForestRegressor models per rubric dimension on historical data.
    Dynamically injects mentor feedback logs with sample weighting (3.0 for overrides, 1.5 for approvals).
    """
    base_train = df[df["dataset_split"] == "historical_train"].copy()
    base_train["sample_weight"] = 1.0

    # Active Learning: Merge feedback log if it exists
    if FEEDBACK_LOG_PATH.exists():
        fb_df = pd.read_csv(FEEDBACK_LOG_PATH)
        fb_latest = fb_df.sort_values("timestamp_utc").drop_duplicates(subset=["submission_id"], keep="last")
        
        calibrated_subs = df[df["submission_id"].isin(fb_latest["submission_id"])].copy()
        
        calibrated_data = calibrated_subs.merge(
            fb_latest[["submission_id", "action", "final_technical_correctness", "final_completeness_score"]],
            on="submission_id",
            how="inner"
        )
        
        # Override AI scores with Mentor scores
        calibrated_data["technical_correctness"] = calibrated_data["final_technical_correctness"]
        calibrated_data["completeness_score"] = calibrated_data["final_completeness_score"]
        
        # Sample Weighting: Prioritize manual overrides heavily
        calibrated_data["sample_weight"] = np.where(calibrated_data["action"] == "Overridden", 3.0, 1.5)
        
        # Append calibrated data to the training pool
        combined_train = pd.concat([base_train, calibrated_data[base_train.columns]], ignore_index=True)
        is_retrained = True
    else:
        combined_train = base_train
        is_retrained = False

    X_train = combined_train[FEATURE_COLS]
    weights = combined_train["sample_weight"].to_numpy()

    models = {}
    explainers = {}
    metrics = {"is_retrained": is_retrained, "train_size": len(combined_train)}

    for target in RUBRIC_TARGETS:
        y_train = combined_train[target]
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=weights)
        
        models[target] = model
        explainers[target] = shap.TreeExplainer(model)
        metrics[f"{target}_r2"] = r2_score(y_train, model.predict(X_train), sample_weight=weights)

    return models, explainers, metrics


def predict_with_uncertainty(model: RandomForestRegressor, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Computes mean prediction and epistemic uncertainty (std across trees)."""
    X_array = X.to_numpy()
    tree_predictions = np.array([tree.predict(X_array) for tree in model.estimators_])
    mean_pred = np.round(np.mean(tree_predictions, axis=0), 1)
    std_uncertainty = np.round(np.std(tree_predictions, axis=0), 2)
    return mean_pred, std_uncertainty


def persist_mentor_decision(record: dict):
    """Appends mentor review decisions to a local audit CSV."""
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([record])
    if FEEDBACK_LOG_PATH.exists():
        row_df.to_csv(FEEDBACK_LOG_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(FEEDBACK_LOG_PATH, mode="w", header=True, index=False)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "mentor_reviews" not in st.session_state:
    st.session_state["mentor_reviews"] = {}


# ---------------------------------------------------------------------------
# Main Application Execution
# ---------------------------------------------------------------------------
df_all = load_or_create_data()
models, explainers, train_metrics = train_rubric_models(df_all)

eval_df = df_all[df_all["dataset_split"] == "pending_evaluation"].copy().reset_index(drop=True)
X_eval = eval_df[FEATURE_COLS]

for target in RUBRIC_TARGETS:
    preds, stds = predict_with_uncertainty(models[target], X_eval)
    eval_df[f"pred_{target}"] = preds
    eval_df[f"std_{target}"] = stds

eval_df["max_uncertainty"] = eval_df[["std_technical_correctness", "std_completeness_score"]].max(axis=1)

# ---------------------------------------------------------------------------
# Sidebar Controls & Active Learning Trigger
# ---------------------------------------------------------------------------
st.sidebar.title(" Calibration Controls")

# ACTIVE LEARNING RETRAIN BUTTON
st.sidebar.subheader(" Active Learning")
if st.sidebar.button(" Retrain AI on Mentor Logs", use_container_width=True, type="primary"):
    if FEEDBACK_LOG_PATH.exists():
        train_rubric_models.clear()  # Flush the model cache to force retraining
        st.toast("Model cache cleared. Rebuilding ensembles with mentor overrides!", icon="🚀")
        st.rerun()
    else:
        st.sidebar.error("No mentor feedback logs exist yet. Calibrate a submission first.")

if train_metrics["is_retrained"]:
    st.sidebar.success(f"**Model actively trained** on {train_metrics['train_size']} samples (Includes weighted overrides).")
else:
    st.sidebar.info("Model trained on base historical data only.")

st.sidebar.divider()

uncertainty_threshold = st.sidebar.slider(
    "Uncertainty Flag Threshold (σ)",
    min_value=3.0, max_value=12.0, value=6.5, step=0.5,
)

def resolve_status(row):
    sub_id = row["submission_id"]
    if sub_id in st.session_state["mentor_reviews"]:
        action = st.session_state["mentor_reviews"][sub_id]["action"]
        return f"✅ Calibrated ({action})"
    if row["max_uncertainty"] >= uncertainty_threshold:
        return "⚠️ Flagged: High Uncertainty"
    return "🟢 High Confidence (Auto)"

eval_df["review_status"] = eval_df.apply(resolve_status, axis=1)

filter_mode = st.sidebar.radio(
    "Filter Submission Queue",
    options=["All Pending Submissions", "⚠️ Flagged for Mentor Review Only", "✅ Calibrated by Mentor"],
)

if filter_mode == "⚠️ Flagged for Mentor Review Only":
    filtered_df = eval_df[eval_df["review_status"].str.contains("Flagged")].copy()
elif filter_mode == "✅ Calibrated by Mentor":
    filtered_df = eval_df[eval_df["review_status"].str.contains("Calibrated")].copy()
else:
    filtered_df = eval_df.copy()

# ---------------------------------------------------------------------------
# Header & Top-Level KPI Metrics
# ---------------------------------------------------------------------------
st.title("🎯 Rubric-Aware Project Quality Intelligence")
st.markdown("Automated rubric scoring with **Active Learning** sample weighting & **SHAP Explainability**.")

total_pending = len(eval_df)
flagged_count = int((eval_df["max_uncertainty"] >= uncertainty_threshold).sum())
calibrated_count = len(st.session_state["mentor_reviews"])
avg_uncertainty = eval_df["max_uncertainty"].mean()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active Submission Queue", total_pending)
kpi2.metric("⚠️ Flagged for Human Review", flagged_count, delta=f"σ ≥ {uncertainty_threshold}", delta_color="inverse")
kpi3.metric("✅ Mentor Calibrated", f"{calibrated_count} / {total_pending}")
kpi4.metric("Mean Ensemble Uncertainty (σ)", f"±{avg_uncertainty:.2f} pts")

st.divider()

# ---------------------------------------------------------------------------
# Submission Queue Table & Selection
# ---------------------------------------------------------------------------
col_queue, col_inspector = st.columns([1.15, 1.35], gap="large")

with col_queue:
    st.subheader(" Student Submission Queue")
    display_cols = ["submission_id", "student_id", "pred_technical_correctness", "pred_completeness_score", "max_uncertainty", "review_status"]
    st.dataframe(
        filtered_df[display_cols].rename(
            columns={
                "submission_id": "Submission ID", "student_id": "Student",
                "pred_technical_correctness": "Tech Score (Pred)", "pred_completeness_score": "Completeness (Pred)",
                "max_uncertainty": "Uncertainty (±σ)", "review_status": "Calibration Status"
            }
        ),
        use_container_width=True, hide_index=True, height=320,
    )

    available_ids = filtered_df["submission_id"].tolist()
    if not available_ids:
        st.info("No submissions match the current filter criteria.")
        st.stop()

    selected_sub_id = st.selectbox("🔍 Select Submission to Inspect & Calibrate:", options=available_ids)
    selected_row = eval_df[eval_df["submission_id"] == selected_sub_id].iloc[0]
    selected_pos_idx = int(eval_df.index[eval_df["submission_id"] == selected_sub_id][0])

    st.markdown("####  Extracted Submission Signals")
    sig1, sig2, sig3, sig4 = st.columns(4)
    sig1.metric("Cyclomatic Complexity", selected_row["code_complexity_score"])
    sig2.metric("Test Coverage", f"{selected_row['test_coverage_ratio'] * 100:.1f}%")
    sig3.metric("Docstring Coverage", f"{selected_row['docstring_coverage'] * 100:.1f}%")
    sig4.metric("Lint Warnings", int(selected_row["lint_warning_count"]))
    
    st.caption("**Student Project Report Excerpt:**")
    st.info(f"\"{selected_row['report_summary']}\"")

# ---------------------------------------------------------------------------
# Right Column: Human-in-the-Loop (HITL) Calibration & SHAP Explainability
# ---------------------------------------------------------------------------
with col_inspector:
    st.subheader(f" AI Evaluation & SHAP Explainability: `{selected_sub_id}`")

    if selected_row["max_uncertainty"] >= uncertainty_threshold:
        st.warning(f"**High Uncertainty (±{selected_row['max_uncertainty']:.2f} pts).** Conflicting signals detected. Review required.")
    else:
        st.success(f"**High Model Confidence (±{selected_row['max_uncertainty']:.2f} pts).**")

    # SHAP Explanation Tabs
    tab_local, tab_global = st.tabs([" Local SHAP Waterfall", " Global Importance"])

    with tab_local:
        rubric_choice = st.radio("Explain Dimension:", options=["technical_correctness", "completeness_score"], horizontal=True)
        explainer = explainers[rubric_choice]
        shap_explanation = explainer(X_eval)
        
        fig_waterfall, _ = plt.subplots(figsize=(8, 4.2))
        shap.plots.waterfall(shap_explanation[selected_pos_idx], max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf(), clear_figure=True)

    with tab_global:
        fig_bar, _ = plt.subplots(figsize=(8, 4.0))
        shap.plots.bar(explainer(X_eval), max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf(), clear_figure=True)

    st.divider()

    # -----------------------------------------------------------------------
    # Mentor Override Console
    # -----------------------------------------------------------------------
    st.markdown("#### Mentor Calibration Console")
    
    existing_review = st.session_state["mentor_reviews"].get(selected_sub_id, {})
    default_tech = float(existing_review.get("final_technical_correctness", selected_row["pred_technical_correctness"]))
    default_comp = float(existing_review.get("final_completeness_score", selected_row["pred_completeness_score"]))
    default_notes = existing_review.get("mentor_notes", "")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        calibrated_tech = st.slider(
            f"Tech Correctness (Pred: {selected_row['pred_technical_correctness']} ± {selected_row['std_technical_correctness']})",
            min_value=0.0, max_value=100.0, value=default_tech, step=0.5,
        )
    with m_col2:
        calibrated_comp = st.slider(
            f"Completeness (Pred: {selected_row['pred_completeness_score']} ± {selected_row['std_completeness_score']})",
            min_value=0.0, max_value=100.0, value=default_comp, step=0.5,
        )

    mentor_notes = st.text_area("Calibration Notes (fed into Active Learning log):", value=default_notes, key=f"notes_{selected_sub_id}")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("✅ Approve AI Predicted Score", use_container_width=True):
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "submission_id": selected_sub_id, "action": "Approved",
                "pred_technical_correctness": selected_row["pred_technical_correctness"], "final_technical_correctness": selected_row["pred_technical_correctness"],
                "pred_completeness_score": selected_row["pred_completeness_score"], "final_completeness_score": selected_row["pred_completeness_score"],
                "uncertainty_std": selected_row["max_uncertainty"], "mentor_notes": mentor_notes or "Approved.",
            }
            st.session_state["mentor_reviews"][selected_sub_id] = record
            persist_mentor_decision(record)
            st.toast(f"Approved {selected_sub_id}! Hit 'Retrain' in the sidebar to update models.", icon="✅")
            st.rerun()

    with btn_col2:
        if st.button("🛠️ Override & Save Calibrated Score", use_container_width=True, type="primary"):
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "submission_id": selected_sub_id, "action": "Overridden",
                "pred_technical_correctness": selected_row["pred_technical_correctness"], "final_technical_correctness": calibrated_tech,
                "pred_completeness_score": selected_row["pred_completeness_score"], "final_completeness_score": calibrated_comp,
                "uncertainty_std": selected_row["max_uncertainty"], "mentor_notes": mentor_notes or "Manual override.",
            }
            st.session_state["mentor_reviews"][selected_sub_id] = record
            persist_mentor_decision(record)
            st.toast(f"Saved override for {selected_sub_id}! Hit 'Retrain' in the sidebar to update models.", icon="🛠️")
            st.rerun()

# Log Drawer
if FEEDBACK_LOG_PATH.exists():
    with st.expander(" View Persistent Mentor Calibration Audit Log"):
        st.dataframe(pd.read_csv(FEEDBACK_LOG_PATH), use_container_width=True)