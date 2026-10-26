"""
Streamlit UI: Rubric-Aware Project Quality Intelligence & Mentor Calibration Engine
Integrated with Active Learning, Live AST Parsing, and Rubric-Specific NLP Alignment.
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
from src.features.code_analyzer import extract_ast_metrics
from src.features.rubric_aligner import compute_rubric_alignment

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Rubric Quality Intelligence",
    page_icon="",
    layout="wide"
)

DATA_PATH = Path("data/raw/synthetic_submissions.csv")
FEEDBACK_LOG_PATH = Path("data/feedback/mentor_calibration_log.csv")

# Upgraded Feature Schema (P3: Replaced PCA embeddings with NLP Rubric Anchors)
FEATURE_COLS = [
    "code_complexity_score", ß
    "test_coverage_ratio", 
    "docstring_coverage",
    "lint_warning_count", 
    "rubric_arch_alignment", 
    "rubric_testing_alignment", 
    "rubric_eval_alignment"
]

RUBRIC_TARGETS = ["technical_correctness", "completeness_score"]

# ---------------------------------------------------------------------------
# Data & Model Caching Layer (Upgraded with P0 Active Learning)
# ---------------------------------------------------------------------------
@st.cache_data
def load_or_create_data() -> pd.DataFrame:
    """Loads dataset from disk. Requires updated generate_data.py with rubric alignment schema."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_dataset(n_samples=150)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)

@st.cache_resource(show_spinner="Retraining AI Models with Mentor Feedback...")
def train_rubric_models(df: pd.DataFrame):
    """
    Trains RandomForestRegressor models per rubric dimension on historical data.
    Dynamically injects mentor feedback logs with sample weighting (3.0 for overrides).
    """
    base_train = df[df["dataset_split"] == "historical_train"].copy()
    base_train["sample_weight"] = 1.0

    if FEEDBACK_LOG_PATH.exists():
        fb_df = pd.read_csv(FEEDBACK_LOG_PATH)
        fb_latest = fb_df.sort_values("timestamp_utc").drop_duplicates(subset=["submission_id"], keep="last")
        calibrated_subs = df[df["submission_id"].isin(fb_latest["submission_id"])].copy()
        
        calibrated_data = calibrated_subs.merge(
            fb_latest[["submission_id", "action", "final_technical_correctness", "final_completeness_score"]],
            on="submission_id", 
            how="inner"
        )
        calibrated_data["technical_correctness"] = calibrated_data["final_technical_correctness"]
        calibrated_data["completeness_score"] = calibrated_data["final_completeness_score"]
        calibrated_data["sample_weight"] = np.where(calibrated_data["action"] == "Overridden", 3.0, 1.5)
        
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

# Sidebar Controls & Active Learning Trigger
st.sidebar.title(" Calibration Controls")
st.sidebar.subheader(" Active Learning")

if st.sidebar.button(" Retrain AI on Mentor Logs", use_container_width=True, type="primary"):
    if FEEDBACK_LOG_PATH.exists():
        train_rubric_models.clear()
        st.toast("Model cache cleared. Rebuilding ensembles with mentor overrides!", icon="")
        st.rerun()
    else:
        st.sidebar.error("No mentor feedback logs exist yet. Calibrate a submission first.")

if train_metrics["is_retrained"]:
    st.sidebar.success(f"**Model actively trained** on {train_metrics['train_size']} samples (Includes weighted overrides).")
else:
    st.sidebar.info("Model trained on base historical data only.")

st.sidebar.divider()
uncertainty_threshold = st.sidebar.slider("Uncertainty Flag Threshold (σ)", 3.0, 12.0, 6.5, 0.5)

st.title("🎯 Rubric-Aware Project Quality Intelligence")
st.markdown("Automated multi-dimensional rubric scoring with **Active Learning**, **Live AST Parsing**, and **Rubric-Specific SHAP Explainability**.")

tab_queue, tab_live = st.tabs([" Batch Queue & Calibration", " Live Multimodal Predictor"])

# ---------------------------------------------------------------------------
# TAB 1: Batch Submission Queue & Calibration (Human-in-the-Loop)
# ---------------------------------------------------------------------------
with tab_queue:
    eval_df = df_all[df_all["dataset_split"] == "pending_evaluation"].copy().reset_index(drop=True)
    X_eval = eval_df[FEATURE_COLS]

    for target in RUBRIC_TARGETS:
        preds, stds = predict_with_uncertainty(models[target], X_eval)
        eval_df[f"pred_{target}"] = preds
        eval_df[f"std_{target}"] = stds

    eval_df["max_uncertainty"] = eval_df[["std_technical_correctness", "std_completeness_score"]].max(axis=1)

    def resolve_status(row):
        sub_id = row["submission_id"]
        if sub_id in st.session_state["mentor_reviews"]:
            return f"✅ Calibrated ({st.session_state['mentor_reviews'][sub_id]['action']})"
        if row["max_uncertainty"] >= uncertainty_threshold:
            return "⚠️ Flagged: High Uncertainty"
        return "🟢 High Confidence (Auto)"

    eval_df["review_status"] = eval_df.apply(resolve_status, axis=1)

    col_queue, col_inspector = st.columns([1.15, 1.35], gap="large")

    with col_queue:
        st.subheader("📋 Student Submission Queue")
        display_cols = ["submission_id", "pred_technical_correctness", "pred_completeness_score", "max_uncertainty", "review_status"]
        st.dataframe(eval_df[display_cols], use_container_width=True, hide_index=True, height=320)
        
        available_ids = eval_df["submission_id"].tolist()
        if not available_ids:
            st.stop()
            
        selected_sub_id = st.selectbox("🔍 Select Submission to Inspect:", options=available_ids)
        selected_row = eval_df[eval_df["submission_id"] == selected_sub_id].iloc[0]
        selected_pos_idx = int(eval_df.index[eval_df["submission_id"] == selected_sub_id][0])

        st.markdown("#### Extracted Submission Signals")
        sig1, sig2, sig3, sig4 = st.columns(4)
        sig1.metric("Cyclomatic Complexity", selected_row["code_complexity_score"])
        sig2.metric("Test Coverage", f"{selected_row['test_coverage_ratio'] * 100:.1f}%")
        sig3.metric("Docstring Coverage", f"{selected_row['docstring_coverage'] * 100:.1f}%")
        sig4.metric("Lint Warnings", int(selected_row["lint_warning_count"]))

        n1, n2, n3 = st.columns(3)
        n1.metric("Architecture Match", f"{selected_row['rubric_arch_alignment']*100:.1f}%")
        n2.metric("Testing Match", f"{selected_row['rubric_testing_alignment']*100:.1f}%")
        n3.metric("Eval Match", f"{selected_row['rubric_eval_alignment']*100:.1f}%")

    with col_inspector:
        st.subheader(f" AI Evaluation & Explainability: `{selected_sub_id}`")
        if selected_row["max_uncertainty"] >= uncertainty_threshold:
            st.warning(f"**High Uncertainty (±{selected_row['max_uncertainty']:.2f} pts).** Conflicting signals detected.")
        else:
            st.success(f"**High Model Confidence (±{selected_row['max_uncertainty']:.2f} pts).**")

        rubric_choice = st.radio("Explain Dimension:", options=["technical_correctness", "completeness_score"], horizontal=True, key="queue_radio")
        explainer = explainers[rubric_choice]
        shap_explanation = explainer(X_eval)
        
        fig_waterfall, _ = plt.subplots(figsize=(8, 4.0))
        shap.plots.waterfall(shap_explanation[selected_pos_idx], max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(fig_waterfall, clear_figure=True)

        st.markdown("#### Mentor Calibration Console")
        existing_review = st.session_state["mentor_reviews"].get(selected_sub_id, {})
        default_tech = float(existing_review.get("final_technical_correctness", selected_row["pred_technical_correctness"]))
        default_comp = float(existing_review.get("final_completeness_score", selected_row["pred_completeness_score"]))
        default_notes = existing_review.get("mentor_notes", "")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            calibrated_tech = st.slider("Technical Correctness", 0.0, 100.0, default_tech, 0.5, key=f"qt_{selected_sub_id}")
        with m_col2:
            calibrated_comp = st.slider("Completeness Score", 0.0, 100.0, default_comp, 0.5, key=f"qc_{selected_sub_id}")

        mentor_notes = st.text_area("Calibration Rationale (for Active Learning):", value=default_notes, key=f"notes_{selected_sub_id}")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("✅ Approve Predicted Score", use_container_width=True):
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "submission_id": selected_sub_id, "action": "Approved",
                    "pred_technical_correctness": selected_row["pred_technical_correctness"], "final_technical_correctness": selected_row["pred_technical_correctness"],
                    "pred_completeness_score": selected_row["pred_completeness_score"], "final_completeness_score": selected_row["pred_completeness_score"],
                    "uncertainty_std": selected_row["max_uncertainty"], "mentor_notes": mentor_notes or "Approved.",
                }
                st.session_state["mentor_reviews"][selected_sub_id] = record
                persist_mentor_decision(record)
                st.rerun()

        with btn_col2:
            if st.button(" Save Override", use_container_width=True, type="primary"):
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "submission_id": selected_sub_id, "action": "Overridden",
                    "pred_technical_correctness": selected_row["pred_technical_correctness"], "final_technical_correctness": calibrated_tech,
                    "pred_completeness_score": selected_row["pred_completeness_score"], "final_completeness_score": calibrated_comp,
                    "uncertainty_std": selected_row["max_uncertainty"], "mentor_notes": mentor_notes or "Manual override.",
                }
                st.session_state["mentor_reviews"][selected_sub_id] = record
                persist_mentor_decision(record)
                st.rerun()

# ---------------------------------------------------------------------------
# TAB 2: Live Multimodal Predictor (P1 Code Analyzer + P3 Rubric NLP)
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Upload raw Python code and project reports for live multimodal scoring.")
    
    col_up, col_res = st.columns([1.1, 1.4], gap="large")
    
    with col_up:
        uploaded_code = st.file_uploader("1. Upload Student Script (.py)", type=["py"])
        student_report_text = st.text_area(
            "2. Paste Student Project Report (Markdown/Text)",
            placeholder="Describe the architecture, testing approach, and evaluation metrics...",
            height=150
        )
        live_test_cov = st.slider("3. Declared Test Coverage (%)", min_value=0, max_value=100, value=75) / 100.0
        
    if uploaded_code is not None and student_report_text.strip():
        source_code = uploaded_code.read().decode("utf-8")
        
        # 1. Run Static AST Extraction
        ast_metrics = extract_ast_metrics(source_code)
        # 2. Run NLP Rubric Alignment
        nlp_metrics = compute_rubric_alignment(student_report_text)
        
        with col_up:
            st.markdown("####  Extracted Live Signals")
            l1, l2, l3 = st.columns(3)
            l1.metric("Cyclomatic Complexity", ast_metrics["code_complexity_score"])
            l2.metric("Docstring Coverage", f"{ast_metrics['docstring_coverage']*100:.0f}%")
            l3.metric("Lint Warnings", ast_metrics["lint_warning_count"])
            
            n1, n2, n3 = st.columns(3)
            n1.metric("Architecture Match", f"{nlp_metrics['rubric_arch_alignment']*100:.1f}%")
            n2.metric("Testing Match", f"{nlp_metrics['rubric_testing_alignment']*100:.1f}%")
            n3.metric("Eval Match", f"{nlp_metrics['rubric_eval_alignment']*100:.1f}%")

        # 3. Build live prediction payload
        live_df = pd.DataFrame([{
            "code_complexity_score": ast_metrics["code_complexity_score"],
            "test_coverage_ratio": live_test_cov,
            "docstring_coverage": ast_metrics["docstring_coverage"],
            "lint_warning_count": ast_metrics["lint_warning_count"],
            "rubric_arch_alignment": nlp_metrics["rubric_arch_alignment"],
            "rubric_testing_alignment": nlp_metrics["rubric_testing_alignment"],
            "rubric_eval_alignment": nlp_metrics["rubric_eval_alignment"],
        }])
        
        # 4. Predict & Explain
        with col_res:
            st.markdown("####  Live Prediction & Explainability")
            
            p_tech, std_tech = predict_with_uncertainty(models["technical_correctness"], live_df)
            p_comp, std_comp = predict_with_uncertainty(models["completeness_score"], live_df)
            
            m1, m2 = st.columns(2)
            m1.metric("Predicted Technical Correctness", f"{p_tech[0]} / 100", delta=f"±{std_tech[0]} σ", delta_color="off")
            m2.metric("Predicted Completeness Score", f"{p_comp[0]} / 100", delta=f"±{std_comp[0]} σ", delta_color="off")
            
            live_rubric = st.radio("Generate SHAP Breakdown For:", options=["technical_correctness", "completeness_score"], horizontal=True, key="live_radio")
            
            live_shap_exp = explainers[live_rubric](live_df)
            fig_live, _ = plt.subplots(figsize=(8, 4.0))
            shap.plots.waterfall(live_shap_exp[0], max_display=7, show=False)
            plt.tight_layout()
            st.pyplot(fig_live, clear_figure=True)

# Log Drawer
if FEEDBACK_LOG_PATH.exists():
    with st.expander(" View Persistent Mentor Calibration Audit Log"):
        st.dataframe(pd.read_csv(FEEDBACK_LOG_PATH), use_container_width=True)