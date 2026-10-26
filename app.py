"""
Streamlit UI: Rubric-Aware Project Quality Intelligence & Mentor Calibration Engine
Integrated Upgrades:
- P0: Closed-Loop Active Learning (Sample-weighted retraining)
- P1: Live AST Static Code Analyzer for Python files
- P2: Conformal Prediction (MAPIE) for strict 90% confidence intervals
- P3: Explicit Rubric-Criterion NLP Alignment (Sentence-Transformers)
"""

import os
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from mapie.regression import MapieRegressor

# Custom Modules (Ensure these exist in your src/ and scripts/ directories)
from scripts.generate_data import generate_synthetic_dataset
from src.features.code_analyzer import extract_ast_metrics
from src.features.rubric_aligner import compute_rubric_alignment

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Rubric Quality Intelligence", page_icon="🎯", layout="wide")

DATA_PATH = Path("data/raw/synthetic_submissions.csv")
FEEDBACK_LOG_PATH = Path("data/feedback/mentor_calibration_log.csv")

# P3 Schema Upgrade: No more PCA embeddings, strictly rubric alignment scores
FEATURE_COLS = [
    "code_complexity_score", 
    "test_coverage_ratio", 
    "docstring_coverage",
    "lint_warning_count", 
    "rubric_arch_alignment",    
    "rubric_testing_alignment", 
    "rubric_eval_alignment"
]

RUBRIC_TARGETS = ["technical_correctness", "completeness_score"]

# ---------------------------------------------------------------------------
# Data & Model Engine
# ---------------------------------------------------------------------------
@st.cache_data
def load_or_create_data() -> pd.DataFrame:
    """Loads synthetic dataset, regenerating it if missing or using old schema."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_dataset(n_samples=150)
        df.to_csv(DATA_PATH, index=False)
    
    df = pd.read_csv(DATA_PATH)
    # Schema check fallback
    if "rubric_arch_alignment" not in df.columns:
        st.warning("⚠️ Old dataset schema detected. Regenerating data to match P3 Rubric NLP upgrades...")
        df = generate_synthetic_dataset(n_samples=150)
        df.to_csv(DATA_PATH, index=False)
    return df

@st.cache_resource(show_spinner="Training Conformal Predictors with Active Learning...")
def train_rubric_models(df: pd.DataFrame):
    """Trains Base Random Forests wrapped in MAPIE for Conformal Prediction bounds."""
    base_train = df[df["dataset_split"] == "historical_train"].copy()
    base_train["sample_weight"] = 1.0

    # P0: Active Learning Injection
    if FEEDBACK_LOG_PATH.exists():
        fb_df = pd.read_csv(FEEDBACK_LOG_PATH)
        fb_latest = fb_df.sort_values("timestamp_utc").drop_duplicates(subset=["submission_id"], keep="last")
        calibrated_subs = df[df["submission_id"].isin(fb_latest["submission_id"])].copy()
        
        calibrated_data = calibrated_subs.merge(
            fb_latest[["submission_id", "action", "final_technical_correctness", "final_completeness_score"]],
            on="submission_id", how="inner"
        )
        calibrated_data["technical_correctness"] = calibrated_data["final_technical_correctness"]
        calibrated_data["completeness_score"] = calibrated_data["final_completeness_score"]
        
        # Override weights: 3.0 for manual overrides, 1.5 for approvals
        calibrated_data["sample_weight"] = np.where(calibrated_data["action"] == "Overridden", 3.0, 1.5)
        combined_train = pd.concat([base_train, calibrated_data[base_train.columns]], ignore_index=True)
        is_retrained = True
    else:
        combined_train = base_train
        is_retrained = False

    X_train = combined_train[FEATURE_COLS]
    weights = combined_train["sample_weight"].to_numpy()
    
    models, explainers = {}, {}
    metrics = {"is_retrained": is_retrained, "train_size": len(combined_train)}

    for target in RUBRIC_TARGETS:
        y_train = combined_train[target]
        
        # P2: Conformal Calibration Split
        X_fit, X_calib, y_fit, y_calib, w_fit, w_calib = train_test_split(
            X_train, y_train, weights, test_size=0.2, random_state=42
        )
        
        # Train Base Estimator
        base_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        base_rf.fit(X_fit, y_fit, sample_weight=w_fit)
        
        # Wrap in MAPIE
        mapie_model = MapieRegressor(estimator=base_rf, cv="prefit", method="plus")
        mapie_model.fit(X_calib, y_calib)
        
        models[target] = mapie_model
        explainers[target] = shap.TreeExplainer(base_rf) # SHAP targets the base forest
        
        y_pred_calib = mapie_model.predict(X_calib)
        metrics[f"{target}_r2"] = r2_score(y_calib, y_pred_calib)

    return models, explainers, metrics

def predict_with_conformal(model: MapieRegressor, X: pd.DataFrame, alpha: float = 0.10):
    """Returns mean prediction alongside strict 90% confidence lower/upper bounds."""
    y_pred, y_pis = model.predict(X, alpha=alpha)
    lower_bound = np.clip(np.round(y_pis[:, 0, 0], 1), 0.0, 100.0)
    upper_bound = np.clip(np.round(y_pis[:, 1, 0], 1), 0.0, 100.0)
    interval_width = np.round(upper_bound - lower_bound, 1)
    return np.round(y_pred, 1), lower_bound, upper_bound, interval_width

def persist_mentor_decision(record: dict):
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([record])
    if FEEDBACK_LOG_PATH.exists():
        row_df.to_csv(FEEDBACK_LOG_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(FEEDBACK_LOG_PATH, mode="w", header=True, index=False)

if "mentor_reviews" not in st.session_state:
    st.session_state["mentor_reviews"] = {}

# ---------------------------------------------------------------------------
# Execution & Layout
# ---------------------------------------------------------------------------
df_all = load_or_create_data()
models, explainers, train_metrics = train_rubric_models(df_all)

# --- SIDEBAR ---
st.sidebar.title(" Calibration Controls")
st.sidebar.subheader(" Active Learning")
if st.sidebar.button(" Retrain AI on Mentor Logs", use_container_width=True, type="primary"):
    if FEEDBACK_LOG_PATH.exists():
        train_rubric_models.clear()
        st.toast("Model cache cleared. Rebuilding conformal ensembles!", icon="🚀")
        st.rerun()
    else:
        st.sidebar.error("No mentor feedback logs exist yet.")

if train_metrics["is_retrained"]:
    st.sidebar.success(f"**Model actively trained** on {train_metrics['train_size']} samples (Weighted).")

st.sidebar.divider()
interval_threshold = st.sidebar.slider("Max Allowed 90% Interval Width (pts)", 5.0, 30.0, 15.0, 1.0)

# --- MAIN HEADER ---
st.title("Rubric-Aware Project Quality Intelligence")
st.markdown("Powered by **Conformal Prediction**, **AST Code Analysis**, and **Rubric-Aligned NLP**.")
tab_queue, tab_live = st.tabs([" Batch Queue & Calibration", "🚀 Live Multimodal File Upload"])

# ---------------------------------------------------------------------------
# TAB 1: Batch Queue & Human-in-the-Loop Calibration
# ---------------------------------------------------------------------------
with tab_queue:
    eval_df = df_all[df_all["dataset_split"] == "pending_evaluation"].copy().reset_index(drop=True)
    X_eval = eval_df[FEATURE_COLS]

    for target in RUBRIC_TARGETS:
        preds, lowers, uppers, widths = predict_with_conformal(models[target], X_eval, alpha=0.10)
        eval_df[f"pred_{target}"] = preds
        eval_df[f"lower_{target}"] = lowers
        eval_df[f"upper_{target}"] = uppers
        eval_df[f"width_{target}"] = widths

    eval_df["max_uncertainty_width"] = eval_df[["width_technical_correctness", "width_completeness_score"]].max(axis=1)

    def resolve_status(row):
        sub_id = row["submission_id"]
        if sub_id in st.session_state["mentor_reviews"]:
            return f"✅ Calibrated ({st.session_state['mentor_reviews'][sub_id]['action']})"
        if row["max_uncertainty_width"] >= interval_threshold:
            return f"⚠️ Flagged (Interval > {interval_threshold} pts)"
        return "🟢 High Confidence (Auto)"

    eval_df["review_status"] = eval_df.apply(resolve_status, axis=1)
    
    col_queue, col_inspector = st.columns([1.15, 1.35], gap="large")
    with col_queue:
        st.dataframe(eval_df[["submission_id", "pred_technical_correctness", "pred_completeness_score", "max_uncertainty_width", "review_status"]], use_container_width=True, hide_index=True, height=320)
        selected_sub_id = st.selectbox(" Select Submission:", options=eval_df["submission_id"].tolist())
        selected_row = eval_df[eval_df["submission_id"] == selected_sub_id].iloc[0]
        
        st.markdown("#### Extracted Signals")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Complexity", selected_row["code_complexity_score"])
        s2.metric("Test Cov", f"{selected_row['test_coverage_ratio']*100:.0f}%")
        s3.metric("Docs Cov", f"{selected_row['docstring_coverage']*100:.0f}%")
        s4.metric("Lint Warns", int(selected_row["lint_warning_count"]))

    with col_inspector:
        st.subheader(f" AI Evaluation: `{selected_sub_id}`")
        if selected_row["max_uncertainty_width"] >= interval_threshold:
            st.warning(f"**High Uncertainty.** 90% confidence interval spans **{selected_row['max_uncertainty_width']:.1f} pts**. Review required.")
        else:
            st.success(f"**High Model Confidence.** 90% confidence interval spans only **{selected_row['max_uncertainty_width']:.1f} pts**.")
            
        m1, m2 = st.columns(2)
        m1.metric("Predicted Tech Score", f"{selected_row['pred_technical_correctness']} / 100", delta=f"90% CI: [{selected_row['lower_technical_correctness']}, {selected_row['upper_technical_correctness']}]", delta_color="off")
        m2.metric("Predicted Completeness", f"{selected_row['pred_completeness_score']} / 100", delta=f"90% CI: [{selected_row['lower_completeness_score']}, {selected_row['upper_completeness_score']}]", delta_color="off")
        
        rubric_choice = st.radio("Explain Dimension:", options=["technical_correctness", "completeness_score"], horizontal=True, key="queue_radio")
        explainer = explainers[rubric_choice]
        shap_explanation = explainer(X_eval)
        
        fig, _ = plt.subplots(figsize=(8, 3.5))
        shap.plots.waterfall(shap_explanation[int(eval_df.index[eval_df["submission_id"] == selected_sub_id][0])], max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.markdown("#### Mentor Console")
        mc1, mc2 = st.columns(2)
        with mc1:
            cal_tech = st.slider("Tech Correctness", 0.0, 100.0, float(selected_row['pred_technical_correctness']), key="qt")
        with mc2:
            cal_comp = st.slider("Completeness", 0.0, 100.0, float(selected_row['pred_completeness_score']), key="qc")
        
        if st.button("🛠️ Save Override", type="primary"):
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "submission_id": selected_sub_id, "action": "Overridden",
                "pred_technical_correctness": selected_row["pred_technical_correctness"], "final_technical_correctness": cal_tech,
                "pred_completeness_score": selected_row["pred_completeness_score"], "final_completeness_score": cal_comp,
                "uncertainty_std": selected_row["max_uncertainty_width"], "mentor_notes": "Live override."
            }
            st.session_state["mentor_reviews"][selected_sub_id] = record
            persist_mentor_decision(record)
            st.rerun()

# ---------------------------------------------------------------------------
# TAB 2: Live Multimodal File Upload
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Upload Python code and project reports for live multimodal scoring.")
    
    col_up, col_res = st.columns([1.1, 1.4], gap="large")
    
    with col_up:
        uploaded_code = st.file_uploader("1. Upload Student Script (.py)", type=["py"])
        student_report_text = st.text_area("2. Paste Student Project Report (Text)", placeholder="Describe architecture, testing, and evaluation...", height=130)
        live_test_cov = st.slider("3. Declared Test Coverage (%)", min_value=0, max_value=100, value=75) / 100.0
        
    if uploaded_code is not None and student_report_text.strip():
        source_code = uploaded_code.read().decode("utf-8")
        
        ast_metrics = extract_ast_metrics(source_code)
        nlp_metrics = compute_rubric_alignment(student_report_text)
        
        with col_up:
            st.markdown("#### Extracted Live Signals")
            l1, l2, l3 = st.columns(3)
            l1.metric("Cyclomatic Complexity", ast_metrics["code_complexity_score"])
            l2.metric("Docstring Coverage", f"{ast_metrics['docstring_coverage']*100:.0f}%")
            l3.metric("Lint Warnings", ast_metrics["lint_warning_count"])
            
            n1, n2, n3 = st.columns(3)
            n1.metric("Architecture Match", f"{nlp_metrics['rubric_arch_alignment']*100:.1f}%")
            n2.metric("Testing Match", f"{nlp_metrics['rubric_testing_alignment']*100:.1f}%")
            n3.metric("Eval Match", f"{nlp_metrics['rubric_eval_alignment']*100:.1f}%")

        live_df = pd.DataFrame([{
            "code_complexity_score": ast_metrics["code_complexity_score"],
            "test_coverage_ratio": live_test_cov,
            "docstring_coverage": ast_metrics["docstring_coverage"],
            "lint_warning_count": ast_metrics["lint_warning_count"],
            "rubric_arch_alignment": nlp_metrics["rubric_arch_alignment"],
            "rubric_testing_alignment": nlp_metrics["rubric_testing_alignment"],
            "rubric_eval_alignment": nlp_metrics["rubric_eval_alignment"],
        }])
        
        with col_res:
            st.markdown("#### Live Prediction & Explainability")
            p_tech, low_tech, up_tech, wid_tech = predict_with_conformal(models["technical_correctness"], live_df)
            p_comp, low_comp, up_comp, wid_comp = predict_with_conformal(models["completeness_score"], live_df)
            
            m1, m2 = st.columns(2)
            m1.metric("Predicted Tech Correctness", f"{p_tech[0]} / 100", delta=f"90% CI: [{low_tech[0]}, {up_tech[0]}]", delta_color="off")
            m2.metric("Predicted Completeness Score", f"{p_comp[0]} / 100", delta=f"90% CI: [{low_comp[0]}, {up_comp[0]}]", delta_color="off")
            
            live_rubric = st.radio("Generate SHAP Breakdown For:", options=["technical_correctness", "completeness_score"], horizontal=True, key="live_radio")
            
            live_shap_exp = explainers[live_rubric](live_df)
            fig_live, _ = plt.subplots(figsize=(8, 4.0))
            shap.plots.waterfall(live_shap_exp[0], max_display=7, show=False)
            plt.tight_layout()
            st.pyplot(fig_live, clear_figure=True)