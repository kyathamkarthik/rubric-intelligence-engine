"""
Streamlit UI: Rubric-Aware Project Quality Intelligence & Mentor Calibration Engine
(Includes Active Learning, Conformal Prediction, AST Parsing, NLP Rubrics, and Mentor Drift)
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from mapie.regression import MapieRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from scripts.generate_data import generate_synthetic_dataset
from src.features.code_analyzer import extract_ast_metrics
from src.features.rubric_aligner import compute_rubric_alignment
from src.monitoring.mentor_drift import calculate_mentor_offsets

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Rubric Quality Intelligence",
    page_icon="",
    layout="wide",
)

DATA_PATH = Path("data/raw/synthetic_submissions.csv")
FEEDBACK_LOG_PATH = Path("data/feedback/mentor_calibration_log.csv")

# Updated Feature Schema mapping to the P3 NLP Upgrade
FEATURE_COLS = [
    "code_complexity_score",
    "test_coverage_ratio",
    "docstring_coverage",
    "lint_warning_count",
    "rubric_arch_alignment",
    "rubric_testing_alignment",
    "rubric_eval_alignment",
]

RUBRIC_TARGETS = ["technical_correctness", "completeness_score"]

# ---------------------------------------------------------------------------
# Data & Model Caching Layer (with Active Learning & Conformal Prediction)
# ---------------------------------------------------------------------------
@st.cache_data
def load_or_create_data() -> pd.DataFrame:
    """Loads synthetic dataset from disk or generates it automatically if missing."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Note: generate_data.py must be updated to output the new rubric_* columns
        df = generate_synthetic_dataset(n_samples=150)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training Conformal Predictors with Mentor Feedback...")
def train_rubric_models(df: pd.DataFrame):
    """Trains Random Forests wrapped in MAPIE for 90% confidence intervals, weighted by human feedback."""
    base_train = df[df["dataset_split"] == "historical_train"].copy()
    base_train["sample_weight"] = 1.0

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
        
        # Split for Conformal Calibration (80/20)
        X_fit, X_calib, y_fit, y_calib, w_fit, w_calib = train_test_split(
            X_train, y_train, weights, test_size=0.2, random_state=42
        )
        
        # Base Model
        base_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        base_rf.fit(X_fit, y_fit, sample_weight=w_fit)
        
        # Conformal Wrapper
        mapie_model = MapieRegressor(estimator=base_rf, cv="prefit", method="plus")
        mapie_model.fit(X_calib, y_calib)
        
        models[target] = mapie_model
        explainers[target] = shap.TreeExplainer(base_rf)
        
        y_pred_calib = mapie_model.predict(X_calib)
        metrics[f"{target}_r2"] = r2_score(y_calib, y_pred_calib)

    return models, explainers, metrics


def predict_with_conformal(model: MapieRegressor, X: pd.DataFrame, alpha: float = 0.10):
    """Returns mean prediction alongside strict 90% confidence bounds."""
    y_pred, y_pis = model.predict(X, alpha=alpha)
    lower_bound = np.clip(np.round(y_pis[:, 0, 0], 1), 0.0, 100.0)
    upper_bound = np.clip(np.round(y_pis[:, 1, 0], 1), 0.0, 100.0)
    interval_width = np.round(upper_bound - lower_bound, 1)
    return np.round(y_pred, 1), lower_bound, upper_bound, interval_width


def persist_mentor_decision(record: dict):
    """Appends human-in-the-loop mentor reviews to the local audit CSV."""
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([record])
    if FEEDBACK_LOG_PATH.exists():
        row_df.to_csv(FEEDBACK_LOG_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(FEEDBACK_LOG_PATH, mode="w", header=True, index=False)


if "mentor_reviews" not in st.session_state:
    st.session_state["mentor_reviews"] = {}


# ---------------------------------------------------------------------------
# Sidebar Controls & Active Learning Trigger
# ---------------------------------------------------------------------------
df_all = load_or_create_data()
models, explainers, train_metrics = train_rubric_models(df_all)

st.sidebar.title(" Calibration Controls")

st.sidebar.subheader(" Active Learning")
if st.sidebar.button(" Retrain AI on Mentor Logs", use_container_width=True, type="primary"):
    if FEEDBACK_LOG_PATH.exists():
        train_rubric_models.clear()
        st.toast("Model cache cleared. Rebuilding ensembles with mentor overrides!", icon="🚀")
        st.rerun()
    else:
        st.sidebar.error("No mentor logs exist yet.")

if train_metrics["is_retrained"]:
    st.sidebar.success(f"**Model actively trained** on {train_metrics['train_size']} samples (Includes weighted overrides).")
else:
    st.sidebar.info("Model trained on base historical data only.")

st.sidebar.divider()

interval_threshold = st.sidebar.slider(
    "Max Allowed 90% Interval Width (pts)", 
    min_value=5.0, max_value=30.0, value=15.0, step=1.0,
    help="Submissions where the 90% confidence interval width exceeds this threshold are flagged for mentor review."
)

st.title(" Rubric-Aware Project Quality Intelligence")
tab_queue, tab_live, tab_drift = st.tabs([
    " Batch Queue & Calibration", 
    " Live File Upload & Predict",
    " Mentor Calibration Health"
])


# ---------------------------------------------------------------------------
# TAB 1: Batch Queue & Calibration
# ---------------------------------------------------------------------------
with tab_queue:
    eval_df = df_all[df_all["dataset_split"] == "pending_evaluation"].copy().reset_index(drop=True)
    X_eval = eval_df[FEATURE_COLS]

    for target in RUBRIC_TARGETS:
        preds, lowers, uppers, widths = predict_with_conformal(models[target], X_eval)
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
            return f"⚠️ Flagged: Interval > {interval_threshold} pts"
        return "🟢 High Confidence (Auto)"

    eval_df["review_status"] = eval_df.apply(resolve_status, axis=1)

    filter_mode = st.radio("Filter Queue", ["All Pending", "⚠️ Flagged Only", "✅ Calibrated"], horizontal=True)
    if filter_mode == "⚠️ Flagged Only":
        filtered_df = eval_df[eval_df["review_status"].str.contains("Flagged")]
    elif filter_mode == "✅ Calibrated":
        filtered_df = eval_df[eval_df["review_status"].str.contains("Calibrated")]
    else:
        filtered_df = eval_df

    col_queue, col_inspector = st.columns([1.15, 1.35], gap="large")

    with col_queue:
        st.dataframe(filtered_df[["submission_id", "pred_technical_correctness", "pred_completeness_score", "max_uncertainty_width", "review_status"]], use_container_width=True, hide_index=True, height=320)
        
        available_ids = filtered_df["submission_id"].tolist()
        if not available_ids:
            st.info("No submissions match the filter.")
            st.stop()
            
        selected_sub_id = st.selectbox("🔍 Select Submission to Inspect:", options=available_ids)
        selected_row = eval_df[eval_df["submission_id"] == selected_sub_id].iloc[0]
        selected_pos_idx = int(eval_df.index[eval_df["submission_id"] == selected_sub_id][0])

        st.markdown("#### Extracted Signals")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Complexity", selected_row["code_complexity_score"])
        s2.metric("Test Cov", f"{selected_row['test_coverage_ratio']*100:.0f}%")
        s3.metric("Arch Align", f"{selected_row['rubric_arch_alignment']*100:.0f}%")
        s4.metric("Test Align", f"{selected_row['rubric_testing_alignment']*100:.0f}%")

    with col_inspector:
        st.subheader(f" AI Evaluation: `{selected_sub_id}`")
        if selected_row["max_uncertainty_width"] >= interval_threshold:
            st.warning(f"**High Uncertainty.** 90% confidence interval spans **{selected_row['max_uncertainty_width']:.1f} pts**. Review required.")
        else:
            st.success(f"**High Confidence.** 90% confidence interval spans only **{selected_row['max_uncertainty_width']:.1f} pts**.")

        m1, m2 = st.columns(2)
        m1.metric("Tech Correctness", f"{selected_row['pred_technical_correctness']} / 100", delta=f"90% CI: [{selected_row['lower_technical_correctness']}, {selected_row['upper_technical_correctness']}]", delta_color="off")
        m2.metric("Completeness", f"{selected_row['pred_completeness_score']} / 100", delta=f"90% CI: [{selected_row['lower_completeness_score']}, {selected_row['upper_completeness_score']}]", delta_color="off")
        
        rubric_choice = st.radio("Explain Dimension:", ["technical_correctness", "completeness_score"], horizontal=True, key="queue_radio")
        explainer = explainers[rubric_choice]
        shap_explanation = explainer(X_eval)
        
        fig, _ = plt.subplots(figsize=(8, 3.5))
        shap.plots.waterfall(shap_explanation[selected_pos_idx], max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.markdown("#### Mentor Calibration Console")
        current_mentor = st.text_input("Mentor ID:", value="mentor_01", key=f"mentor_{selected_sub_id}")
        
        existing_rev = st.session_state["mentor_reviews"].get(selected_sub_id, {})
        mc1, mc2 = st.columns(2)
        with mc1:
            cal_tech = st.slider("Calibrate Tech", 0.0, 100.0, float(existing_rev.get("final_technical_correctness", selected_row['pred_technical_correctness'])))
        with mc2:
            cal_comp = st.slider("Calibrate Comp", 0.0, 100.0, float(existing_rev.get("final_completeness_score", selected_row['pred_completeness_score'])))
        
        notes = st.text_area("Notes:", value=existing_rev.get("mentor_notes", ""))

        b1, b2 = st.columns(2)
        with b1:
            if st.button("✅ Approve AI Score", use_container_width=True):
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "submission_id": selected_sub_id, 
                    "mentor_id": current_mentor, "action": "Approved",
                    "pred_technical_correctness": selected_row["pred_technical_correctness"], "final_technical_correctness": selected_row["pred_technical_correctness"],
                    "pred_completeness_score": selected_row["pred_completeness_score"], "final_completeness_score": selected_row["pred_completeness_score"],
                    "uncertainty_std": selected_row["max_uncertainty_width"], "mentor_notes": notes or "Approved."
                }
                st.session_state["mentor_reviews"][selected_sub_id] = record
                persist_mentor_decision(record)
                st.rerun()
        with b2:
            if st.button(" Save Override", use_container_width=True, type="primary"):
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(), "submission_id": selected_sub_id, 
                    "mentor_id": current_mentor, "action": "Overridden",
                    "pred_technical_correctness": selected_row["pred_technical_correctness"], "final_technical_correctness": cal_tech,
                    "pred_completeness_score": selected_row["pred_completeness_score"], "final_completeness_score": cal_comp,
                    "uncertainty_std": selected_row["max_uncertainty_width"], "mentor_notes": notes or "Manual override."
                }
                st.session_state["mentor_reviews"][selected_sub_id] = record
                persist_mentor_decision(record)
                st.rerun()


# ---------------------------------------------------------------------------
# TAB 2: Live File Upload
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Upload Python code and project reports for live multimodal scoring.")
    
    col_up, col_res = st.columns([1.1, 1.4], gap="large")
    
    with col_up:
        uploaded_code = st.file_uploader("1. Upload Student Script (.py)", type=["py"])
        student_report_text = st.text_area("2. Paste Project Report (Markdown/Text)", height=150)
        live_test_cov = st.slider("3. Declared Test Coverage (%)", 0, 100, 75) / 100.0
        
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
            n1.metric("Arch Match", f"{nlp_metrics['rubric_arch_alignment']*100:.1f}%")
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
            
            p_tech, l_tech, u_tech, _ = predict_with_conformal(models["technical_correctness"], live_df)
            p_comp, l_comp, u_comp, _ = predict_with_conformal(models["completeness_score"], live_df)
            
            lm1, lm2 = st.columns(2)
            lm1.metric("Predicted Tech Score", f"{p_tech[0]}", delta=f"90% CI: [{l_tech[0]}, {u_tech[0]}]", delta_color="off")
            lm2.metric("Predicted Comp Score", f"{p_comp[0]}", delta=f"90% CI: [{l_comp[0]}, {u_comp[0]}]", delta_color="off")
            
            live_rubric = st.radio("Generate SHAP Breakdown For:", ["technical_correctness", "completeness_score"], horizontal=True, key="live_radio")
            
            live_shap_exp = explainers[live_rubric](live_df)
            fig_live, _ = plt.subplots(figsize=(8, 4.0))
            shap.plots.waterfall(live_shap_exp[0], max_display=7, show=False)
            plt.tight_layout()
            st.pyplot(fig_live, clear_figure=True)


# ---------------------------------------------------------------------------
# TAB 3: Mentor Drift Monitoring
# ---------------------------------------------------------------------------
with tab_drift:
    st.subheader("Mentor Harshness & Drift Analytics")
    st.markdown("Tracks systematic grading biases (Harsh vs. Lenient) across human reviewers to maintain calibration health.")
    
    if FEEDBACK_LOG_PATH.exists():
        fb_df = pd.read_csv(FEEDBACK_LOG_PATH)
        
        if "mentor_id" in fb_df.columns:
            drift_df = calculate_mentor_offsets(fb_df)
            
            d_col1, d_col2 = st.columns([1, 1.5])
            with d_col1:
                st.dataframe(
                    drift_df.rename(columns={
                        "mentor_id": "Mentor", "reviews_completed": "Reviews", 
                        "mean_tech_offset": "Tech Offset (Δ)", "mean_comp_offset": "Comp Offset (Δ)"
                    }), 
                    use_container_width=True, hide_index=True
                )
            
            with d_col2:
                fig_drift, ax = plt.subplots(figsize=(7, 3.5))
                y_pos = np.arange(len(drift_df["mentor_id"]))
                
                ax.barh(y_pos - 0.2, drift_df["mean_tech_offset"], height=0.4, label="Tech Correctness Bias")
                ax.barh(y_pos + 0.2, drift_df["mean_comp_offset"], height=0.4, label="Completeness Bias")
                
                ax.set_yticks(y_pos, labels=drift_df["mentor_id"])
                ax.axvline(0, color='black', linewidth=1, linestyle='--')
                ax.set_xlabel("Mean Score Offset (Pts from AI Baseline)")
                ax.set_title("Mentor Grading Bias Profile")
                ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig_drift)
        else:
            st.warning("Update your calibration log to include 'mentor_id' to view drift analytics.")
    else:
        st.info("No calibration data logged yet. Complete reviews in the Batch Queue to populate this dashboard.")