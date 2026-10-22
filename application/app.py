from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

from model_utils import (
    FEATURES,
    LOCKED_THRESHOLD,
    add_scores,
    aggregate_shap_values,
    load_json,
    load_model,
    risk_tier,
    transform_features,
    validate_input,
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

st.set_page_config(
    page_title="Bank Churn Decision Support",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    .small-note {font-size:0.88rem; opacity:0.78;}
    .risk-box {padding: 0.75rem 1rem; border-radius: 0.6rem; border: 1px solid #d7dce2;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model(ART / "xgb_churn_model.json")


@st.cache_resource
def get_explainer():
    model = get_model()
    bg = pd.read_csv(ART / "shap_background.csv")
    return shap.TreeExplainer(
        model,
        data=bg,
        feature_perturbation="interventional",
        model_output="probability",
    )


@st.cache_data
def get_metadata():
    return load_json(ART / "input_metadata.json")


@st.cache_data
def get_demo_metrics():
    return load_json(ART / "demo_metrics.json")


@st.cache_data
def get_study_evidence():
    return load_json(ROOT / "study_evidence.json")


@st.cache_data
def get_holdout_demo():
    return pd.read_csv(ART / "demo_holdout.csv")


@st.cache_data
def get_global_shap():
    return pd.read_csv(ART / "global_xgb_shap.csv")


def probability_gauge(prob: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.8,
                    "value": LOCKED_THRESHOLD * 100,
                },
                "steps": [
                    {"range": [0, 20]},
                    {"range": [20, LOCKED_THRESHOLD * 100]},
                    {"range": [LOCKED_THRESHOLD * 100, 70]},
                    {"range": [70, 100]},
                ],
            },
            title={"text": "Predicted churn probability"},
        )
    )
    fig.update_layout(height=310, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def local_explanation(row: pd.DataFrame) -> pd.DataFrame:
    xt = transform_features(row)
    sv = get_explainer()(xt)
    grouped = aggregate_shap_values(sv.values, list(xt.columns)).iloc[0]
    result = pd.DataFrame({"Feature": grouped.index, "Contribution": grouped.values})
    result["Absolute"] = result["Contribution"].abs()
    return result.sort_values("Absolute", ascending=False)


def overview_page():
    study = get_study_evidence()
    published = study["published_xgboost"]
    holdout = get_holdout_demo()
    global_shap = get_global_shap().sort_values("NormalizedImportance", ascending=True)

    st.title("Bank Customer Churn Decision Support System")
    st.caption(
        "Explainable decision-support prototype derived from the locked churn benchmarking study. "
        "It is not a production banking decision system."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Locked ROC-AUC", f"{published['ROC_AUC']:.4f}")
    c2.metric("Locked PR-AUC", f"{published['PR_AUC']:.4f}")
    c3.metric("Brier score", f"{published['Brier_Score']:.4f}")
    c4.metric("Decision threshold", f"{published['Locked_Threshold']:.3f}")

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Risk distribution on the sealed holdout")
        fig = px.histogram(
            holdout,
            x="ChurnProbability",
            nbins=35,
            labels={"ChurnProbability": "Predicted churn probability"},
        )
        fig.add_vline(x=LOCKED_THRESHOLD, line_dash="dash", annotation_text="Locked threshold")
        fig.update_layout(height=420, yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Demo XGBoost global SHAP importance")
        fig = px.bar(
            global_shap,
            x="NormalizedImportance",
            y="Feature",
            orientation="h",
            labels={"NormalizedImportance": "Normalized mean |SHAP|"},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Research finding: ranking customers by predicted risk captured approximately "
        f"{published['Top20_Recall']*100:.1f}% of observed churners within the highest-risk 20% "
        "of the sealed holdout set."
    )



def individual_page():
    meta = get_metadata()
    model = get_model()

    st.title("Individual Customer Prediction")
    st.caption("Enter one customer profile. Explanations describe model behavior, not causal effects.")

    with st.form("customer_form"):
        a, b, c = st.columns(3)
        with a:
            credit = st.number_input("Credit score", int(meta["CreditScore"]["min"]), int(meta["CreditScore"]["max"]), int(meta["CreditScore"]["median"]))
            geography = st.selectbox("Geography", meta["Geography"], index=meta["Geography"].index("France") if "France" in meta["Geography"] else 0)
            gender = st.selectbox("Gender", meta["Gender"])
            age = st.number_input("Age", int(meta["Age"]["min"]), int(meta["Age"]["max"]), int(meta["Age"]["median"]))
        with b:
            tenure = st.number_input("Tenure", int(meta["Tenure"]["min"]), int(meta["Tenure"]["max"]), int(meta["Tenure"]["median"]))
            balance = st.number_input("Balance", float(meta["Balance"]["min"]), float(meta["Balance"]["max"]), float(meta["Balance"]["median"]), step=100.0)
            products = st.number_input("Number of products", int(meta["NumOfProducts"]["min"]), int(meta["NumOfProducts"]["max"]), int(meta["NumOfProducts"]["median"]))
        with c:
            card = st.selectbox("Has credit card", [0, 1], format_func=lambda x: "Yes" if x else "No")
            active = st.selectbox("Active member", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")
            salary = st.number_input("Estimated salary", float(meta["EstimatedSalary"]["min"]), float(meta["EstimatedSalary"]["max"]), float(meta["EstimatedSalary"]["median"]), step=1000.0)
        submitted = st.form_submit_button("Score customer", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "CreditScore": credit,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": card,
            "IsActiveMember": active,
            "EstimatedSalary": salary,
        }])
        prob = float(model.predict_proba(transform_features(row))[:, 1][0])
        decision = "FLAGGED FOR REVIEW" if prob >= LOCKED_THRESHOLD else "NOT FLAGGED"
        tier = risk_tier(prob)

        left, right = st.columns([1, 1.25])
        with left:
            st.plotly_chart(probability_gauge(prob), use_container_width=True)
            x1, x2 = st.columns(2)
            x1.metric("Risk tier", tier)
            x2.metric("Locked decision", decision)
            st.caption(
                "Risk tiers are descriptive UI bands. The only locked classification threshold from the study is 0.392."
            )
        with right:
            st.subheader("Local SHAP explanation")
            exp = local_explanation(row).head(8).sort_values("Contribution")
            fig = px.bar(
                exp,
                x="Contribution",
                y="Feature",
                orientation="h",
                labels={"Contribution": "Contribution to predicted probability"},
            )
            fig.add_vline(x=0)
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Positive values increase this model's predicted churn probability relative to its SHAP background; "
                "negative values decrease it. These are predictive associations, not causal recommendations."
            )



def portfolio_page():
    model = get_model()
    st.title("Portfolio Prioritization")
    st.caption(
        "Rank a customer portfolio by predicted churn probability and select a capacity-constrained review list."
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        data = get_holdout_demo().copy()
        st.info("No file uploaded. Using the bundled 2,000-customer sealed holdout as a demonstration portfolio.")
        is_bundled = True
    else:
        data = pd.read_csv(uploaded)
        is_bundled = False

    ok, missing = validate_input(data)
    if not ok:
        st.error(f"Missing required columns: {missing}")
        return

    scored = add_scores(model, data)
    scored = scored.sort_values("ChurnProbability", ascending=False).reset_index(drop=True)
    scored["PriorityRank"] = np.arange(1, len(scored) + 1)

    capacity = st.slider("Retention / review capacity", 5, 50, 20, 5)
    selected_n = max(1, int(np.ceil(len(scored) * capacity / 100)))
    scored["SelectedForReview"] = False
    scored.loc[: selected_n - 1, "SelectedForReview"] = True

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio size", f"{len(scored):,}")
    c2.metric("Selected", f"{selected_n:,}")
    c3.metric("Mean predicted risk", f"{scored['ChurnProbability'].mean()*100:.1f}%")
    c4.metric("Flagged at 0.392", f"{(scored['ChurnProbability'] >= LOCKED_THRESHOLD).sum():,}")

    if "Exited" in scored.columns:
        observed_churners = int(pd.to_numeric(scored["Exited"]).sum())
        captured = int(pd.to_numeric(scored.loc[scored["SelectedForReview"], "Exited"]).sum())
        capture_rate = captured / observed_churners if observed_churners else np.nan
        st.success(
            f"Evaluation-only ground truth detected: the selected top {capacity}% captures "
            f"{captured}/{observed_churners} observed churners ({capture_rate*100:.1f}%)."
        )
        if not is_bundled:
            st.warning(
                "Ground-truth capture metrics are meaningful only for genuinely unseen evaluation data. "
                "Do not interpret them as validation if the uploaded records were used to train the model."
            )

    left, right = st.columns([1, 1.2])
    with left:
        fig = px.histogram(scored, x="ChurnProbability", nbins=35)
        fig.add_vline(x=LOCKED_THRESHOLD, line_dash="dash", annotation_text="0.392")
        fig.update_layout(height=370, xaxis_title="Predicted churn probability", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        top_plot = scored.head(min(30, len(scored))).copy()
        id_col = "CustomerId" if "CustomerId" in top_plot.columns else "PriorityRank"
        fig = px.bar(
            top_plot.sort_values("ChurnProbability"),
            x="ChurnProbability",
            y=top_plot[id_col].astype(str),
            orientation="h",
            labels={"y": "Customer", "ChurnProbability": "Predicted risk"},
        )
        fig.update_layout(height=370, yaxis_title="Customer")
        st.plotly_chart(fig, use_container_width=True)

    show_cols = [c for c in [
        "PriorityRank", "CustomerId", "ChurnProbability", "RiskTier",
        "LockedDecision", "SelectedForReview", "Exited"
    ] if c in scored.columns]
    st.subheader("Prioritized customers")
    st.dataframe(scored[show_cols].head(100), use_container_width=True, hide_index=True)

    csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button("Download scored portfolio", csv, "scored_customer_portfolio.csv", "text/csv")



def governance_page():
    study = get_study_evidence()
    st.title("Model Transparency & Governance")
    st.caption(
        "Study evidence is shown here to support responsible interpretation of the prototype. "
        "It does not establish legal fairness or production readiness."
    )

    st.subheader("Cross-model SHAP consensus")
    cons = pd.DataFrame(study["consensus_shap"], columns=["Feature", "Importance"]).sort_values("Importance")
    fig = px.bar(cons, x="Importance", y="Feature", orientation="h")
    fig.update_layout(height=390, xaxis_title="Mean normalized importance")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Age, NumOfProducts and IsActiveMember accounted for approximately 71% of consensus importance.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Robustness stress tests")
        rob = pd.DataFrame(
            list(study["robustness_xgboost_roc_delta_approx"].items()),
            columns=["Scenario", "ROC_AUC_Delta"],
        )
        fig = px.bar(rob, x="Scenario", y="ROC_AUC_Delta")
        fig.update_layout(height=360, xaxis_tickangle=-20, yaxis_title="Approx. change in ROC-AUC")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Simulated missingness produced substantially larger degradation than categorical substitutions.")

    with c2:
        st.subheader("Observed subgroup churn prevalence")
        rows = []
        for group, vals in study["subgroup_prevalence"].items():
            for level, value in vals.items():
                rows.append([group, level, value])
        sub = pd.DataFrame(rows, columns=["Group", "Level", "Prevalence"])
        fig = px.bar(sub, x="Level", y="Prevalence", color="Group", barmode="group")
        fig.update_layout(height=360, yaxis_tickformat=".0%", xaxis_title="Subgroup")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Direct sensitivity diagnostics")
    s = study["direct_sensitivity_xgboost"]
    d1, d2 = st.columns(2)
    d1.metric("Female − Male mean probability", f"+{s['Female_minus_Male_mean_probability']:.4f}")
    d2.metric("Mean Geography within-customer range", f"{s['Geography_mean_within_customer_range']:.3f}")
    st.caption(
        "These diagnostics hold other recorded inputs fixed. They measure direct model sensitivity, not causal effects or legal fairness."
    )

    st.subheader("Operational limitations")
    st.markdown(
        """
        - Validation is based on a single public banking dataset; external institutional validity is not established.
        - The data contain ten static predictors and no longitudinal transactions, contact history, or intervention outcomes.
        - Risk scores support prioritization; they do not determine which retention action will work.
        - Gender and Geography require governance review because the fitted model is directly sensitive to these attributes.
        - Calibration and subgroup behavior should be revalidated on the target institution and monitored over time.
        - Production use would additionally require drift monitoring, access control, reliability testing, audit logging, and cost-sensitive policy design.
        """
    )


PAGES = {
    "Overview": overview_page,
    "Individual Prediction": individual_page,
    "Portfolio Prioritization": portfolio_page,
    "Transparency & Governance": governance_page,
}

with st.sidebar:
    st.header("Churn Decision Support")
    page = st.radio("Navigate", list(PAGES.keys()))
    st.divider()
    st.caption("Primary demo model: XGBoost")
    st.caption(f"Locked threshold: {LOCKED_THRESHOLD:.3f}")
    st.caption("Research prototype · non-causal explanations")

PAGES[page]()
