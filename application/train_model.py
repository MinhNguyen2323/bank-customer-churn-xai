from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from model_utils import (
    FEATURES,
    LOCKED_THRESHOLD,
    TRANSFORMED_FEATURES,
    aggregate_shap_values,
    transform_features,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "churn.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.045438329,
    "min_child_weight": 3.243329533,
    "subsample": 0.732972223,
    "colsample_bytree": 0.930320661,
    "gamma": 4.001822689,
    "reg_alpha": 2.1997e-7,
    "reg_lambda": 0.001056813,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "scale_pos_weight": 1.0,
    "n_jobs": 4,
    "random_state": 42,
    "verbosity": 0,
}


def main() -> None:
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()
    y = df["Exited"].astype(int)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_dev_t = transform_features(X_dev)
    X_test_t = transform_features(X_test)

    print("Training frozen XGBoost configuration...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_dev_t, y_dev)

    prob = model.predict_proba(X_test_t)[:, 1]
    pred = (prob >= LOCKED_THRESHOLD).astype(int)

    top_n = int(np.ceil(0.20 * len(prob)))
    top_idx = np.argsort(-prob)[:top_n]
    top20_recall = float(y_test.iloc[top_idx].sum() / y_test.sum())

    metrics = {
        "ROC_AUC": float(roc_auc_score(y_test, prob)),
        "PR_AUC": float(average_precision_score(y_test, prob)),
        "Brier_Score": float(brier_score_loss(y_test, prob)),
        "F1": float(f1_score(y_test, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_test, pred)),
        "MCC": float(matthews_corrcoef(y_test, pred)),
        "Precision": float(precision_score(y_test, pred)),
        "Recall": float(recall_score(y_test, pred)),
        "Top20_Recall": top20_recall,
        "Threshold": LOCKED_THRESHOLD,
        "Holdout_N": int(len(y_test)),
        "Holdout_Churners": int(y_test.sum()),
    }

    # Save model in XGBoost's native format for portability.
    model.save_model(ARTIFACT_DIR / "xgb_churn_model.json")

    # Holdout demo file. This is the exact unseen partition from the fixed split.
    demo = df.loc[X_test.index, ["CustomerId"] + FEATURES + ["Exited"]].copy()
    demo["ChurnProbability"] = prob
    demo.to_csv(ARTIFACT_DIR / "demo_holdout.csv", index=False)

    # Background for probability-scale SHAP.
    background = X_dev_t.sample(n=100, random_state=42)
    background.to_csv(ARTIFACT_DIR / "shap_background.csv", index=False)

    # Global XGBoost SHAP for the demo artifact.
    explain_x, _, _, _ = train_test_split(
        X_test_t,
        y_test,
        train_size=300,
        random_state=42,
        stratify=y_test,
    )
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    explanation = explainer(explain_x)
    grouped = aggregate_shap_values(explanation.values, list(explain_x.columns))
    global_imp = grouped.abs().mean().sort_values(ascending=False)
    global_imp = (global_imp / global_imp.sum()).rename("NormalizedImportance")
    global_imp.rename_axis("Feature").reset_index().to_csv(
        ARTIFACT_DIR / "global_xgb_shap.csv", index=False
    )

    # Feature ranges/defaults for input widgets.
    ranges = {}
    for col in [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
    ]:
        series = pd.to_numeric(X_dev[col])
        ranges[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }
    ranges["Geography"] = sorted(X_dev["Geography"].astype(str).unique().tolist())
    ranges["Gender"] = sorted(X_dev["Gender"].astype(str).unique().tolist())

    with open(ARTIFACT_DIR / "demo_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(ARTIFACT_DIR / "input_metadata.json", "w", encoding="utf-8") as f:
        json.dump(ranges, f, indent=2)

    print("Artifacts created in:", ARTIFACT_DIR)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
