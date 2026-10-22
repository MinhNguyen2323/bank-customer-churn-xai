from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL_LEVELS = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Female", "Male"],
}
TRANSFORMED_FEATURES = NUMERIC_FEATURES + [
    "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male",
]
ORIGINAL_GROUPS = {
    "CreditScore": ["CreditScore"],
    "Geography": ["Geography_France", "Geography_Germany", "Geography_Spain"],
    "Gender": ["Gender_Female", "Gender_Male"],
    "Age": ["Age"],
    "Tenure": ["Tenure"],
    "Balance": ["Balance"],
    "NumOfProducts": ["NumOfProducts"],
    "HasCrCard": ["HasCrCard"],
    "IsActiveMember": ["IsActiveMember"],
    "EstimatedSalary": ["EstimatedSalary"],
}

LOCKED_THRESHOLD = 0.392


def validate_input(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in FEATURES if c not in df.columns]
    return len(missing) == 0, missing


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    ok, missing = validate_input(df)
    if not ok:
        raise ValueError(f"Missing required columns: {missing}")

    x = df.copy()
    out = pd.DataFrame(index=x.index)

    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(x[col], errors="raise")

    for col, levels in CATEGORICAL_LEVELS.items():
        values = x[col].astype(str)
        invalid = sorted(set(values.unique()) - set(levels))
        if invalid:
            raise ValueError(
                f"Unsupported values in {col}: {invalid}. Allowed values: {levels}"
            )
        for level in levels:
            out[f"{col}_{level}"] = (values == level).astype(int)

    return out[TRANSFORMED_FEATURES]


def load_model(model_path: str | Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


def predict_probabilities(model: XGBClassifier, df: pd.DataFrame) -> np.ndarray:
    xt = transform_features(df)
    return model.predict_proba(xt)[:, 1]


def risk_tier(prob: float) -> str:
    # Descriptive display bands only. The scientific binary decision threshold remains 0.392.
    if prob < 0.20:
        return "Low"
    if prob < LOCKED_THRESHOLD:
        return "Moderate"
    if prob < 0.70:
        return "High"
    return "Critical"


def add_scores(model: XGBClassifier, df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    prob = predict_probabilities(model, result)
    result["ChurnProbability"] = prob
    result["LockedDecision"] = np.where(prob >= LOCKED_THRESHOLD, "Flagged", "Not flagged")
    result["RiskTier"] = [risk_tier(float(p)) for p in prob]
    return result


def aggregate_shap_values(
    shap_values: np.ndarray,
    transformed_columns: List[str] | None = None,
) -> pd.DataFrame:
    transformed_columns = transformed_columns or TRANSFORMED_FEATURES
    values = np.asarray(shap_values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    index = {name: i for i, name in enumerate(transformed_columns)}
    rows = []
    for row in values:
        item: Dict[str, float] = {}
        for original, expanded in ORIGINAL_GROUPS.items():
            item[original] = float(sum(row[index[c]] for c in expanded))
        rows.append(item)
    return pd.DataFrame(rows)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
