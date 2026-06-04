"""
Inference module for salary prediction.
Loads the trained model and provides prediction APIs.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import (
    MODEL_PATH, SCALER_PATH, PREPROCESSOR_PATH, METADATA_PATH,
    FEATURE_COLS, BRANCHES,
)
from .preprocessing import load_data


def load_model():
    """Load the trained model, scaler, preprocessor, and metadata from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, preprocessor, metadata


def _build_input_df(student: dict) -> pd.DataFrame:
    """
    Convert a student profile dict into a DataFrame suitable for preprocessing.

    Expected keys: cgpa, branch, college_tier, python_skill, dsa_skill,
                   ml_skill, web_dev_skill, coding_score, communication_score,
                   aptitude_score, internships, projects, backlogs,
                   resume_score, skill_score
    """
    row = {}
    for col in FEATURE_COLS:
        if col in student:
            row[col] = student[col]
        else:
            raise ValueError(f"Missing required feature: {col}")
    return pd.DataFrame([row])


def predict_salary(
    student: dict,
    model=None, scaler=None, preprocessor=None, metadata=None,
) -> dict:
    """
    Predict salary for a single student profile.

    Args:
        student: Dict with keys matching FEATURE_COLS.
        model, scaler, preprocessor, metadata: Optionally pre-loaded objects.

    Returns:
        dict with predicted_salary_lpa, predicted_salary_log, and top_factors.
    """
    if model is None or scaler is None or preprocessor is None or metadata is None:
        model, scaler, preprocessor, metadata = load_model()

    df_input = _build_input_df(student)

    # Ensure branch is a known category
    branch_cats = metadata.get("branch_categories", BRANCHES)
    df_input["branch"] = df_input["branch"].apply(
        lambda b: b if b in branch_cats else branch_cats[0]
    )

    # Use saved preprocessor to transform
    X_encoded = preprocessor.transform(df_input[FEATURE_COLS])
    X_scaled = scaler.transform(X_encoded)

    # Predict log salary
    y_pred_log = model.predict(X_scaled)[0]

    # Inverse log transform
    predicted_salary = np.expm1(y_pred_log)

    # Get top 3 driving factors for this prediction
    top_factors = _get_top_factors(model, X_scaled, metadata)

    return {
        "predicted_salary_lpa": round(float(predicted_salary), 2),
        "predicted_salary_log": round(float(y_pred_log), 4),
        "top_factors": top_factors,
    }


def _get_top_factors(model: XGBRegressor, X: np.ndarray, metadata: dict) -> list[dict]:
    """
    Identify which features contributed most to this prediction.
    Uses feature importance from metadata weighted by the feature value.
    """
    feature_names = metadata.get("feature_names", [])
    importances = metadata.get("feature_importance", [])

    if not importances:
        return []

    # Build importance dict
    imp_dict = {item["feature"]: item["importance"] for item in importances}

    # For each feature: |value| * importance
    top = []
    for i, name in enumerate(feature_names):
        if name in imp_dict and i < X.shape[1]:
            score = abs(float(X[0, i])) * imp_dict.get(name, 0)
            top.append({
                "feature": name,
                "value": float(X[0, i]),
                "score": round(score, 6),
            })

    top.sort(key=lambda x: x["score"], reverse=True)
    return top[:3]


def predict_batch(students_df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch prediction for multiple students.

    Args:
        students_df: DataFrame with FEATURE_COLS columns.

    Returns:
        DataFrame with added 'predicted_salary_lpa' column.
    """
    model, scaler, preprocessor, metadata = load_model()

    branch_cats = metadata.get("branch_categories", BRANCHES)
    students_df["branch"] = students_df["branch"].apply(
        lambda b: b if b in branch_cats else branch_cats[0]
    )

    X_encoded = preprocessor.transform(students_df[FEATURE_COLS])
    X_scaled = scaler.transform(X_encoded)
    y_pred_log = model.predict(X_scaled)
    result = students_df.copy()
    result["predicted_salary_lpa"] = np.expm1(y_pred_log).round(2)
    return result


def get_feature_importance() -> list[dict]:
    """Return feature importance from saved metadata."""
    _, _, _, metadata = load_model()
    return metadata.get("feature_importance", [])


def get_model_metrics() -> dict:
    """Return model evaluation metrics from saved metadata."""
    _, _, _, metadata = load_model()
    return metadata.get("metrics", {})
