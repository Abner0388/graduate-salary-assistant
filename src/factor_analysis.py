"""
Factor analysis: feature importance, counterfactual gains, and group statistics.
"""
import numpy as np
import pandas as pd

from .config import (
    FEATURE_COLS, BRANCHES, NUMERICAL_COLS, ORDINAL_COLS, BINARY_COLS,
    TARGET_COL, PLACED_COL,
)
from .inference import predict_salary, load_model, get_feature_importance
from .preprocessing import load_data, get_dataset_stats


def counterfactual_gains(student: dict) -> list[dict]:
    """
    For a given student profile, compute the marginal salary gain
    for improving each feature by one unit (or adding a skill).

    Returns a sorted list of {feature, current_value, improved_value, salary_gain}.
    """
    model, scaler, preprocessor, metadata = load_model()
    base_prediction = predict_salary(student, model, scaler, preprocessor, metadata)
    base_salary = base_prediction["predicted_salary_lpa"]

    gains = []

    # For each numerical feature, try increasing by ~10%
    for col in NUMERICAL_COLS:
        modified = student.copy()
        current = float(student[col])
        delta = max(0.5, current * 0.1)  # at least 0.5 increase
        modified[col] = current + delta
        new_pred = predict_salary(modified, model, scaler, preprocessor, metadata)
        gain = new_pred["predicted_salary_lpa"] - base_salary
        if gain > 0.01:
            gains.append({
                "feature": col,
                "current_value": round(current, 2),
                "improved_value": round(modified[col], 2),
                "salary_gain_lpa": round(gain, 2),
            })

    # For binary features, try adding the skill (0 -> 1)
    for col in BINARY_COLS:
        if student.get(col, 0) == 0:
            modified = student.copy()
            modified[col] = 1
            new_pred = predict_salary(modified, model, scaler, preprocessor, metadata)
            gain = new_pred["predicted_salary_lpa"] - base_salary
            if gain > 0.01:
                gains.append({
                    "feature": col,
                    "current_value": 0,
                    "improved_value": 1,
                    "salary_gain_lpa": round(gain, 2),
                })

    # For ordinal features
    for col in ORDINAL_COLS:
        current = int(student[col])
        if col == "college_tier" and current > 1:
            modified = student.copy()
            modified[col] = current - 1  # lower tier = better
            new_pred = predict_salary(modified, model, scaler, preprocessor, metadata)
            gain = new_pred["predicted_salary_lpa"] - base_salary
            if gain > 0.01:
                gains.append({
                    "feature": col,
                    "current_value": current,
                    "improved_value": current - 1,
                    "salary_gain_lpa": round(gain, 2),
                })

    # Internships: increment by 1
    if student.get("internships", 0) < 5:
        modified = student.copy()
        modified["internships"] = student["internships"] + 1
        new_pred = predict_salary(modified, model, scaler, preprocessor, metadata)
        gain = new_pred["predicted_salary_lpa"] - base_salary
        if gain > 0.01:
            gains.append({
                "feature": "internships",
                "current_value": student["internships"],
                "improved_value": student["internships"] + 1,
                "salary_gain_lpa": round(gain, 2),
            })

    # Projects: increment by 1
    if student.get("projects", 0) < 10:
        modified = student.copy()
        modified["projects"] = student["projects"] + 1
        new_pred = predict_salary(modified, model, scaler, preprocessor, metadata)
        gain = new_pred["predicted_salary_lpa"] - base_salary
        if gain > 0.01:
            gains.append({
                "feature": "projects",
                "current_value": student["projects"],
                "improved_value": student["projects"] + 1,
                "salary_gain_lpa": round(gain, 2),
            })

    # Sort by gain descending
    gains.sort(key=lambda x: x["salary_gain_lpa"], reverse=True)
    return gains


def get_group_stats() -> dict:
    """Compute and return pre-computed group comparison statistics."""
    df = load_data()
    return get_dataset_stats(df)


def compare_branches() -> pd.DataFrame:
    """Return average salary and placement stats by branch."""
    df = load_data()
    placed = df[df[PLACED_COL] == 1]

    result = []
    for branch in BRANCHES:
        branch_all = df[df["branch"] == branch]
        branch_placed = placed[placed["branch"] == branch]
        result.append({
            "branch": branch,
            "total_students": len(branch_all),
            "placed": len(branch_placed),
            "placement_rate": round(len(branch_placed) / max(len(branch_all), 1) * 100, 1),
            "avg_salary": round(branch_placed[TARGET_COL].mean(), 2),
            "median_salary": round(branch_placed[TARGET_COL].median(), 2),
            "max_salary": round(branch_placed[TARGET_COL].max(), 2),
        })

    return pd.DataFrame(result)


def compare_skills() -> pd.DataFrame:
    """Return average salary by skill combination."""
    df = load_data()
    placed = df[df[PLACED_COL] == 1]

    result = []
    for skill in BINARY_COLS:
        with_skill = placed[placed[skill] == 1]
        without_skill = placed[placed[skill] == 0]
        result.append({
            "skill": skill,
            "with_skill_avg_salary": round(with_skill[TARGET_COL].mean(), 2),
            "without_skill_avg_salary": round(without_skill[TARGET_COL].mean(), 2),
            "premium": round(with_skill[TARGET_COL].mean() - without_skill[TARGET_COL].mean(), 2),
            "with_skill_count": len(with_skill),
            "without_skill_count": len(without_skill),
        })

    return pd.DataFrame(result)


def compare_tiers() -> pd.DataFrame:
    """Return average salary by college tier."""
    df = load_data()
    placed = df[df[PLACED_COL] == 1]

    result = []
    for tier in [1, 2, 3]:
        tier_data = placed[placed["college_tier"] == tier]
        result.append({
            "college_tier": tier,
            "avg_salary": round(tier_data[TARGET_COL].mean(), 2),
            "median_salary": round(tier_data[TARGET_COL].median(), 2),
            "count": len(tier_data),
        })

    return pd.DataFrame(result)
