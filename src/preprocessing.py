"""
Data loading, cleaning, and feature engineering for the salary prediction pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .config import (
    DATA_PATH, CATEGORICAL_COLS, NUMERICAL_COLS, ORDINAL_COLS,
    BINARY_COLS, FEATURE_COLS, TARGET_COL, PLACED_COL, ID_COL,
    POST_PLACEMENT_COLS, BRANCHES,
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    return df


def get_feature_names(encoder: OneHotEncoder | None = None) -> list[str]:
    """
    Return the ordered list of feature names after preprocessing.
    Must be called after fitting the preprocessor.
    """
    feature_names = []
    # Numerical + ordinal + binary pass through as-is
    feature_names.extend(NUMERICAL_COLS)
    feature_names.extend(ORDINAL_COLS)
    feature_names.extend(BINARY_COLS)
    # Categorical (branch) gets one-hot encoded
    if encoder is not None:
        branch_cols = [f"branch_{cat}" for cat in encoder.categories_[0]]
        feature_names.extend(branch_cols)
    else:
        # Before fitting, provide the expected names
        feature_names.extend([f"branch_{b}" for b in BRANCHES])
    return feature_names


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - OneHotEncodes 'branch' (drop='first' to avoid dummy variable trap)
      - Passes through ordinal and binary columns
      - Leaves numerical scaling to a separate StandardScaler (applied later)

    Returns an unfitted ColumnTransformer.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("branch_ohe", OneHotEncoder(drop="first", sparse_output=False), ["branch"]),
        ],
        remainder="passthrough",  # All other columns pass through unchanged
        verbose_feature_names_out=False,
    )
    return preprocessor


def preprocess_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, ColumnTransformer]:
    """
    Apply feature engineering to the dataset.

    Args:
        df: Raw DataFrame.
        preprocessor: Existing fitted ColumnTransformer, or None.
        fit: If True, fit the preprocessor on this data.

    Returns:
        X: Feature array (n_samples, n_features).
        preprocessor: The (possibly fitted) ColumnTransformer.
    """
    X_raw = df[FEATURE_COLS].copy()

    # Ensure branch is a known category; map any unseen values to the most common one
    X_raw["branch"] = X_raw["branch"].apply(
        lambda b: b if b in BRANCHES else "CSE"
    )

    if fit:
        X_transformed = preprocessor.fit_transform(X_raw)
    else:
        X_transformed = preprocessor.transform(X_raw)

    return X_transformed, preprocessor


def prepare_training_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer, StandardScaler]:
    """
    Full training data preparation pipeline.

    1. Filter to placed students only
    2. Build & fit preprocessor (OneHotEncode branch)
    3. Log-transform target (salary_lpa)
    4. Scale numerical features

    Returns:
        X: Scaled feature matrix.
        y_log: Log-transformed target values.
        preprocessor: Fitted ColumnTransformer.
        scaler: Fitted StandardScaler.
    """
    # Only train on placed students
    placed_df = df[df[PLACED_COL] == 1].copy()

    # Build preprocessor
    preprocessor = build_preprocessor()

    # Fit preprocessor on features
    X_encoded, preprocessor = preprocess_features(placed_df, preprocessor, fit=True)

    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Log-transform target (salary is right-skewed)
    y = placed_df[TARGET_COL].values
    y_log = np.log1p(y)

    return X_scaled, y_log, preprocessor, scaler


def prepare_all_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the entire dataset (including unplaced students) for analysis.
    Returns X_scaled, y (raw salary), placed_mask.
    """
    X_encoded, _ = preprocess_features(df, preprocessor, fit=False)
    X_scaled = scaler.transform(X_encoded)
    y = df[TARGET_COL].values
    placed_mask = df[PLACED_COL] == 1
    return X_scaled, y, placed_mask.values


def get_feature_names_after_encoding(preprocessor: ColumnTransformer) -> list[str]:
    """Return the ordered feature names after OneHotEncoding branch."""
    branch_cats = preprocessor.named_transformers_["branch_ohe"].categories_[0]
    # drop='first' means the first category is dropped
    branch_features = [f"branch_{cat}" for cat in branch_cats[1:]]
    # Remainder columns are NUMERICAL_COLS + ORDINAL_COLS + BINARY_COLS (in that order)
    remainder_features = NUMERICAL_COLS + ORDINAL_COLS + BINARY_COLS
    # The ColumnTransformer output order: transformer outputs first, then remainder
    return branch_features + remainder_features


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Compute aggregate statistics for the UI."""
    placed = df[df[PLACED_COL] == 1]

    stats = {
        "total_students": len(df),
        "placed_students": len(placed),
        "placement_rate": round(len(placed) / len(df) * 100, 1),
        "avg_salary": round(placed[TARGET_COL].mean(), 2),
        "median_salary": round(placed[TARGET_COL].median(), 2),
        "min_salary": round(placed[TARGET_COL].min(), 2),
        "max_salary": round(placed[TARGET_COL].max(), 2),
        "std_salary": round(placed[TARGET_COL].std(), 2),
        "salary_by_branch": placed.groupby("branch")[TARGET_COL].mean().round(2).to_dict(),
        "salary_by_tier": placed.groupby("college_tier")[TARGET_COL].mean().round(2).to_dict(),
        "salary_by_company": placed.groupby("company_type")[TARGET_COL].mean().round(2).to_dict(),
        "salary_by_role": placed.groupby("job_role")[TARGET_COL].mean().round(2).to_dict(),
    }
    return stats
