"""
Model training pipeline for salary prediction.
Trains an XGBoost Regressor with hyperparameter tuning via GridSearchCV.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .config import (
    DATA_PATH, MODEL_PATH, SCALER_PATH, PREPROCESSOR_PATH, METADATA_PATH, MODEL_DIR,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, XGB_PARAM_GRID, FEATURE_COLS,
    NUMERICAL_COLS, ORDINAL_COLS, BINARY_COLS, BRANCHES,
)
from .preprocessing import (
    load_data, prepare_training_data, get_feature_names_after_encoding,
)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
) -> XGBRegressor:
    """
    Train an XGBoost Regressor with optional hyperparameter tuning.

    Args:
        X_train: Scaled feature matrix.
        y_train: Log-transformed target values.
        tune: If True, run GridSearchCV for hyperparameter tuning.

    Returns:
        Trained XGBRegressor.
    """
    if tune:
        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        grid = GridSearchCV(
            base_model,
            XGB_PARAM_GRID,
            cv=CV_FOLDS,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print(f"Best params: {grid.best_params_}")
        print(f"Best CV MAE (log): {-grid.best_score_:.4f}")
        return grid.best_estimator_
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
        )
        model.fit(X_train, y_train)
        return model


def evaluate_model(
    model: XGBRegressor,
    X_test: np.ndarray,
    y_test_log: np.ndarray,
) -> dict:
    """
    Evaluate the model on test data. Metrics are computed in original salary scale.

    Returns dict with r2, mae, rmse, mean_salary, and predictions.
    """
    y_pred_log = model.predict(X_test)
    # Inverse log-transform
    y_test = np.expm1(y_test_log)
    y_pred = np.expm1(y_pred_log)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

    print(f"\n{'='*50}")
    print(f"Model Evaluation (Original Salary Scale - LPA)")
    print(f"{'='*50}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE:      {mae:.2f} LPA")
    print(f"RMSE:     {rmse:.2f} LPA")
    print(f"MAPE:     {mape:.2f}%")
    print(f"Mean Actual Salary: {y_test.mean():.2f} LPA")
    print(f"{'='*50}")

    return {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "mean_actual_salary": round(y_test.mean(), 2),
        "mean_predicted_salary": round(y_pred.mean(), 2),
        "n_test_samples": len(y_test),
    }


def save_model_and_metadata(
    model: XGBRegressor,
    scaler,
    preprocessor,
    metrics: dict,
) -> None:
    """Save the trained model, scaler, preprocessor, and metadata to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # Extract feature names and importance
    feature_names = get_feature_names_after_encoding(preprocessor)
    importances = model.feature_importances_
    feature_importance = [
        {"feature": name, "importance": round(float(imp), 6)}
        for name, imp in sorted(
            zip(feature_names, importances),
            key=lambda x: float(x[1]),
            reverse=True,
        )
    ]

    # Get branch encoding mapping
    branch_encoder = preprocessor.named_transformers_["branch_ohe"]
    branch_categories = branch_encoder.categories_[0].tolist()

    # Convert metrics values to native Python types for JSON serialization
    serializable_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer,)):
            serializable_metrics[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable_metrics[k] = float(v)
        else:
            serializable_metrics[k] = v

    metadata = {
        "feature_names": feature_names,
        "branch_categories": branch_categories,
        "branch_reference": branch_categories[0],  # dropped category for one-hot
        "numerical_cols": NUMERICAL_COLS,
        "ordinal_cols": ORDINAL_COLS,
        "binary_cols": BINARY_COLS,
        "log_transform": True,
        "feature_importance": feature_importance,
        "metrics": serializable_metrics,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


def main():
    """Run the full training pipeline."""
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} records. {df['placed'].sum()} placed students.")

    # Prepare training data (placed students only, log-transform target, scale features)
    print("\nPreparing training data...")
    X_scaled, y_log, preprocessor, scaler = prepare_training_data(df)

    print(f"Training features shape: {X_scaled.shape}")
    feature_names = get_feature_names_after_encoding(preprocessor)
    print(f"Features ({len(feature_names)}): {feature_names}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # Train model
    print("\nTraining XGBoost Regressor (with GridSearchCV)...")
    model = train_model(X_train, y_train, tune=True)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Save
    save_model_and_metadata(model, scaler, preprocessor, metrics)

    # Print top 10 features
    print("\nTop 10 Feature Importance:")
    for item in json.load(open(METADATA_PATH, "r", encoding="utf-8"))["feature_importance"][:10]:
        print(f"  {item['feature']:30s}: {item['importance']:.6f}")


if __name__ == "__main__":
    main()
