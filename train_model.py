# -*- coding: utf-8 -*-
"""
Thermal Prediction Model — Academic Evaluation Engine
======================================================
Trains and cross-validates surrogate regression models to predict T_next from
(heat_load, flow_rate, nano_pct, T_current).

Features:
- K-Fold (5-Fold) Cross-Validation with standard deviation
- Residual error distribution analytics
- Feature importance attribution (Tree models)
- Parity plot generation data (Predicted vs Actual)
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from simulator import generate_training_data

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "models", "thermal_model.pkl")
DATA_PATH    = os.path.join(os.path.dirname(__file__), "models", "training_data.csv")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "model_metrics.pkl")

FEATURES = ["heat_load", "flow_rate", "nano_pct", "T_current"]
TARGET   = "T_next"


def build_models():
    return {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Ridge": Ridge(alpha=10.0),
        "LinearRegression": LinearRegression(),
    }


def train(n_samples: int = 5000, verbose: bool = True) -> dict:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 1. Dataset Generation or Loading
    if os.path.exists(DATA_PATH):
        if verbose:
            print(f"[train] Loading cached data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        if verbose:
            print(f"[train] Generating {n_samples} synthetic physics samples …")
        df = generate_training_data(n_samples=n_samples)
        df.to_csv(DATA_PATH, index=False)
        if verbose:
            print(f"[train] Data saved -> {DATA_PATH}")

    X = df[FEATURES].values
    y = df[TARGET].values

    # Train/Test 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = build_models()
    evaluation_records = {}

    if verbose:
        print(f"\n{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'5-Fold R² (μ ± σ)':>22}")
        print("─" * 70)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, mdl in models.items():
        # 5-Fold Cross-Validation
        cv_scores = cross_val_score(mdl, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std()

        # Fit on training split
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        mae  = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))

        evaluation_records[name] = {
            "model": mdl,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_r2_mean": float(cv_mean),
            "cv_r2_std": float(cv_std),
            "sample_actual": y_test[:150].tolist(),
            "sample_pred": y_pred[:150].tolist(),
            "residuals": (y_test - y_pred).tolist(),
        }

        if verbose:
            print(f"{name:<20} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {cv_mean:>10.4f} ± {cv_std:<6.4f}")

    # Pick Best Model
    best_name = min(evaluation_records, key=lambda k: evaluation_records[k]["mae"])
    best_mdl  = evaluation_records[best_name]["model"]

    # Feature Importance (if applicable)
    feat_importance = {}
    if hasattr(best_mdl, "feature_importances_"):
        for f, imp in zip(FEATURES, best_mdl.feature_importances_):
            feat_importance[f] = float(imp)

    evaluation_records[best_name]["feature_importances"] = feat_importance

    # Save Model Artifacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_mdl, "features": FEATURES, "name": best_name}, f)

    # Save Academic Evaluation Metrics Bundle
    clean_metrics = {}
    for k, v in evaluation_records.items():
        clean_metrics[k] = {
            "mae": v["mae"],
            "rmse": v["rmse"],
            "r2": v["r2"],
            "cv_r2_mean": v["cv_r2_mean"],
            "cv_r2_std": v["cv_r2_std"],
            "sample_actual": v["sample_actual"],
            "sample_pred": v["sample_pred"],
            "residuals_mean": float(np.mean(v["residuals"])),
            "residuals_std": float(np.std(v["residuals"])),
            "feature_importances": v.get("feature_importances", {}),
        }

    with open(METRICS_PATH, "wb") as f:
        pickle.dump({"best_model": best_name, "metrics": clean_metrics}, f)

    if verbose:
        print(f"\n✅ Optimal Surrogate Selected: {best_name} (MAE = {evaluation_records[best_name]['mae']:.4f}°C, R² = {evaluation_records[best_name]['r2']:.4f})")
        print(f"[train] Academic metrics bundle saved -> {METRICS_PATH}")

    return clean_metrics


def load_model():
    """Load the trained model and metadata."""
    if not os.path.exists(MODEL_PATH):
        print("[load_model] No saved model found – executing full academic training & K-Fold cross validation …")
        train(verbose=True)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["name"]


def load_metrics():
    """Load cached academic evaluation metrics for dashboard inspection."""
    if not os.path.exists(METRICS_PATH):
        train(verbose=False)
    with open(METRICS_PATH, "rb") as f:
        return pickle.load(f)


def predict_next_temp(
    model,
    T_current: float,
    heat_load: float,
    flow_rate: float,
    nano_pct: float = 1.0,
) -> float:
    """Inference for single time step."""
    X = np.array([[heat_load, flow_rate, nano_pct, T_current]])
    return float(model.predict(X)[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="Force retrain")
    ap.add_argument("--samples", type=int, default=5000)
    args = ap.parse_args()

    if args.retrain and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if args.retrain and os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)

    train(n_samples=args.samples, verbose=True)
