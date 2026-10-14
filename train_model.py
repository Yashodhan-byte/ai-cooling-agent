# -*- coding: utf-8 -*-
"""
Thermal Prediction Model
========================
Trains a Random Forest regressor to predict T_next from
(heat_load, flow_rate, nano_pct, T_current).

Usage
-----
    python train_model.py          # train and save model
    python train_model.py --eval   # also print evaluation metrics
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from simulator import generate_training_data

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "thermal_model.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "models", "training_data.csv")

FEATURES = ["heat_load", "flow_rate", "nano_pct", "T_current"]
TARGET   = "T_next"


# ------------------------------------------------------------------
def build_models():
    return {
        "RandomForest":       RandomForestRegressor(
                                  n_estimators=200, max_depth=10,
                                  min_samples_leaf=3, random_state=42, n_jobs=-1),
        "GradientBoosting":   GradientBoostingRegressor(
                                  n_estimators=150, max_depth=4,
                                  learning_rate=0.08, random_state=42),
        "LinearRegression":   Pipeline([
                                  ("scaler", StandardScaler()),
                                  ("model",  LinearRegression()),
                              ]),
        "Ridge":              Pipeline([
                                  ("scaler", StandardScaler()),
                                  ("model",  Ridge(alpha=1.0)),
                              ]),
    }


def train(n_samples: int = 5000, verbose: bool = True) -> object:
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # ── 1. Generate / cache data ──────────────────────────────────
    if os.path.exists(DATA_PATH):
        if verbose:
            print(f"[train] Loading cached data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        if verbose:
            print(f"[train] Generating {n_samples} synthetic samples …")
        df = generate_training_data(n_samples=n_samples)
        df.to_csv(DATA_PATH, index=False)
        if verbose:
            print(f"[train] Data saved -> {DATA_PATH}")

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 2. Compare models ─────────────────────────────────────────
    models = build_models()
    results = {}
    if verbose:
        print(f"\n{'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>6}")
        print("─" * 46)

    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        results[name] = {"model": mdl, "mae": mae, "rmse": rmse, "r2": r2}
        if verbose:
            print(f"{name:<22} {mae:>7.4f} {rmse:>7.4f} {r2:>6.4f}")

    # ── 3. Pick best model (lowest MAE) ──────────────────────────
    best_name = min(results, key=lambda k: results[k]["mae"])
    best      = results[best_name]["model"]
    if verbose:
        print(f"\n✅ Best model: {best_name}  (MAE = {results[best_name]['mae']:.4f}°C)")

    # ── 4. Save ───────────────────────────────────────────────────
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best, "features": FEATURES, "name": best_name}, f)
    if verbose:
        print(f"[train] Model saved -> {MODEL_PATH}")

    return best


def load_model():
    """Load the trained model (trains it first if not cached - trains if missing)."""
    if not os.path.exists(MODEL_PATH):
        print("[load_model] No saved model found – training now …")
        train(verbose=True)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["name"]


def predict_next_temp(
    model,
    T_current: float,
    heat_load: float,
    flow_rate: float,
    nano_pct: float = 1.0,
) -> float:
    """Convenience wrapper for single-step inference."""
    X = np.array([[heat_load, flow_rate, nano_pct, T_current]])
    return float(model.predict(X)[0])


# ------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="Force retrain")
    ap.add_argument("--samples", type=int, default=5000)
    args = ap.parse_args()

    if args.retrain and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    if args.retrain and os.path.exists(DATA_PATH):
        os.remove(DATA_PATH)

    train(n_samples=args.samples, verbose=True)
