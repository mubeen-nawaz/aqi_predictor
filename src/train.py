"""
Training Pipeline
Trains multiple ML models and saves the best one locally.
Uses a local 'models/' folder instead of Hopsworks Model Registry.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model   import Ridge
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

from src.features import load_feature_store

os.makedirs("models", exist_ok=True)

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend",
    "temperature", "humidity", "pressure",
    "wind_speed", "wind_deg", "clouds",
    "heat_index", "wind_chill", "is_calm", "is_humid",
]
TARGET_COL = "aqi"
MODEL_PATH     = "models/best_model.joblib"
SCALER_PATH    = "models/scaler.joblib"
METADATA_PATH  = "models/metadata.json"


def evaluate(model, X_test, y_test, name: str) -> dict:
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    print(f"  {name:30s} | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.3f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "model": model}


def train():
    print("\n" + "="*60)
    print("  🌿 AQI Predictor - Training Pipeline")
    print("="*60)

    # 1. Load data
    print("\n📂 Loading data from feature store...")
    df = load_feature_store()
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"   Total records: {len(df)}")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 4. Train multiple models
    print("\n🤖 Training models...")
    candidates = [
        ("Random Forest",        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("Gradient Boosting",    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
        ("Ridge Regression",     Ridge(alpha=10.0)),
    ]
    results = [evaluate(m.fit(X_train, y_train), X_test, y_test, n) for n, m in candidates]

    # 5. Pick best model (lowest MAE)
    best = min(results, key=lambda r: r["mae"])
    print(f"\n🏆 Best model: {best['name']} (MAE={best['mae']:.2f})")

    # 6. Feature importance (simple permutation-based for any model)
    print("\n📊 Computing feature importance...")
    feat_importance = _permutation_importance(best["model"], X_test, y_test)

    # 7. Save model + scaler + metadata (local model registry)
    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)

    metadata = {
        "model_name":    best["name"],
        "trained_at":    datetime.now().isoformat(),
        "n_samples":     len(df),
        "features":      FEATURE_COLS,
        "metrics": {
            "mae":  round(best["mae"],  3),
            "rmse": round(best["rmse"], 3),
            "r2":   round(best["r2"],   3),
        },
        "feature_importance": feat_importance,
        "all_results": [
            {"name": r["name"], "mae": round(r["mae"], 3),
             "rmse": round(r["rmse"], 3), "r2": round(r["r2"], 3)}
            for r in results
        ]
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model saved  → {MODEL_PATH}")
    print(f"✅ Scaler saved → {SCALER_PATH}")
    print(f"✅ Metadata     → {METADATA_PATH}")
    print("\n" + "="*60 + "\n")
    return metadata


def _permutation_importance(model, X, y, n_repeats=5) -> dict:
    """Simple permutation importance - no need for SHAP."""
    baseline_mae = mean_absolute_error(y, model.predict(X))
    importances  = {}
    rng = np.random.RandomState(42)
    for i, feat in enumerate(FEATURE_COLS):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, i])
            scores.append(mean_absolute_error(y, model.predict(X_perm)))
        importances[feat] = round(float(np.mean(scores) - baseline_mae), 4)
    # Sort descending
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


def load_model():
    """Load saved model for making predictions."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found! Run 'python src/train.py' first."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    return model, scaler, meta


if __name__ == "__main__":
    train()
