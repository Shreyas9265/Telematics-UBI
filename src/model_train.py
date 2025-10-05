# src/model_train.py
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

from .features import build_daily_features, AGG_FEATURES

DEFAULT_INPUT = "data/ingested_events.csv"   # per-event
DEFAULT_OUTPUT_DIR = "models"
DEFAULT_MODEL_PATH = "models/risk_model.pkl"

def _make_labels(daily: pd.DataFrame) -> pd.Series:
    """
    Produce a binary label for training.
    If you already have a column like `incident` per day/driver, use it.
    Otherwise, synthesize a proxy label from aggressive behavior.
    """
    if "incident" in daily.columns:
        y = (daily["incident"].fillna(0).astype(int) > 0).astype(int)
        return y

    # proxy: flag top-risk days as "1"
    score_proxy = (
        0.40 * (daily["hard_brakes_per_100km"].fillna(0)) +
        0.25 * (daily["avg_speed"].fillna(0) / daily["p95_speed"].replace(0, np.nan).fillna(1)) +
        0.20 * (daily["phone_use_min_per_100km"].fillna(0)) +
        0.15 * (daily["night_km_share"].fillna(0))
    )
    # label top ~20% as 1
    thresh = np.nanquantile(score_proxy, 0.80) if len(score_proxy) else np.inf
    y = (score_proxy >= thresh).astype(int)
    return y

def train(
    events_csv: str = DEFAULT_INPUT,
    out_model: str = DEFAULT_MODEL_PATH,
    min_km_per_day: float = 2.0,
    n_splits: int = 5,
):
    # 1) load raw events
    if not os.path.exists(events_csv):
        raise FileNotFoundError(f"Events CSV not found: {events_csv}")

    events = pd.read_csv(events_csv)
    # ensure ts is parseable (features will coerce again)
    if "ts" in events.columns:
        events["ts"] = pd.to_datetime(events["ts"], errors="coerce")

    # 2) daily features
    daily = build_daily_features(events)

    # 3) filter low-coverage days
    daily = daily[daily["km_total"] >= min_km_per_day].copy()
    daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4) labels
    y = _make_labels(daily)

    X = daily[AGG_FEATURES].values.astype(float)

    # 5) model (logreg + calibration). You can switch to GradientBoosting/LightGBM here.
    base = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",   # handles class imbalance
        max_iter=200,
        solver="liblinear",
    )
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", base),
    ])

    # Calibrate with CV for better probabilities
    model = CalibratedClassifierCV(base_estimator=clf, cv=3, method="sigmoid")

    # 6) CV metrics
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, aps = [], []
    for tr, te in skf.split(X, y):
        model.fit(X[tr], y.iloc[tr])
        proba = model.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], proba))
        aps.append(average_precision_score(y.iloc[te], proba))

    print(f"[CV] ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"[CV] PR-AUC : {np.mean(aps):.3f} ± {np.std(aps):.3f}")

    # 7) Fit on full data & persist
    model.fit(X, y)
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": AGG_FEATURES,
            "meta": {
                "n_samples": int(len(daily)),
                "event_csv": events_csv,
            },
        },
        out_model,
    )
    print(f"[OK] Saved model to {out_model}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT, help="Per-event CSV (API appends here)")
    p.add_argument("--output", default=DEFAULT_MODEL_PATH, help="Path to save trained model")
    p.add_argument("--min_km_per_day", type=float, default=2.0)
    p.add_argument("--cv", type=int, default=5)
    args = p.parse_args()

    train(
        events_csv=args.input,
        out_model=args.output,
        min_km_per_day=args.min_km_per_day,
        n_splits=args.cv,
    )
