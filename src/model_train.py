# # src/model_train.py
# from __future__ import annotations
# import argparse
# import os
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score, average_precision_score
# import joblib

# from .features import build_daily_features, AGG_FEATURES

# DEFAULT_INPUT = "data/ingested_events.csv"   # per-event
# DEFAULT_OUTPUT_DIR = "models"
# DEFAULT_MODEL_PATH = "models/risk_model.pkl"

# def _make_labels(daily: pd.DataFrame) -> pd.Series:
#     """
#     Produce a binary label for training.
#     If you already have a column like `incident` per day/driver, use it.
#     Otherwise, synthesize a proxy label from aggressive behavior.
#     """
#     if "incident" in daily.columns:
#         y = (daily["incident"].fillna(0).astype(int) > 0).astype(int)
#         return y

#     # proxy: flag top-risk days as "1"
#     score_proxy = (
#         0.40 * (daily["hard_brakes_per_100km"].fillna(0)) +
#         0.25 * (daily["avg_speed"].fillna(0) / daily["p95_speed"].replace(0, np.nan).fillna(1)) +
#         0.20 * (daily["phone_use_min_per_100km"].fillna(0)) +
#         0.15 * (daily["night_km_share"].fillna(0))
#     )
#     # label top ~20% as 1
#     thresh = np.nanquantile(score_proxy, 0.80) if len(score_proxy) else np.inf
#     y = (score_proxy >= thresh).astype(int)
#     return y

# def train(
#     events_csv: str = DEFAULT_INPUT,
#     out_model: str = DEFAULT_MODEL_PATH,
#     min_km_per_day: float = 2.0,
#     n_splits: int = 5,
# ):
#     # 1) load raw events
#     if not os.path.exists(events_csv):
#         raise FileNotFoundError(f"Events CSV not found: {events_csv}")

#     events = pd.read_csv(events_csv)
#     # ensure ts is parseable (features will coerce again)
#     if "ts" in events.columns:
#         events["ts"] = pd.to_datetime(events["ts"], errors="coerce")

#     # 2) daily features
#     daily = build_daily_features(events)

#     # 3) filter low-coverage days
#     daily = daily[daily["km_total"] >= min_km_per_day].copy()
#     daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0)

#     # 4) labels
#     y = _make_labels(daily)

#     X = daily[AGG_FEATURES].values.astype(float)

#     # 5) model (logreg + calibration). You can switch to GradientBoosting/LightGBM here.
#     base = LogisticRegression(
#         penalty="l2",
#         C=1.0,
#         class_weight="balanced",   # handles class imbalance
#         max_iter=200,
#         solver="liblinear",
#     )
#     clf = Pipeline(steps=[
#         ("scaler", StandardScaler()),
#         ("logreg", base),
#     ])

#     # Calibrate with CV for better probabilities
#     model = CalibratedClassifierCV(base_estimator=clf, cv=3, method="sigmoid")

#     # 6) CV metrics
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     aucs, aps = [], []
#     for tr, te in skf.split(X, y):
#         model.fit(X[tr], y.iloc[tr])
#         proba = model.predict_proba(X[te])[:, 1]
#         aucs.append(roc_auc_score(y.iloc[te], proba))
#         aps.append(average_precision_score(y.iloc[te], proba))

#     print(f"[CV] ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
#     print(f"[CV] PR-AUC : {np.mean(aps):.3f} ± {np.std(aps):.3f}")

#     # 7) Fit on full data & persist
#     model.fit(X, y)
#     os.makedirs(os.path.dirname(out_model), exist_ok=True)
#     joblib.dump(
#         {
#             "model": model,
#             "features": AGG_FEATURES,
#             "meta": {
#                 "n_samples": int(len(daily)),
#                 "event_csv": events_csv,
#             },
#         },
#         out_model,
#     )
#     print(f"[OK] Saved model to {out_model}")


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--input", default=DEFAULT_INPUT, help="Per-event CSV (API appends here)")
#     p.add_argument("--output", default=DEFAULT_MODEL_PATH, help="Path to save trained model")
#     p.add_argument("--min_km_per_day", type=float, default=2.0)
#     p.add_argument("--cv", type=int, default=5)
#     args = p.parse_args()

#     train(
#         events_csv=args.input,
#         out_model=args.output,
#         min_km_per_day=args.min_km_per_day,
#         n_splits=args.cv,
#     )
#----model comaprision bet 
# src/model_train.py
from __future__ import annotations
import argparse
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss,
    brier_score_loss, accuracy_score, roc_curve
)
import matplotlib.pyplot as plt
import joblib

# NOTE: absolute import so you can run from src/ directly
from features import build_daily_features, AGG_FEATURES

DEFAULT_INPUT = "data/ingested_events.csv"   # per-event
DEFAULT_MODEL_PATH = "models/risk_model.pkl" # final (best) model
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODELS = ROOT / "models"
DOCS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)


def _make_labels(daily: pd.DataFrame) -> pd.Series:
    """
    Produce a binary label for training.
    If you already have a column like `incident` per day/driver, use it.
    Otherwise, synthesize a proxy label from aggressive behavior.
    """
    if "incident" in daily.columns:
        y = (daily["incident"].fillna(0).astype(int) > 0).astype(int)
        return y

    score_proxy = (
        0.40 * daily["hard_brakes_per_100km"].fillna(0) +
        0.25 * (daily["avg_speed"].fillna(0) /
                daily["p95_speed"].replace(0, np.nan).fillna(1)) +
        0.20 * daily["phone_use_min_per_100km"].fillna(0) +
        0.15 * daily["night_km_share"].fillna(0)
    )
    thresh = np.nanquantile(score_proxy, 0.80) if len(score_proxy) else np.inf
    return (score_proxy >= thresh).astype(int)


def _load_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv" and path.exists():
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".ndjson", ".jsonl"} and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(rows)
    else:
        raise FileNotFoundError(f"Events file not found or unsupported: {path}")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


def _models():
    """Return the two candidate models with nice display names."""
    # Baseline: calibrated Logistic Regression (+ scaling)
# Baseline: calibrated Logistic Regression (+ scaling)
    base_lr = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced", max_iter=400, solver="liblinear"
        )),
    ])

# ✅ compatible with all versions of scikit-learn
    try:
        lr_cal = CalibratedClassifierCV(estimator=base_lr, cv=3, method="sigmoid")   # new API
    except TypeError:
        lr_cal = CalibratedClassifierCV(base_estimator=base_lr, cv=3, method="sigmoid")  # old API

    

    # Nonlinear: Gradient Boosting (captures feature interactions)
    gboost = GradientBoostingClassifier(random_state=42)

    return {
        "LogisticRegression+Calibrated": lr_cal,
        "GradientBoosting": gboost,
    }


def _evaluate_models(X, y, models: dict, docs_dir: Path, models_dir: Path, final_path: Path):
    """Train/test split for plots + CV metrics for stability. Save artifacts & best model."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    # Split once to build ROC/calibration plots on a held-out set
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # CV (robustness) + holdout (plots)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    fitted = {}
    for name, model in models.items():
        # Cross-validated metrics
        aucs, aps = [], []
        for tr_idx, te_idx in skf.split(X, y):
            model.fit(X[tr_idx], y[tr_idx])
            proba = model.predict_proba(X[te_idx])[:, 1]
            aucs.append(roc_auc_score(y[te_idx], proba))
            aps.append(average_precision_score(y[te_idx], proba))

        # Fit on train split for plots
        model.fit(X_tr, y_tr)
        fitted[name] = model

        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        rows.append({
            "model": name,
            "roc_auc_cv_mean": float(np.mean(aucs)),
            "roc_auc_cv_std": float(np.std(aucs)),
            "pr_auc_cv_mean": float(np.mean(aps)),
            "pr_auc_cv_std": float(np.std(aps)),
            "roc_auc_holdout": float(roc_auc_score(y_te, proba)),
            "log_loss_holdout": float(log_loss(y_te, proba, labels=[0, 1])),
            "brier_holdout": float(brier_score_loss(y_te, proba)),
            "accuracy_holdout": float(accuracy_score(y_te, pred)),
        })

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc_cv_mean", ascending=False)
    metrics_csv = docs_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # ROC plot
    plt.figure()
    for name, model in fitted.items():
        proba = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, proba):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC – Logistic vs GradientBoosting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(docs_dir / "roc_comparison.png", dpi=160)
    plt.close()

    # Calibration plot
    plt.figure()
    for name, model in fitted.items():
        proba = model.predict_proba(X_te)[:, 1]
        CalibrationDisplay.from_predictions(y_te, proba, n_bins=10, name=name)
    plt.title("Calibration – Reliability Diagram")
    plt.tight_layout()
    plt.savefig(docs_dir / "calibration.png", dpi=160)
    plt.close()

    # Save the best model by mean CV AUC
    best_name = metrics_df.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, final_path)

    return metrics_df, best_name, metrics_csv


def train(
    events_csv: str = DEFAULT_INPUT,
    out_model: str = DEFAULT_MODEL_PATH,
    min_km_per_day: float = 2.0,
):
    # 1) load raw events
    events = _load_events(events_csv)

    # 2) daily features
    daily = build_daily_features(events)

    # 3) filter low-coverage days
    if "km_total" in daily.columns:
        daily = daily[daily["km_total"] >= min_km_per_day].copy()

    daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4) labels
    y = _make_labels(daily)
    X = daily[AGG_FEATURES].values.astype(float)

    # 5) compare models + save artifacts
    models = _models()
    metrics_df, best_name, metrics_csv = _evaluate_models(
        X, y, models, DOCS, MODELS, Path(out_model)
    )

    # 6) print summary
    print("\n== Model Comparison ==")
    print(metrics_df.to_string(index=False))
    print(f"\nSaved best model to: {out_model} (best = {best_name})")
    print("Artifacts:")
    print(f" - {metrics_csv}")
    print(f" - {DOCS/'roc_comparison.png'}")
    print(f" - {DOCS/'calibration.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT, help="Per-event CSV or events.ndjson")
    p.add_argument("--output", default=DEFAULT_MODEL_PATH, help="Path to save trained model")
    p.add_argument("--min_km_per_day", type=float, default=2.0)
    args = p.parse_args()

    train(
        events_csv=args.input,
        out_model=args.output,
        min_km_per_day=args.min_km_per_day,
    )
