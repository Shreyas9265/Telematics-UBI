# src/model_runtime.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from .features import build_daily_features, AGG_FEATURES

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/risk_model.pkl")

class RiskModel:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model_bundle: Dict[str, Any] | None = None
        self._load()

    def _load(self):
        if not Path(self.model_path).exists():
            self.model_bundle = None
            return
        self.model_bundle = joblib.load(self.model_path)

    @property
    def ready(self) -> bool:
        return self.model_bundle is not None

    def predict_driver_prob(self, events_for_driver: pd.DataFrame) -> dict:
        """
        Returns:
            {
              "prob_incident": float in [0,1],
              "risk_score": int in [0,100],
              "n_days": int,
              "details": {...optional...}
            }
        """
        if events_for_driver.empty:
            return {"prob_incident": None, "risk_score": None, "n_days": 0}

        daily = build_daily_features(events_for_driver)
        if daily.empty:
            return {"prob_incident": None, "risk_score": None, "n_days": 0}

        daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = daily[AGG_FEATURES].values.astype(float)

        if not self.ready:
            # Fallback heuristic if model missing
            h = (
                0.50 * _z(daily["hard_brakes_per_100km"]) +
                0.20 * _z(daily["avg_speed"]) +
                0.15 * _z(daily["phone_use_min_per_100km"]) +
                0.15 * _z(daily["night_km_share"])
            )
            prob = float(np.clip(h.mean() / 3.0, 0, 1))  # crude
        else:
            model = self.model_bundle["model"]
            prob = float(model.predict_proba(X).mean(axis=0)[1])

        risk_score = int(round(prob * 100))
        return {
            "prob_incident": prob,
            "risk_score": risk_score,
            "n_days": int(daily.shape[0]),
        }


def _z(s: pd.Series) -> pd.Series:
    s = s.astype(float).fillna(0)
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0:
        return s*0
    return (s - mu) / (sd + 1e-9)
