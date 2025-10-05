# tests/test_features.py
import pandas as pd
from src.features import build_daily_features

def test_build_daily_features_empty():
    df = pd.DataFrame(columns=["ts","driver_id"])
    out = build_daily_features(df)
    assert out.empty

def test_build_daily_features_basic():
    df = pd.DataFrame({
        "ts": pd.date_range("2025-01-01", periods=3, freq="T"),
        "driver_id": ["D_001"]*3,
        "speed_kph": [30, 40, 50],
        "hard_brake": [0,1,0],
        "hard_accel": [0,0,1],
        "phone_use_sec": [0,5,0],
        "is_rush_hour": [0,1,0],
        "is_urban": [1,1,0],
    })
    out = build_daily_features(df)
    assert out.shape[0] == 1
    assert "avg_speed" in out.columns
    assert out["km_total"].iloc[0] >= 0
