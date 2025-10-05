# import pandas as pd
# import numpy as np

# RISK_FEATURES = ["r_speed","r_brake","r_accel","r_night","r_phone","r_ctx"]

# def to_features(df_events: pd.DataFrame) -> pd.DataFrame:
#     df = df_events.copy()
#     if df.empty:
#         return pd.DataFrame()
#     df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
#     df["hour"] = df["ts"].dt.hour
#     df["is_night"] = df["hour"].between(21, 23) | df["hour"].between(0, 6)
#     df["is_rush"]  = df["hour"].between(7,9) | df["hour"].between(16,18)
#     df["dist_km"] = df.groupby("driver_id")["odometer_km"].diff().clip(lower=0).fillna(0)
#     df.loc[df["dist_km"]>3, "dist_km"] = 0  # basic GPS jump guard

#     df["is_hard_brake"] = (df["brake_mps2"] <= -3.0).astype(int)
#     df["is_hard_accel"] = (df["accel_mps2"] >=  3.0).astype(int)

#     g = df.groupby("driver_id")
#     agg = g.apply(lambda gdf: pd.Series({
#         "avg_speed": gdf["speed_kph"].mean(),
#         "p95_speed": gdf["speed_kph"].quantile(0.95),
#         "speed_std": gdf["speed_kph"].std(ddof=0),
#         "hard_brakes_per_100km": 100 * gdf["is_hard_brake"].sum() / (gdf["dist_km"].sum() + 1e-6),
#         "hard_accels_per_100km": 100 * gdf["is_hard_accel"].sum() / (gdf["dist_km"].sum() + 1e-6),
#         "night_km_share": (gdf.loc[gdf["is_night"], "dist_km"].sum()) / (gdf["dist_km"].sum() + 1e-6),
#         "rush_km_share": (gdf.loc[gdf["is_rush"], "dist_km"].sum()) / (gdf["dist_km"].sum() + 1e-6),
#         "phone_use_min_per_100km": 100 * gdf["phone_use"].sum() / (gdf["dist_km"].sum() + 1e-6),
#         "urban_km_share": (gdf.loc[gdf["road_type"]=="city","dist_km"].sum()) / (gdf["dist_km"].sum() + 1e-6),
#         "highway_km_share": (gdf.loc[gdf["road_type"]=="highway","dist_km"].sum()) / (gdf["dist_km"].sum() + 1e-6),
#     })).reset_index()

#     for col in ["local_crime_idx","traffic_incident_idx","weather_risk_idx"]:
#         if col not in agg.columns:
#             agg[col] = 0.5

#     def clip01(s): return np.clip(s, 0, 1)
#     X = agg.copy()
#     X["r_speed"]  = clip01((X["p95_speed"]-50)/60)
#     X["r_brake"]  = clip01(X["hard_brakes_per_100km"]/12)
#     X["r_accel"]  = clip01(X["hard_accels_per_100km"]/12)
#     X["r_night"]  = clip01(X["night_km_share"])
#     X["r_phone"]  = clip01(X["phone_use_min_per_100km"]/30)
#     X["r_ctx"]    = clip01(0.34*X["local_crime_idx"] + 0.33*X["traffic_incident_idx"] + 0.33*X["weather_risk_idx"])
#     return X


# src/features.py
from __future__ import annotations
import pandas as pd
import numpy as np

# ---- public API -------------------------------------------------------------

AGG_FEATURES = [
    "avg_speed",
    "p95_speed",
    "speed_std",
    "km_total",
    "hard_brakes_per_100km",
    "hard_accels_per_100km",
    "phone_use_min_per_100km",
    "night_km_share",
    "rush_km_share",
    "urban_km_share",
]

def build_daily_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Input: raw per-event rows with at least:
        ts (datetime or string), driver_id (str), speed_kph (float),
        hard_brake (0/1), hard_accel (0/1),
        phone_use_sec (float, optional),
        is_rush_hour (0/1, optional), is_urban (0/1, optional)

    Output: one row per (driver_id, date) with engineered features defined in AGG_FEATURES.
    """
    if events.empty:
        return pd.DataFrame(columns=["driver_id", "date", *AGG_FEATURES])

    df = events.copy()

    # Coerce types / fill defaults
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        raise ValueError("events must include 'ts' column")

    if "driver_id" not in df.columns:
        raise ValueError("events must include 'driver_id' column")

    # basic safety defaults
    for col, default in [
        ("speed_kph", 0.0),
        ("hard_brake", 0.0),
        ("hard_accel", 0.0),
        ("phone_use_sec", 0.0),
        ("is_rush_hour", 0.0),
        ("is_urban", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    # derive fields
    # very rough distance step (km) if not available: dt * speed
    df = df.sort_values(["driver_id", "ts"])
    # approximate per-row delta time in hours
    df["dt_hours"] = df.groupby("driver_id")["ts"].diff().dt.total_seconds().fillna(0) / 3600.0
    df["dt_hours"] = df["dt_hours"].clip(lower=0, upper=1)   # guard outliers
    df["km_step"] = (df["speed_kph"].clip(lower=0) * df["dt_hours"]).fillna(0)

    # contextual masks
    # night: 20:00–06:00
    hour = df["ts"].dt.hour
    df["is_night"] = ((hour >= 20) | (hour < 6)).astype(float)

    # group by driver/date
    df["date"] = df["ts"].dt.date

    g = df.groupby(["driver_id", "date"], as_index=False).apply(_agg_daily).reset_index(drop=True)

    # final selection & order
    cols = ["driver_id", "date"] + AGG_FEATURES
    for c in cols:
        if c not in g.columns:  # safety, but should be present
            g[c] = np.nan
    return g[cols]


def _agg_daily(group: pd.DataFrame) -> pd.Series:
    # avoid divide-by-zero
    km_total = group["km_step"].sum()
    km_safe = max(km_total, 1e-6)

    # speeds
    speeds = group["speed_kph"].clip(lower=0)
    avg_speed = speeds.mean()
    p95_speed = np.percentile(speeds, 95) if len(speeds) > 0 else 0.0
    speed_std = speeds.std(ddof=0)

    # per 100 km rates
    hard_brakes_per_100km = float(group["hard_brake"].sum()) / km_safe * 100.0
    hard_accels_per_100km = float(group["hard_accel"].sum()) / km_safe * 100.0

    phone_use_min_per_100km = float(group["phone_use_sec"].sum()) / 60.0 / km_safe * 100.0

    night_km_share = float((group["km_step"] * group["is_night"]).sum()) / km_safe
    rush_km_share = float((group["km_step"] * group["is_rush_hour"]).sum()) / km_safe
    urban_km_share = float((group["km_step"] * group["is_urban"]).sum()) / km_safe

    return pd.Series({
        "driver_id": group["driver_id"].iloc[0],
        "date": group["date"].iloc[0],
        "avg_speed": avg_speed,
        "p95_speed": p95_speed,
        "speed_std": speed_std,
        "km_total": km_total,
        "hard_brakes_per_100km": hard_brakes_per_100km,
        "hard_accels_per_100km": hard_accels_per_100km,
        "phone_use_min_per_100km": phone_use_min_per_100km,
        "night_km_share": night_km_share,
        "rush_km_share": rush_km_share,
        "urban_km_share": urban_km_share,
    })
