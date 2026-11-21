"""Fleet-wide summary for the web dashboard: one row per driver plus fleet totals.

Uses the same in-memory events and the same scoring rule as /driver/{id}/score and the
same premium formula as /price, so the dashboard and the per-driver endpoints agree.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Query

from . import api as core
from .pricing import premium_from_score, risk_score

router = APIRouter(tags=["fleet"])


def incident_prob(mean_speed: float) -> float:
    """The API's v1.2 rule (see driver_score): probability rises with mean speed."""
    return min(max(0.05 + 0.002 * float(mean_speed), 0.01), 0.99)


def _share(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def summarize(df: pd.DataFrame, base_premium: float) -> Dict[str, Any]:
    if df.empty or "driver_id" not in df.columns:
        return {"drivers": [], "fleet": {"drivers": 0, "events": 0, "base_premium": base_premium}}
    df = df.copy()
    df["ts"] = pd.to_datetime(df.get("ts"), errors="coerce")
    hour = df["ts"].dt.hour

    rows: List[Dict[str, Any]] = []
    for driver_id, g in df.groupby(df["driver_id"].astype(str)):
        h = hour.loc[g.index]
        mean_speed = float(g["speed_kph"].mean())
        prob = incident_prob(mean_speed)
        score = risk_score(prob)
        premium = premium_from_score(base_premium, score)
        road = g["road_type"].astype(str) if "road_type" in g else pd.Series(dtype=str)
        rows.append(
            {
                "driver_id": driver_id,
                "events": int(len(g)),
                "avg_speed": round(mean_speed, 1),
                "p95_speed": round(float(g["speed_kph"].quantile(0.95)), 1),
                "max_speed": round(float(g["speed_kph"].max()), 1),
                "hard_brake_rate": round(_share(g["hard_brake"].astype(float) > 0), 3)
                if "hard_brake" in g
                else 0.0,
                "phone_use_rate": round(_share(g["phone_use"].astype(float) > 0), 3)
                if "phone_use" in g
                else 0.0,
                "night_share": round(_share((h < 6) | (h >= 22)), 3),
                "highway_share": round(_share(road == "highway"), 3),
                "city_share": round(_share(road == "city"), 3),
                "urban_share": round(_share(road == "urban"), 3),
                "rain_share": round(_share(g["weather_code"].astype(str) == "RA"), 3)
                if "weather_code" in g
                else 0.0,
                "prob_incident": round(prob, 4),
                "risk_score": score,
                "premium": premium,
            }
        )
    rows.sort(key=lambda r: r["risk_score"], reverse=True)

    scores = [r["risk_score"] for r in rows]
    hist = [0] * 10
    for s in scores:
        hist[min(s // 10, 9)] += 1
    road_mix = df["road_type"].astype(str).value_counts().to_dict() if "road_type" in df else {}
    fleet = {
        "drivers": len(rows),
        "events": int(len(df)),
        "start": df["ts"].min().isoformat() if df["ts"].notna().any() else None,
        "end": df["ts"].max().isoformat() if df["ts"].notna().any() else None,
        "base_premium": base_premium,
        "avg_risk_score": round(sum(scores) / len(scores), 1),
        "avg_premium": round(sum(r["premium"] for r in rows) / len(rows), 2),
        "score_histogram": [{"bucket": f"{i * 10}-{i * 10 + 9}", "drivers": n} for i, n in enumerate(hist)],
        "road_mix": {k: int(v) for k, v in road_mix.items()},
    }
    return {"drivers": rows, "fleet": fleet}


@router.get("/fleet")
def fleet(base_premium: float = Query(1200.0, ge=100, le=20000)):
    """Every driver's behavior stats, risk score and premium, plus fleet totals."""
    return summarize(core._df_from_events_list(core.EVENTS), base_premium)


@router.get("/fleet/points")
def fleet_points(limit: int = Query(5000, ge=1, le=20000)):
    """GPS points for the map: position, speed, hard-brake flag and driver."""
    df = core._df_from_events_list(core.EVENTS)
    if df.empty:
        return {"count": 0, "points": []}
    cols = [c for c in ["driver_id", "lat", "lon", "speed_kph", "hard_brake"] if c in df.columns]
    pts = df[cols].dropna(subset=["lat", "lon"]).head(limit).round(
        {"lat": 5, "lon": 5, "speed_kph": 1}
    )
    return {"count": int(len(pts)), "points": pts.to_dict(orient="records")}
