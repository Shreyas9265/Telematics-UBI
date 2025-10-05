# src/api.py
from __future__ import annotations

from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import defaultdict
from src.db import SessionLocal, init_db
from sqlalchemy import text

import pandas as pd
import joblib

from src.features import to_features, RISK_FEATURES
from src.pricing import risk_score, premium_from_score

CSV_PATH = Path("data/ingested.csv")
MODEL_PATH = Path("models/model.joblib")

app = FastAPI(title="Telematics UBI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.events: List[Dict[str, Any]] = []
app.state.by_driver = defaultdict(int)
app.state.seen_keys = set()
app.state.model = None

import numpy as np

def risk_score(prob: float) -> int:
    """
    Map 0..1 probability to 0..100 risk score smoothly, avoiding early saturation.
    Simple base mapping: score = round(prob * 100).
    """
    if prob is None or np.isnan(prob):
        return 0
    return int(max(0, min(100, round(prob * 100))))

def premium_from_score(base_premium: float, score: int) -> float:
    """
    Guardrails: +/-35% from base across 0..100.
    Linear around 50 (neutral).
    """
    swing = 0.35  # max uplift/downlift
    factor = 1.0 + swing * (score - 50) / 50.0
    return round(base_premium * factor, 2)

def _append_event_csv(event: Dict[str, Any]) -> None:
    CSV_PATH.parent.mkdir(exist_ok=True)
    df = pd.DataFrame([event])
    write_header = not CSV_PATH.exists()
    if CSV_PATH.exists():
        try:
            existing_cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
            for c in existing_cols:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[[*existing_cols, *[c for c in df.columns if c not in existing_cols]]]
        except Exception:
            pass
    df.to_csv(CSV_PATH, mode="a", index=False, header=write_header)


def _load_csv_chunks_for_driver(
    driver_id: str, start: Optional[str], end: Optional[str]
) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    if not CSV_PATH.exists():
        return frames
    for chunk in pd.read_csv(CSV_PATH, chunksize=200_000):
        sub = chunk[chunk["driver_id"].astype(str) == driver_id]
        if not sub.empty:
            if start:
                sub = sub[pd.to_datetime(sub["ts"]) >= pd.to_datetime(start)]
            if end:
                sub = sub[pd.to_datetime(sub["ts"]) <= pd.to_datetime(end)]
            if not sub.empty:
                frames.append(sub)
    return frames

def _events_df(driver_id: str, start: Optional[str], end: Optional[str], source: str="db") -> pd.DataFrame:
    with SessionLocal() as s:
        where = "driver_id=:d"
        params = {"d": driver_id}
        if start:
            where += " AND ts >= :start"; params["start"] = start
        if end:
            where += " AND ts <= :end"; params["end"] = end
        q = f"SELECT * FROM events WHERE {where} ORDER BY ts"
        df = pd.read_sql(text(q), s.bind, params=params)
    return df


def _predict_for_driver(driver_id: str, start=None, end=None, source="both"):
    if app.state.model is None:
        return {"message": "Model not found yet. Train first and restart the API."}

    df = _events_df(driver_id, start, end, source)  # combines memory+CSV
    if df.empty:
        return {"message": f"No events for driver {driver_id} in selected range/source."}

    feats = to_features(df)
    row = feats.loc[feats["driver_id"] == driver_id]
    if row.empty:
        return {"message": f"Could not assemble features for driver {driver_id}."}

    X = row[RISK_FEATURES].values
    prob = float(app.state.model.predict_proba(X)[0, 1])
    score = risk_score(prob)
    return {"driver_id": driver_id, "prob_incident": prob, "risk_score": score}
from src.db import init_db

@app.on_event("startup")
def on_startup():
    try:
        init_db()
    except Exception as e:
        print(f"DB init failed: {e}")


@app.on_event("startup")
def _startup() -> None:
    try:
        if MODEL_PATH.exists():
            app.state.model = joblib.load(MODEL_PATH)
    except Exception:
        app.state.model = None

    CSV_PATH.parent.mkdir(exist_ok=True)
    if CSV_PATH.exists():
        try:
            for chunk in pd.read_csv(CSV_PATH, dtype=str, chunksize=200_000):
                if {"driver_id", "ts"}.issubset(chunk.columns):
                    keys = (chunk["driver_id"].astype(str) + "|" + chunk["ts"].astype(str))
                    app.state.seen_keys.update(keys.tolist())
        except Exception:
            pass
    init_db()


class PriceRequest(BaseModel):
    driver_id: str
    base_premium: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Telematics UBI API. See /docs"}


@app.get("/health")
def health():
    with SessionLocal() as s:
        rows = s.execute(text("SELECT count(*) FROM events")).scalar()
        drivers = s.execute(text("SELECT count(DISTINCT driver_id) FROM events")).scalar()
    return {"ok": True, "model_loaded": app.state.model is not None,
            "in_memory_events": len(app.state.events), "csv_rows": 0,
            "db_rows": rows, "unique_drivers_db": drivers}


from fastapi import Depends, Header, HTTPException
API_KEY = os.getenv("API_KEY", "dev-key")

def require_key(x_api_key: str = Header(default=None)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(401, "Unauthorized")
    return True

# Protect mutating endpoints (and sensitive reads)
#@app.post("/ingest", dependencies=[Depends(require_key)])

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: PlainTextResponse("Too Many Requests", 429))

@app.post("/ingest")
@limiter.limit("30/second")

#@app.post("/ingest")
def ingest(event: Dict[str, Any] = Body(...)):
    payload = event
    driver_id = str(payload.get("driver_id")); ts = str(payload.get("ts"))
    if not driver_id or not ts:
        return {"status":"error","message":"Event must include 'driver_id' and 'ts'."}

    with SessionLocal() as s, s.begin():
        # upsert-like: on conflict do nothing (prevent dup)
        s.execute(text("""
            INSERT INTO events(driver_id, ts, lat, lon, speed_kph, accel_mps2, brake_mps2,
                               heading_deg, odometer_km, phone_use, road_type, weather_code)
            VALUES (:driver_id, :ts, :lat, :lon, :speed_kph, :accel_mps2, :brake_mps2,
                    :heading_deg, :odometer_km, :phone_use, :road_type, :weather_code)
            ON CONFLICT (driver_id, ts) DO NOTHING
        """), payload)
    # maintain minimal in-memory counters if you like:
    app.state.by_driver[driver_id] += 1
    return {"status":"ok","stored": True}



@app.get("/drivers")
def drivers():
    return {"drivers": sorted(app.state.by_driver.keys())}


@app.get("/stats")
def stats():
    by_driver = [{"driver_id": k, "events": v} for k, v in app.state.by_driver.items()]
    by_driver = sorted(by_driver, key=lambda x: x["events"], reverse=True)
    csv_rows = 0
    if CSV_PATH.exists():
        try:
            for chunk in pd.read_csv(CSV_PATH, chunksize=200_000):
                csv_rows += len(chunk)
        except Exception:
            pass
    return {
        "in_memory_events": len(app.state.events),
        "unique_drivers_memory": len(app.state.by_driver),
        "file_rows_csv": csv_rows,
        "by_driver_top": by_driver[:50],
    }


@app.get("/driver/{driver_id}/events")
def driver_events(
    driver_id: str,
    start: Optional[str] = Query(None, description="ISO8601 start timestamp"),
    end: Optional[str]   = Query(None, description="ISO8601 end timestamp"),
    limit: int           = Query(5000, ge=0, description="Max rows to return; 0 = all"),
    source: str          = Query("both", regex="^(both|memory|csv)$"),
):
    """Return raw events for a driver from memory and/or CSV."""
    df = _events_df(driver_id, start, end, source)
    if df.empty:
        return {"count": 0, "events": []}
    if limit and limit > 0:
        df = df.head(limit)
    return {"count": int(len(df)), "events": df.to_dict(orient="records")}


@app.get("/driver/{driver_id}/features")
def driver_features(
    driver_id: str,
    start: Optional[str] = Query(None),
    end: Optional[str]   = Query(None),
    source: str          = Query("both", regex="^(both|memory|csv)$"),
):
    """Return engineered feature vector for a driver."""
    df = _events_df(driver_id, start, end, source)
    if df.empty:
        return {"message": f"No events for driver {driver_id} in selected range."}
    feats = to_features(df)
    try:
        row = feats.loc[feats["driver_id"] == driver_id].iloc[0].to_dict()
    except Exception:
        # if driver_id becomes index
        row = feats.iloc[0].to_dict()
    return {"driver_id": driver_id, "features": row}


@app.get("/driver/{driver_id}/score")
def driver_score(
    driver_id: str,
    start: Optional[str] = Query(None, description="ISO8601 start timestamp"),
    end: Optional[str]   = Query(None, description="ISO8601 end timestamp"),
    source: str          = Query("both", regex="^(both|memory|csv)$"),
):
    return _predict_for_driver(driver_id, start=start, end=end, source=source)



@app.post("/price")
def price(req: PriceRequest, source: str = Query("both", regex="^(both|memory|csv)$")):
    result = _predict_for_driver(req.driver_id, source=source)
    if "message" in result:
        return result
    premium = float(premium_from_score(req.base_premium, result["risk_score"]))
    return {
        "driver_id": req.driver_id,
        "risk_score": result["risk_score"],
        "premium": premium,
    }

@app.get("/debug/probs")
def debug_probs(limit: int = 50):
    # build per-driver features from CSV only, return a sample of predicted probabilities
    drivers = sorted(app.state.csv_df["driver_id"].dropna().unique().tolist())
    out = []
    for d in drivers[:limit]:
        df = _events_df(d, source="csv")
        if df.empty:
            continue
        feats = to_features(df)
        row = feats.loc[feats["driver_id"] == d]
        if row.empty:
            continue
        p = float(app.state.model.predict_proba(row[RISK_FEATURES].values)[0, 1])
        out.append({"driver_id": d, "prob": p, "score": risk_score(p)})
    return {"n": len(out), "probs": out}
