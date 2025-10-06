from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import json

from pricing import risk_score, premium_from_score
from schemas import Event, PriceReq

app = FastAPI(title="Telematics UBI API", version="1.1.0")

# Allow local Streamlit & localhost by default
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8502",
    "http://127.0.0.1:8502",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Storage & paths
# -----------------------------
EVENTS: List[Dict[str, Any]] = []  # in-memory demo store

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "ingested_events.csv"
NDJSON_PATH = DATA_DIR / "events.ndjson"

# -----------------------------
# Utilities
# -----------------------------
def _df_from_events_list(events: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(events) if events else pd.DataFrame()

def _append_events(df: pd.DataFrame) -> int:
    """Append rows from a dataframe into in-memory EVENTS."""
    if df.empty:
        return 0
    rows = df.to_dict(orient="records")
    EVENTS.extend(rows)
    return len(rows)

# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    df = _df_from_events_list(EVENTS)
    return {
        "ok": True,
        "csv_rows": int(df.shape[0]),
        "in_memory_events": int(len(EVENTS)),
        "model_loaded": True,  # you can wire a real check if needed
    }

# -----------------------------
# Bootstrap loader
# -----------------------------
class BootstrapReq(BaseModel):
    generate_if_empty: bool = False
    count: int = 500  # used only if we must generate

@app.post("/bootstrap")
def bootstrap(req: BootstrapReq):
    """
    Load data into memory following this precedence:
      1) data/ingested_events.csv (if exists & non-empty)
      2) data/events.ndjson      (if exists & non-empty)
      3) (optional) generate 'count' synthetic rows IF generate_if_empty=True
    Returns number of rows loaded and source.
    """
    global EVENTS
    loaded = 0
    source = None

    # 1) CSV
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        try:
            df = pd.read_csv(CSV_PATH)
            loaded = _append_events(df)
            source = str(CSV_PATH)
            return {"ok": True, "loaded": loaded, "source": source}
        except Exception as e:
            raise HTTPException(400, f"Failed reading CSV: {e}")

    # 2) NDJSON
    if NDJSON_PATH.exists() and NDJSON_PATH.stat().st_size > 0:
        try:
            with open(NDJSON_PATH, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            df = pd.DataFrame(rows)
            loaded = _append_events(df)
            source = str(NDJSON_PATH)
            return {"ok": True, "loaded": loaded, "source": source}
        except Exception as e:
            raise HTTPException(400, f"Failed reading NDJSON: {e}")

    # 3) Optional generate (only if explicitly asked)
    if req.generate_if_empty:
        try:
            # very simple synthetic fallback
            import numpy as np
            from datetime import datetime, timedelta

            drivers = [f"D_{i:03d}" for i in range(50)]
            now = datetime.utcnow()
            rows = []
            rng = np.random.default_rng(42)
            for _ in range(req.count):
                d = rng.choice(drivers)
                ts = now - timedelta(minutes=int(rng.integers(0, 60*24)))
                rows.append({
                    "driver_id": d,
                    "ts": ts.isoformat(),
                    "lat": 43.1 + float(rng.normal(0, 0.05)),
                    "lon": -78.8 + float(rng.normal(0, 0.05)),
                    "speed_kph": float(rng.uniform(0, 130)),
                    "accel_mps2": float(rng.normal(0.4, 1.2)),
                    "brake_mps2": float(rng.normal(0.3, 1.0)),
                    "heading_deg": float(rng.uniform(0, 360)),
                    "odometer_km": float(rng.integers(10000, 40000)),
                    "phone_use": int(rng.integers(0, 2)),
                    "road_type": "city",
                    "weather_code": "CLR",
                    "hard_brake": 0,
                })
            df = pd.DataFrame(rows)
            loaded = _append_events(df)
            source = "generated"
            return {"ok": True, "loaded": loaded, "source": source}
        except Exception as e:
            raise HTTPException(400, f"Failed generating fallback data: {e}")

    # If we got here, nothing to load
    return {"ok": False, "loaded": 0, "source": None, "message": "No CSV/NDJSON found; generation disabled."}

# -----------------------------
# Optional: expose existing data
# -----------------------------
@app.get("/data")
def get_data(limit: Optional[int] = 1000):
    df = _df_from_events_list(EVENTS)
    if df.empty:
        return {"count": 0, "events": []}
    if limit and limit > 0:
        df = df.iloc[: int(limit)]
    return {"count": int(df.shape[0]), "events": df.to_dict(orient="records")}

# -----------------------------
# Ingest individual events (kept for compatibility)
# -----------------------------
@app.post("/ingest")
def ingest_event(event: Event):
    EVENTS.append(event.dict())
    return {"ok": True, "count": len(EVENTS)}

# -----------------------------
# Driver endpoints
# -----------------------------
@app.get("/driver/{driver_id}/events")
def get_driver_events(driver_id: str, start: Optional[str] = None, end: Optional[str] = None, limit: int = 0):
    df = _df_from_events_list(EVENTS)
    if df.empty:
        return {"count": 0, "events": []}
    if "driver_id" in df.columns:
        df = df[df["driver_id"] == driver_id].copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        if start:
            df = df[df["ts"] >= pd.to_datetime(start, errors="coerce")]
        if end:
            df = df[df["ts"] <= pd.to_datetime(end, errors="coerce")]
        df = df.sort_values("ts")
    if limit and limit > 0:
        df = df.head(limit)
    return {"count": int(df.shape[0]), "events": df.to_dict(orient="records")}

@app.get("/driver/{driver_id}/features")
def driver_features(driver_id: str):
    df = _df_from_events_list(EVENTS)
    if df.empty:
        return {"features": {}}
    dfd = df[df.get("driver_id") == driver_id].copy()
    if dfd.empty:
        return {"features": {}}

    feats = {
        "driver_id": driver_id,
        "avg_speed": float(dfd["speed_kph"].mean()),
        "p95_speed": float(dfd["speed_kph"].quantile(0.95)),
        "speed_std": float(dfd["speed_kph"].std()),
        "hard_brakes_per_100km": float((dfd["brake_mps2"] > 3).sum() / max(len(dfd), 1) * 100.0),
        "hard_accels_per_100km": float((dfd["accel_mps2"] > 3).sum() / max(len(dfd), 1) * 100.0),
        "phone_use_min_per_100km": float(dfd["phone_use"].astype(float).mean() * 100.0),
        "night_km_share": 0.3,
        "rush_km_share": 0.4,
        "urban_km_share": 0.5,
    }
    return {"features": feats}

@app.get("/driver/{driver_id}/score")
def driver_score(driver_id: str):
    df = _df_from_events_list(EVENTS)
    dfd = df[df.get("driver_id") == driver_id].copy()
    if dfd.empty:
        return {"message": "No events for this driver yet."}
    prob = 0.05 + 0.002 * float(dfd["speed_kph"].mean())
    prob = min(max(prob, 0.01), 0.99)
    score = risk_score(prob)
    return {"risk_score": score, "prob_incident": prob}

@app.post("/price")
def compute_price(req: PriceReq):
    s = driver_score(req.driver_id)
    if "risk_score" not in s:
        raise HTTPException(400, "No score yet.")
    new_price = premium_from_score(req.base_premium, s["risk_score"])
    return {"premium": new_price}

@app.get("/")
def root():
    return {"message": "Telematics UBI API running", "events_in_memory": len(EVENTS)}
