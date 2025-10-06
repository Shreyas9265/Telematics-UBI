# # src/api.py
# from __future__ import annotations

# import os
# import csv
# import math
# import json
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Optional, Any

# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from joblib import load as joblib_load


# # -----------------------------------------------------------------------------
# # Configuration
# # -----------------------------------------------------------------------------
# ROOT = Path(__file__).resolve().parents[1]          # project root
# DATA_DIR = ROOT / "data"
# MODELS_DIR = ROOT / "models"
# DATA_DIR.mkdir(parents=True, exist_ok=True)

# # CSV sink for all ingested events (append-only)
# CSV_PATH = DATA_DIR / "ingested_events.csv"

# # Optional API key (not strictly required by default)
# API_KEY = os.getenv("API_KEY", "dev-key")

# # Known tabular schema for CSV (new keys will be ignored to keep header stable)
# CSV_FIELDS = [
#     "driver_id", "ts",
#     "lat", "lon",
#     "speed_kph", "accel_mps2", "brake_mps2",
#     "heading_deg", "odometer_km",
#     "phone_use", "road_type", "weather_code",
#     "hard_brake"
# ]


# # -----------------------------------------------------------------------------
# # Model loading (optional)
# # -----------------------------------------------------------------------------
# MODEL_PATH = MODELS_DIR / "model.joblib"
# _model = None
# _model_loaded = False

# if MODEL_PATH.exists():
#     try:
#         _model = joblib_load(MODEL_PATH)
#         _model_loaded = True
#     except Exception:
#         # Model is optional; we degrade gracefully to a heuristic below
#         _model = None
#         _model_loaded = False


# # -----------------------------------------------------------------------------
# # In-memory store
# # -----------------------------------------------------------------------------
# # Keeps recent events this process has seen (clears when API restarts)
# events_by_driver: Dict[str, List[Dict[str, Any]]] = {}


# # -----------------------------------------------------------------------------
# # Pydantic schema
# # -----------------------------------------------------------------------------
# class TelemetryEvent(BaseModel):
#     driver_id: str = Field(..., description="Driver identifier like D_021")
#     ts: datetime = Field(..., description="Timestamp in ISO8601")
#     lat: Optional[float] = None
#     lon: Optional[float] = None
#     speed_kph: Optional[float] = None
#     accel_mps2: Optional[float] = None
#     brake_mps2: Optional[float] = None
#     heading_deg: Optional[float] = None
#     odometer_km: Optional[float] = None
#     phone_use: Optional[int] = Field(default=0, description="0/1 indicator")
#     road_type: Optional[str] = Field(default=None, description="city|highway|rural|...")
#     weather_code: Optional[str] = None
#     hard_brake: Optional[int] = Field(default=0, description="0/1 indicator")


# # -----------------------------------------------------------------------------
# # App
# # -----------------------------------------------------------------------------
# app = FastAPI(title="Telematics UBI API", version="0.1.0")

# # CORS for local Streamlit / browsers
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # -----------------------------------------------------------------------------
# # Helpers
# # -----------------------------------------------------------------------------
# def _append_to_memory(e: Dict[str, Any]) -> None:
#     did = e["driver_id"]
#     events_by_driver.setdefault(did, []).append(e)


# def _ensure_csv_header(path: Path) -> None:
#     """Create CSV file with header if it doesn't exist."""
#     if not path.exists():
#         with path.open("w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
#             writer.writeheader()


# def _append_to_csv(e: Dict[str, Any]) -> None:
#     """Append a single event to the CSV sink with a fixed schema."""
#     _ensure_csv_header(CSV_PATH)
#     row = {k: e.get(k, None) for k in CSV_FIELDS}
#     # Render datetimes as ISO strings
#     if isinstance(row.get("ts"), datetime):
#         row["ts"] = row["ts"].isoformat()
#     with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
#         writer.writerow(row)


# def _events_query(
#     driver_id: str,
#     start: Optional[datetime],
#     end: Optional[datetime],
# ) -> List[Dict[str, Any]]:
#     """Return filtered in-memory events for driver."""
#     items = events_by_driver.get(driver_id, [])
#     if start:
#         items = [e for e in items if e.get("ts") and e["ts"] >= start]
#     if end:
#         items = [e for e in items if e.get("ts") and e["ts"] < end]
#     # Always sort by time
#     items = sorted(items, key=lambda x: x.get("ts") or datetime.min)
#     return items


# def _p95(x: List[float]) -> float:
#     if not x:
#         return 0.0
#     return float(np.nanpercentile(np.array(x, dtype=float), 95))


# def _safe_mean(x: List[float]) -> float:
#     if not x:
#         return 0.0
#     arr = np.array(x, dtype=float)
#     return float(np.nanmean(arr))


# def _safe_std(x: List[float]) -> float:
#     if not x:
#         return 0.0
#     arr = np.array(x, dtype=float)
#     return float(np.nanstd(arr))


# def _compute_features(driver_id: str, evs: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Very small, robust feature set that doesn't assume every field is present.
#     """
#     if not evs:
#         return {"driver_id": driver_id}

#     # numeric traces (ignore None)
#     spd = [e["speed_kph"] for e in evs if e.get("speed_kph") is not None]
#     acc = [e["accel_mps2"] for e in evs if e.get("accel_mps2") is not None]
#     brk = [e["brake_mps2"] for e in evs if e.get("brake_mps2") is not None]
#     odo = [e["odometer_km"] for e in evs if e.get("odometer_km") is not None]

#     # distance proxy from odometer if available
#     km_total = 0.0
#     if len(odo) >= 2:
#         odos = [x for x in odo if isinstance(x, (int, float))]
#         if odos:
#             km_total = max(0.0, float(max(odos) - min(odos)))

#     # time-of-day heuristics
#     def _hour(e):
#         t = e.get("ts")
#         return t.hour if isinstance(t, datetime) else None

#     hours = [h for h in (_hour(e) for e in evs) if h is not None]
#     night_fraction = 0.0
#     rush_fraction = 0.0
#     if hours:
#         night_fraction = sum(1 for h in hours if (h >= 22 or h < 6)) / len(hours)
#         rush_fraction = sum(1 for h in hours if (7 <= h <= 9) or (16 <= h <= 19)) / len(hours)

#     # phone/road usage
#     phone_use = [e.get("phone_use", 0) for e in evs]
#     road_types = [e.get("road_type") for e in evs]
#     urban_fraction = 0.0
#     if road_types:
#         urban_fraction = sum(1 for r in road_types if str(r).lower() in {"city", "urban", "residential"}) / len(road_types)

#     # events flagged as hard brakes
#     hard_brakes = [e.get("hard_brake", 0) for e in evs]
#     hard_brake_count = sum(int(bool(x)) for x in hard_brakes)

#     # derive per-100km rates; guard against km_total ~ 0
#     denom = max(km_total, 1e-6)
#     per_100km = lambda n: (n / denom) * 100.0  # noqa: E731

#     feats = {
#         "driver_id": driver_id,
#         "avg_speed": _safe_mean(spd),
#         "p95_speed": _p95(spd),
#         "speed_std": _safe_std(spd),
#         "hard_brakes_per_100km": per_100km(hard_brake_count),
#         "hard_accel_per_100km": per_100km(sum(1 for a in acc if (a is not None and a > 2.5))),
#         "night_km_share": float(night_fraction),
#         "rush_km_share": float(rush_fraction),
#         "phone_use_min_per_100km": per_100km(sum(1 for p in phone_use if p)),
#         "urban_km_share": float(urban_fraction),
#     }
#     return feats


# def _predict_incident_probability(feats: Dict[str, Any]) -> float:
#     """
#     If a trained model is available, use it.
#     Otherwise apply a light heuristic that correlates with risky driving.
#     """
#     # Model path: expect something like a sklearn Pipeline
#     if _model_loaded and _model is not None:
#         try:
#             X = pd.DataFrame([feats]).drop(columns=["driver_id"], errors="ignore")
#             # Many pipelines need all numeric; coerce
#             for c in X.columns:
#                 X[c] = pd.to_numeric(X[c], errors="coerce")
#             X = X.fillna(0.0)
#             proba = float(_model.predict_proba(X)[:, 1][0])
#             return float(np.clip(proba, 0.0, 1.0))
#         except Exception:
#             pass  # fall through to heuristic

#     # Heuristic: weighted logistic on a few features
#     x = 0.0
#     x += 0.020 * float(feats.get("avg_speed", 0.0))
#     x += 0.030 * float(feats.get("speed_std", 0.0))
#     x += 0.040 * float(feats.get("hard_brakes_per_100km", 0.0))
#     x += 0.030 * float(feats.get("hard_accel_per_100km", 0.0))
#     x += 0.800 * float(feats.get("night_km_share", 0.0))
#     x += 0.600 * float(feats.get("rush_km_share", 0.0))
#     x += 0.020 * float(feats.get("phone_use_min_per_100km", 0.0))
#     x += 0.300 * float(feats.get("urban_km_share", 0.0))
#     # logistic
#     proba = 1.0 / (1.0 + math.exp(-x))
#     return float(np.clip(proba, 0.0, 1.0))


# def _score_from_probability(p: float) -> int:
#     # simple 0..100 mapping
#     return int(round(float(np.clip(p, 0.0, 1.0)) * 100.0))


# def _guardrailed_premium(base: float, prob: float) -> float:
#     """
#     Translate probability into a premium multiplier with guardrails:
#       - cap surcharge at +35%,
#       - allow mild discounts down to -15%.
#     """
#     raw = prob * 0.5  # 50% of probability -> surcharge
#     adj = float(np.clip(raw, -0.15, 0.35))
#     return round(base * (1.0 + adj), 2)


# # -----------------------------------------------------------------------------
# # Endpoints
# # -----------------------------------------------------------------------------
# @app.get("/health")
# def health():
#     rows = 0
#     if CSV_PATH.exists():
#         try:
#             with CSV_PATH.open("r", encoding="utf-8") as f:
#                 # subtract header line
#                 rows = max(sum(1 for _ in f) - 1, 0)
#         except Exception:
#             rows = 0
#     return {
#         "ok": True,
#         "model_loaded": _model_loaded,
#         "in_memory_events": sum(len(v) for v in events_by_driver.values()),
#         "csv_rows": rows,
#     }


# @app.post("/ingest")
# def ingest(event: TelemetryEvent):
#     e = event.dict()
#     _append_to_memory(e)
#     try:
#         _append_to_csv(e)
#     except Exception as ex:
#         # CSV persistence should not block ingestion
#         return {"status": "accepted", "csv_error": str(ex)}
#     return {"status": "accepted"}


# @app.get("/driver/{driver_id}/events")
# def driver_events(
#     driver_id: str,
#     start: Optional[datetime] = Query(default=None),
#     end: Optional[datetime] = Query(default=None),
#     limit: int = Query(default=0, ge=0, description="0 = all"),
# ):
#     items = _events_query(driver_id, start, end)
#     if limit and limit > 0:
#         items = items[:limit]
#     # serialize datetimes for JSON
#     out = []
#     for e in items:
#         e2 = dict(e)
#         if isinstance(e2.get("ts"), datetime):
#             e2["ts"] = e2["ts"].isoformat()
#         out.append(e2)
#     return {"driver_id": driver_id, "count": len(out), "events": out}


# @app.get("/driver/{driver_id}/features")
# def driver_features(driver_id: str):
#     evs = events_by_driver.get(driver_id, [])
#     if not evs:
#         return {"driver_id": driver_id, "message": "No events for this driver. Ingest events first."}
#     feats = _compute_features(driver_id, evs)
#     return {"driver_id": driver_id, "features": feats}


# @app.get("/driver/{driver_id}/score")
# def driver_score(driver_id: str):
#     evs = events_by_driver.get(driver_id, [])
#     if not evs:
#         return {"driver_id": driver_id, "message": "No events for this driver. Ingest events first."}
#     feats = _compute_features(driver_id, evs)
#     prob = _predict_incident_probability(feats)
#     return {"driver_id": driver_id, "prob_incident": prob, "risk_score": _score_from_probability(prob)}


# class PriceReq(BaseModel):
#     driver_id: str
#     base_premium: float = Field(ge=0.0)


# @app.post("/price")
# def price(req: PriceReq):
#     evs = events_by_driver.get(req.driver_id, [])
#     if not evs:
#         raise HTTPException(status_code=400, detail="No events for this driver. Ingest events first.")
#     feats = _compute_features(req.driver_id, evs)
#     prob = _predict_incident_probability(feats)
#     premium = _guardrailed_premium(req.base_premium, prob)
#     return {
#         "driver_id": req.driver_id,
#         "risk_score": _score_from_probability(prob),
#         "premium": premium,
#         "prob_incident": prob,
#     }
# from fastapi.middleware.cors import CORSMiddleware

# origins = [
#     "https://telematics-ubi.onrender.com",
#     "http://localhost:8501",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# package-relative imports (run server as:  uvicorn src.api:app --reload --port 8000)
from .pricing import risk_score, premium_from_score
from .schemas import Event, PriceReq


# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Telematics UBI API", version="1.2.0")

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


# -----------------------------------------------------------------------------
# Storage & Paths
# -----------------------------------------------------------------------------
EVENTS: List[Dict[str, Any]] = []              # in-memory store the dashboard reads
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_PATH = DATA_DIR / "ingested_events.csv"
NDJSON_PATH = DATA_DIR / "events.ndjson"       # optional fallback

DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _df_from_events_list(events: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(events) if events else pd.DataFrame()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column names so downstream code stays stable."""
    if df.empty:
        return df

    # unify likely alternatives to driver_id
    for cand in ["driver", "driverID", "DriverId", "Driver", "driverId"]:
        if cand in df.columns and "driver_id" not in df.columns:
            df = df.rename(columns={cand: "driver_id"})
            break

    # ensure timestamps parseable
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    return df


def _append_events(df: pd.DataFrame) -> int:
    """Append rows from a DataFrame into in-memory EVENTS."""
    if df is None or df.empty:
        return 0
    df = _normalize_columns(df.copy())
    EVENTS.extend(df.to_dict(orient="records"))
    return int(len(df))


def _read_csv() -> pd.DataFrame:
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        try:
            df = pd.read_csv(CSV_PATH)
            return _normalize_columns(df)
        except Exception as e:
            raise HTTPException(400, f"Reading CSV failed: {e}")
    return pd.DataFrame()


def _filter_by_driver(df: pd.DataFrame, driver_id: str) -> pd.DataFrame:
    """Safe filter; returns empty df if column missing."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "driver_id" not in df.columns:
        return pd.DataFrame()
    return df[df["driver_id"].astype(str) == str(driver_id)].copy()


def _csv_row_count() -> int:
    try:
        if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
            return 0
        # fast-ish: read just one column to count
        return int(pd.read_csv(CSV_PATH, usecols=[0]).shape[0])
    except Exception:
        return 0


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "in_memory_events": int(len(EVENTS)),
        "csv_rows": _csv_row_count(),
        "source": str(CSV_PATH) if CSV_PATH.exists() else None,
        "model_loaded": True,   # hook your real model status here if needed
    }


# -----------------------------------------------------------------------------
# CSV / Memory Control Endpoints
# -----------------------------------------------------------------------------
class LoadCSVReq(BaseModel):
    clear: bool = True  # if True, reset in-memory to CSV contents


@app.post("/load_csv")
def load_csv(req: LoadCSVReq):
    df = _read_csv()
    global EVENTS
    if req.clear:
        EVENTS = []
    added = _append_events(df)
    return {
        "ok": True,
        "loaded": added,
        "in_memory": len(EVENTS),
        "csv_rows": _csv_row_count(),
        "source": str(CSV_PATH),
    }


class SimAppendReq(BaseModel):
    n: int = 500


@app.post("/simulate_and_append")
def simulate_and_append(req: SimAppendReq):
    """Generate N synthetic rows, append to CSV, and extend memory."""
    n = max(1, int(req.n))
    rng = np.random.default_rng(42)
    drivers = [f"D_{i:03d}" for i in range(200)]
    now = datetime.utcnow()

    rows = []
    for _ in range(n):
        d = rng.choice(drivers)
        ts = now - timedelta(minutes=int(rng.integers(0, 60 * 24)))
        rows.append(
            {
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
                "road_type": rng.choice(["city", "highway", "urban"]),
                "weather_code": rng.choice(["CLR", "RA", "OVC"]),
                "hard_brake": int(rng.integers(0, 2)),
            }
        )

    df_new = _normalize_columns(pd.DataFrame(rows))

    # append to CSV on disk (write header iff file missing)
    header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    df_new.to_csv(CSV_PATH, mode="a", header=header, index=False)

    # extend memory too
    added = _append_events(df_new)

    return {
        "ok": True,
        "added": int(added),
        "csv_rows": _csv_row_count(),
        "in_memory": len(EVENTS),
        "source": str(CSV_PATH),
    }


class PersistReq(BaseModel):
    dedupe: bool = True
    overwrite: bool = True
    subset: Optional[List[str]] = None  # e.g. ["driver_id", "ts"]


@app.post("/persist_csv")
def persist_csv(req: PersistReq):
    """Persist current in-memory events to CSV (overwrite or append)."""
    df = _df_from_events_list(EVENTS)
    if df.empty:
        raise HTTPException(400, "No in-memory events to persist.")

    df = _normalize_columns(df)

    if req.dedupe:
        subset = req.subset or [c for c in ["driver_id", "ts"] if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset, keep="last")

    if req.overwrite:
        df.to_csv(CSV_PATH, index=False)
    else:
        header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
        df.to_csv(CSV_PATH, mode="a", header=header, index=False)

    return {
        "ok": True,
        "csv_rows": _csv_row_count(),
        "in_memory": len(EVENTS),
        "source": str(CSV_PATH),
    }


@app.post("/clear")
def clear_memory():
    global EVENTS
    EVENTS = []
    return {"ok": True, "in_memory": 0}


# -----------------------------------------------------------------------------
# Optional bootstrap (CSV -> NDJSON -> generate)
# -----------------------------------------------------------------------------
class BootstrapReq(BaseModel):
    generate_if_empty: bool = False
    count: int = 500


@app.post("/bootstrap")
def bootstrap(req: BootstrapReq):
    """Try CSV, then NDJSON, then (optionally) generate synthetic data."""
    global EVENTS
    loaded, source = 0, None

    # 1) CSV
    df = _read_csv()
    if not df.empty:
        EVENTS = []
        loaded = _append_events(df)
        source = str(CSV_PATH)
        return {"ok": True, "loaded": loaded, "source": source}

    # 2) NDJSON
    if NDJSON_PATH.exists() and NDJSON_PATH.stat().st_size > 0:
        try:
            with open(NDJSON_PATH, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            df = _normalize_columns(pd.DataFrame(rows))
            EVENTS = []
            loaded = _append_events(df)
            source = str(NDJSON_PATH)
            return {"ok": True, "loaded": loaded, "source": source}
        except Exception as e:
            raise HTTPException(400, f"Reading NDJSON failed: {e}")

    # 3) Optional generate
    if req.generate_if_empty:
        return simulate_and_append(SimAppendReq(n=req.count))

    return {"ok": False, "loaded": 0, "source": None, "message": "No CSV/NDJSON found"}


# -----------------------------------------------------------------------------
# Generic Data endpoints
# -----------------------------------------------------------------------------
@app.get("/data")
def get_data(limit: Optional[int] = 1000):
    df = _df_from_events_list(EVENTS)
    if df.empty:
        return {"count": 0, "events": []}
    if limit and limit > 0:
        df = df.head(int(limit))
    return {"count": int(df.shape[0]), "events": df.to_dict(orient="records")}


@app.post("/ingest")
def ingest_event(event: Event):
    EVENTS.append(event.dict())
    return {"ok": True, "count": len(EVENTS)}


# -----------------------------------------------------------------------------
# Driver endpoints
# -----------------------------------------------------------------------------
@app.get("/driver/{driver_id}/events")
def get_driver_events(
    driver_id: str, start: Optional[str] = None, end: Optional[str] = None, limit: int = 0
):
    df = _df_from_events_list(EVENTS)
    dfd = _filter_by_driver(df, driver_id)
    if dfd.empty:
        return {"count": 0, "events": []}

    if "ts" in dfd.columns:
        if start:
            dfd = dfd[dfd["ts"] >= pd.to_datetime(start, errors="coerce")]
        if end:
            dfd = dfd[dfd["ts"] <= pd.to_datetime(end, errors="coerce")]
        dfd = dfd.sort_values("ts")

    if limit and limit > 0:
        dfd = dfd.head(limit)

    return {"count": int(dfd.shape[0]), "events": dfd.to_dict(orient="records")}


@app.get("/driver/{driver_id}/features")
def driver_features(driver_id: str):
    df = _df_from_events_list(EVENTS)
    dfd = _filter_by_driver(df, driver_id)
    if dfd.empty:
        return {"features": {}}

    feats = {
        "driver_id": driver_id,
        "avg_speed": float(dfd["speed_kph"].mean()) if "speed_kph" in dfd.columns else 0.0,
        "p95_speed": float(dfd["speed_kph"].quantile(0.95)) if "speed_kph" in dfd.columns else 0.0,
        "speed_std": float(dfd["speed_kph"].std()) if "speed_kph" in dfd.columns else 0.0,
        "hard_brakes_per_100km": float((dfd["brake_mps2"] > 3).sum() / max(len(dfd), 1) * 100.0)
        if "brake_mps2" in dfd.columns
        else 0.0,
        "hard_accels_per_100km": float((dfd["accel_mps2"] > 3).sum() / max(len(dfd), 1) * 100.0)
        if "accel_mps2" in dfd.columns
        else 0.0,
        "phone_use_min_per_100km": float(dfd["phone_use"].astype(float).mean() * 100.0)
        if "phone_use" in dfd.columns
        else 0.0,
        "night_km_share": 0.3,
        "rush_km_share": 0.4,
        "urban_km_share": 0.5,
    }
    return {"features": feats}


@app.get("/driver/{driver_id}/score")
def driver_score(driver_id: str):
    df = _df_from_events_list(EVENTS)
    dfd = _filter_by_driver(df, driver_id)
    if dfd.empty:
        return {"message": "No events for this driver yet."}

    mean_speed = float(dfd["speed_kph"].mean()) if "speed_kph" in dfd.columns else 0.0
    prob = 0.05 + 0.002 * mean_speed
    prob = min(max(prob, 0.01), 0.99)

    score = risk_score(prob)
    return {"risk_score": score, "prob_incident": prob}


# -----------------------------------------------------------------------------
# Pricing
# -----------------------------------------------------------------------------
@app.post("/price")
def compute_price(req: PriceReq):
    s = driver_score(req.driver_id)
    if "risk_score" not in s:
        raise HTTPException(400, "No score yet.")
    new_price = premium_from_score(req.base_premium, s["risk_score"])
    return {"premium": new_price}


# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Telematics UBI API running", "events_in_memory": len(EVENTS)}

