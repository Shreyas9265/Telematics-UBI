from fastapi import FastAPI
import pandas as pd
from .schemas import Event, PriceReq
from .features import to_features, RISK_FEATURES
from .model import load_model
from .pricing import risk_score, premium_from_score

app = FastAPI(title="Telematics UBI API")
app.state.events = []                 # in-memory list of events (dicts)
app.state.by_driver = defaultdict(int)
app.state.seen_keys = set()           # 'driver_id|ts' to avoid duplicates

MODEL = None
EVENTS = []  # Replace with DB in production

from pathlib import Path
import pandas as pd
from collections import defaultdict

CSV_PATH = Path("data/ingested.csv")
if CSV_PATH.exists():
    EVENTS = pd.read_csv(CSV_PATH).to_dict(orient="records")

@app.get("/")
def root(): return {"status":"ok","message":"Telematics UBI API. See /docs"}

@app.on_event("startup")
def load():
    global MODEL
    try:
        MODEL = load_model()
    except Exception as e:
        print("Model not found yet. Train first.", e)
        
@app.on_event("startup")
def _load_existing_csv_keys():
    CSV_PATH.parent.mkdir(exist_ok=True)
    if CSV_PATH.exists():
        # load in chunks so it scales
        for chunk in pd.read_csv(CSV_PATH, dtype=str, chunksize=200_000):
            # ensure columns exist; skip if malformed
            if "driver_id" in chunk.columns and "ts" in chunk.columns:
                keys = (chunk["driver_id"].astype(str) + "|" + chunk["ts"].astype(str))
                app.state.seen_keys.update(keys.tolist())

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest")
def ingest(ev: Event):
    EVENTS.append(ev.dict())
    return {"ok": True, "count": len(EVENTS)}

@app.get("/driver/{driver_id}/score")
def get_score(driver_id: str):
    if not EVENTS:
        return {"driver_id": driver_id, "message":"no data"}
    df = pd.DataFrame(EVENTS)
    ddf = df[df["driver_id"]==driver_id]
    if ddf.empty:
        return {"driver_id": driver_id, "message":"no data for driver"}
    feats = to_features(ddf)
    if feats.empty or MODEL is None:
        return {"driver_id": driver_id, "message":"model not ready"}
    x = feats[RISK_FEATURES].values
    prob = float(MODEL.predict_proba(x)[0,1])
    score = risk_score(prob)
    return {"driver_id": driver_id, "prob_incident": prob, "risk_score": score}

@app.post("/price")
def price(req: PriceReq):
    sc = get_score(req.driver_id)
    if "risk_score" not in sc:
        return {"message":"no score available"}
    premium = premium_from_score(req.base_premium, sc["risk_score"])
    return {"driver_id": req.driver_id, "risk_score": sc["risk_score"], "premium": premium}

def _append_event_csv(event: dict):
    """Append one event to data/ingested.csv with header on first write."""
    df = pd.DataFrame([event])
    write_header = not CSV_PATH.exists()
    # Keep column order stable if file exists
    if CSV_PATH.exists():
        existing_cols = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
        # align to existing cols (add any missing)
        for c in existing_cols:
            if c not in df.columns:
                df[c] = pd.NA
        # also keep any new columns at the end
        df = df[[*existing_cols, *[c for c in df.columns if c not in existing_cols]]]
    df.to_csv(CSV_PATH, mode="a", index=False, header=write_header)


# curl examples:
# curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"driver_id":"D_000","ts":"2025-10-03T12:00:00Z","lat":42.9,"lon":-78.8,"speed_kph":60,"accel_mps2":0.2,"brake_mps2":-0.4,"heading_deg":120,"odometer_km":1000.0,"phone_use":0,"road_type":"city","weather_code":"CLR"}'
# curl http://localhost:8000/driver/D_000/score
# curl -X POST http://localhost:8000/price -H "Content-Type: application/json" -d '{"driver_id":"D_000","base_premium":120}'
