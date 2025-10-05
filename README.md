# Telematics UBI – Starter POC

## Overview
Local-first proof of concept for a telematics-driven usage-based insurance (PAYD/PHYD) system:
- Ingest simulated GPS/accelerometer-like events
- Engineer behavior features
- Score driver risk (0–100) from calibrated incident probability
- Price premiums dynamically with guardrails
- Expose APIs (FastAPI) and a simple dashboard (Streamlit)

> NOTE: This repo is a scaffold. You must implement/extend the stubs per the assignment rules and your own work.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/simulate_telematics.py        # writes data/events.ndjson
python -m src.model --train data/events.ndjson  # trains baseline & saves models/*
uvicorn src.api:app --reload --port 8000  # API at http://localhost:8000
streamlit run src/dashboard.py            # UI at http://localhost:8501
```

## API
- `POST /ingest` — ingest a single event
- `GET /driver/{driver_id}/score` — return probability, RiskScore
- `POST /price` — input: driver_id, base_premium → output: adjusted premium

See inline curl examples in `src/api.py`.

## Structure
```
/src, /models, /docs, /bin, /data
```
Follow the assignment to submit a `.zip` named `Lastname_Firstname_ProjectName.zip` and include a ≤5‑min video.

## Credits
Use open-source responsibly; cite any datasets, models, or assets you choose to include.
