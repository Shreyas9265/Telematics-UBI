# src/dashboard.py
import os
import json
import random
from datetime import datetime

import requests
import pandas as pd
import streamlit as st

# -------------------------------------------------
# SAFE API base resolver: ENV -> secrets -> default
# -------------------------------------------------
def get_api_base() -> str:
    env = os.getenv("API_BASE")
    if env:
        return env
    try:
        # Only use secrets if it actually exists
        return st.secrets["API_BASE"]  # will raise if secrets.toml is missing
    except Exception:
        return "http://127.0.0.1:8000"  # fallback

API = get_api_base()

st.set_page_config(page_title="Telematics UBI – Driver Dashboard", layout="wide")
st.title("Telematics UBI – Driver Dashboard")

# Small helper wrappers
def api_get(path, **kwargs):
    return requests.get(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)

def api_post(path, **kwargs):
    return requests.post(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)

# Health banner
try:
    h = api_get("/health", timeout=3).json()
    ok = h.get("ok", False)
    if ok:
        st.success(
            f"Connected to API at {API}  |  "
            f"Model loaded: {h.get('model_loaded')}  |  "
            f"In-memory events: {h.get('in_memory_events')}  |  "
            f"CSV rows: {h.get('csv_rows')}"
        )
    else:
        st.warning(f"API reachable but not healthy: {h}")
except Exception as e:
    st.error(f"Cannot reach API at {API}. Error: {e}")

# Inputs
driver_id = st.text_input("Driver ID", "D_000")
base_premium = st.number_input("Base Premium ($/mo)", min_value=0.0, value=120.0, step=1.0)

tabs = st.tabs(["Overview", "Events", "Features"])

# -----------------------------
# OVERVIEW
# -----------------------------
with tabs[0]:
    c1, c2 = st.columns(2)
    if c1.button("Load 500 sample events"):
        try:
            lines = open("data/events.ndjson").read().splitlines()
            for line in random.sample(lines, k=min(500, len(lines))):
                api_post("/ingest", json=json.loads(line))
            st.success("Loaded 500 random events into the API.")
        except Exception as e:
            st.error(f"Load failed: {e}")

    if c2.button("Refresh"):
        pass  # values are pulled each render below

    # Pull score and price
    try:
        s = api_get(f"/driver/{driver_id}/score").json()
        if "prob_incident" in s:
            p = api_post("/price", json={"driver_id": driver_id, "base_premium": base_premium}).json()
            a, b, c = st.columns(3)
            a.metric("Risk Score", s["risk_score"])
            b.metric("Incident Probability", f"{s['prob_incident']*100:.2f}%")
            c.metric("Estimated Premium", f"${p.get('premium', 0):.1f}")
        else:
            st.info(s.get("message", "No data yet. Load events first."))
    except Exception as e:
        st.error(f"API error: {e}")

# -----------------------------
# EVENTS
# -----------------------------
with tabs[1]:
    st.subheader("Driver Events")
    colA, colB, colC = st.columns([1, 1, 1])
    start = colA.text_input("Start (ISO8601, optional)", value="")
    end   = colB.text_input("End (ISO8601, optional)", value="")
    limit = colC.number_input("Max rows (0 = all)", min_value=0, value=0, step=100)

    params = {}
    if start.strip():
        params["start"] = start.strip()
    if end.strip():
        params["end"] = end.strip()
    params["limit"] = limit

    try:
        r = api_get(f"/driver/{driver_id}/events", params=params).json()
        df = pd.DataFrame(r.get("events", []))
        st.caption(f"Returned rows: {r.get('count', 0)}")
        if df.empty:
            st.info("No events for this driver/time range.")
        else:
            # tidy
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                df = df.sort_values("ts")
            st.dataframe(df, use_container_width=True, height=420)

            # Download
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{driver_id}_events.csv",
                mime="text/csv"
            )

            # Quick charts (only if columns exist)
            st.markdown("### Quick Charts")
            if "ts" in df.columns and "speed_kph" in df.columns:
                st.line_chart(df.set_index("ts")["speed_kph"], height=220)
            if "ts" in df.columns and "hard_brake" in df.columns:
                hb = df.dropna(subset=["ts"]).set_index("ts")["hard_brake"].resample("5min").sum()
                st.bar_chart(hb, height=220)
    except Exception as e:
        st.error(f"API error: {e}")

# -----------------------------
# FEATURES
# -----------------------------
with tabs[2]:
    st.subheader("Engineered Features")
    try:
        r = api_get(f"/driver/{driver_id}/features").json()
        if "features" in r:
            feats = r["features"].copy()

            # keep driver_id separate so the table column 'value' stays numeric
            driver_row = feats.pop("driver_id", driver_id)

            # turn dict -> tidy table with consistent dtypes
            fdf = pd.DataFrame(list(feats.items()), columns=["feature", "value"])

            # ensure numbers are numeric (prevents Arrow mixed-type issues)
            if pd.api.types.is_numeric_dtype(fdf["value"]) is False:
                # this will convert numbers to float and non-numbers to NaN
                fdf["value"] = pd.to_numeric(fdf["value"], errors="coerce")

            st.caption(f"Driver: {driver_row}")
            st.dataframe(fdf, width="stretch")   # <- new Streamlit arg
        else:
            st.info(r.get("message", "No features available for this driver yet."))
    except Exception as e:
        st.error(f"API error: {e}")
