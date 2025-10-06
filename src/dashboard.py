# # src/dashboard.py
# import os
# import json
# import random
# from datetime import datetime

# import requests
# import pandas as pd
# import numpy as np
# import streamlit as st


# # -------------------------------------------------
# # SAFE API base resolver: ENV -> secrets -> default
# # -------------------------------------------------
# def get_api_base() -> str:
#     env = os.getenv("API_BASE")
#     if env:
#         return env
#     try:
#         # Only use secrets if it actually exists
#         return st.secrets["API_BASE"]  # raises if secrets.toml missing
#     except Exception:
#         return "http://127.0.0.1:8000"  # local fallback


# API = get_api_base()
# EVENTS_FILE = os.getenv("EVENTS_FILE", "data/events.ndjson")  # local file for the “Load 500” button


# st.set_page_config(page_title="Telematics UBI – Driver Dashboard", layout="wide")
# st.title("Telematics UBI – Driver Dashboard")


# # Small helper wrappers
# def api_get(path, **kwargs):
#     return requests.get(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)


# def api_post(path, **kwargs):
#     return requests.post(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)


# # Health banner
# try:
#     h = api_get("/health", timeout=3).json()
#     ok = h.get("ok", False)
#     if ok:
#         st.success(
#             f"Connected to API at {API}  |  "
#             f"Model loaded: {h.get('model_loaded')}  |  "
#             f"In-memory events: {h.get('in_memory_events')}  |  "
#             f"CSV rows: {h.get('csv_rows')}"
#         )
#     else:
#         st.warning(f"API reachable but not healthy: {h}")
# except Exception as e:
#     st.error(f"Cannot reach API at {API}. Error: {e}")

# # Inputs
# driver_id = st.text_input("Driver ID", "D_000")
# base_premium = st.number_input("Base Premium ($/mo)", min_value=0.0, value=120.0, step=1.0)

# tabs = st.tabs(["Overview", "Events", "Features", "Clear"])  # <— added the new tab


# # Utility caches
# @st.cache_data(show_spinner=False)
# def _get_features(driver: str) -> dict | None:
#     try:
#         r = api_get(f"/driver/{driver}/features").json()
#         return r.get("features")
#     except Exception:
#         return None


# @st.cache_data(show_spinner=False)
# def _get_score(driver: str) -> dict | None:
#     try:
#         r = api_get(f"/driver/{driver}/score").json()
#         return r
#     except Exception:
#         return None


# @st.cache_data(show_spinner=False)
# def _get_events(driver: str, start: str | None, end: str | None, limit: int) -> pd.DataFrame:
#     params = {"limit": limit}
#     if start:
#         params["start"] = start
#     if end:
#         params["end"] = end
#     try:
#         r = api_get(f"/driver/{driver}/events", params=params).json()
#         df = pd.DataFrame(r.get("events", []))
#         if not df.empty and "ts" in df.columns:
#             df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
#             df = df.sort_values("ts")
#         return df
#     except Exception:
#         return pd.DataFrame()


# # -----------------------------
# # OVERVIEW
# # -----------------------------
# with tabs[3]:
#     c1, c2, c3 = st.columns([1, 1, 2])
#     if c1.button("Load 500 sample events"):
#         try:
#             lines = open(EVENTS_FILE, "r", encoding="utf-8").read().splitlines()
#             for line in random.sample(lines, k=min(500, len(lines))):
#                 api_post("/ingest", json=json.loads(line))
#             st.success(f"Loaded 500 random events from {EVENTS_FILE} into the API.")
#             _get_events.clear()  # invalidate cache
#             _get_features.clear()
#             _get_score.clear()
#         except Exception as e:
#             st.error(f"Load failed: {e}")

#     if c2.button("Refresh"):
#         _get_events.clear()
#         _get_features.clear()
#         _get_score.clear()

#     # Pull score and price
#     try:
#         s = _get_score(driver_id)
#         if s and "prob_incident" in s:
#             p = api_post("/price", json={"driver_id": driver_id, "base_premium": base_premium}).json()
#             a, b, c = st.columns(3)
#             a.metric("Risk Score", s.get("risk_score"))
#             b.metric("Incident Probability", f"{s.get('prob_incident', 0)*100:.2f}%")
#             c.metric("Estimated Premium", f"${p.get('premium', 0):.1f}")
#         else:
#             st.info((s or {}).get("message", "No data yet. Load events first."))
#     except Exception as e:
#         st.error(f"API error: {e}")

    
# # -----------------------------
# # EVENTS
# # -----------------------------
# with tabs[1]:
#     st.subheader("Driver Events")
#     colA, colB, colC = st.columns([1, 1, 1])
#     start = colA.text_input("Start (ISO8601, optional)", value="")
#     end = colB.text_input("End (ISO8601, optional)", value="")
#     limit = colC.number_input("Max rows (0 = all)", min_value=0, value=0, step=100)

#     df = _get_events(driver_id, start.strip() or None, end.strip() or None, limit)
#     st.caption(f"Returned rows: {len(df)}")
#     if df.empty:
#         st.info("No events for this driver/time range.")
#     else:
#         st.dataframe(df, use_container_width=True, height=420)

#         # Download
#         csv = df.to_csv(index=False).encode()
#         st.download_button(
#             label="Download CSV",
#             data=csv,
#             file_name=f"{driver_id}_events.csv",
#             mime="text/csv",
#         )

#         # Quick charts (only if columns exist)
#         st.markdown("### Quick Charts")
#         if "ts" in df.columns and "speed_kph" in df.columns:
#             st.line_chart(df.set_index("ts")["speed_kph"], height=220)
#         if "ts" in df.columns and "hard_brake" in df.columns:
#             # hard_brake often 0/1; aggregate per 5 min
#             hb = (
#                 df.dropna(subset=["ts"])
#                 .set_index("ts")["hard_brake"]
#                 .astype(float)
#                 .resample("5min")
#                 .sum()
#             )
#             st.bar_chart(hb, height=220)


# # -----------------------------
# # FEATURES
# # -----------------------------
# with tabs[2]:
#     st.subheader("Engineered Features")
#     feats = _get_features(driver_id)
#     if feats:
#         feats_copy = dict(feats)  # do not mutate cached object
#         driver_row = feats_copy.pop("driver_id", driver_id)
#         fdf = pd.DataFrame(list(feats_copy.items()), columns=["feature", "value"])
#         # guarantee numeric where possible
#         fdf["value"] = pd.to_numeric(fdf["value"], errors="coerce")
#         st.caption(f"Driver: {driver_row}")
#         st.dataframe(fdf, use_container_width=True)
#     else:
#         st.info("No features available for this driver yet.")


# # -----------------------------
# # REQUIRED  (new tab)
# # -----------------------------
# with tabs[0]:
#     st.subheader("Required: Behavior, Scores & Premium Changes")

#     # 1) Summary cards
#     s = _get_score(driver_id) or {}
#     if "prob_incident" in s:
#         price_resp = api_post("/price", json={"driver_id": driver_id, "base_premium": base_premium}).json()
#         premium_now = price_resp.get("premium", 0.0)

#         k1, k2, k3 = st.columns(3)
#         k1.metric("Risk Score", s.get("risk_score", 0))
#         k2.metric("Incident Probability", f"{s.get('prob_incident', 0)*100:.2f}%")
#         k3.metric("Current Estimated Premium", f"${premium_now:.2f}")
#     else:
#         st.info("No score yet. Load/ingest events first.")
#         premium_now = None

#     st.divider()

#     # 2) Driving behavior snapshot (from features)
#     feats = _get_features(driver_id) or {}
#     if feats:
#         st.markdown("### Driving Behavior Snapshot")
#         cols = st.columns(4)
#         def _fmt(v, default="0"):
#             try:
#                 if v is None:
#                     return default
#                 if isinstance(v, (int, float)):
#                     return f"{v:.2f}"
#                 return str(v)
#             except Exception:
#                 return default

#         kmap = [
#             ("avg_speed", "Avg speed (kph)"),
#             ("p95_speed", "P95 speed (kph)"),
#             ("hard_brakes_per_100km", "Hard brakes /100km"),
#             ("hard_accels_per_100km", "Hard accels /100km"),
#             ("phone_use_min_per_100km", "Phone use min /100km"),
#             ("night_km_share", "Night km share"),
#             ("rush_km_share", "Rush-hour km share"),
#             ("urban_km_share", "Urban km share"),
#         ]
#         # display first 4, then next 4
#         for i, (key, label) in enumerate(kmap[:4]):
#             cols[i].metric(label, _fmt(feats.get(key)))
#         cols2 = st.columns(4)
#         for i, (key, label) in enumerate(kmap[4:]):
#             val = feats.get(key)
#             if "share" in key and isinstance(val, (int, float)):
#                 cols2[i].metric(label, f"{val*100:.1f}%")
#             else:
#                 cols2[i].metric(label, _fmt(val))

#     else:
#         st.info("No features to summarize driving behavior.")

#     st.divider()

#     # 3) Trends from events (if any)
#     st.markdown("### Recent Trends")
#     events_df = _get_events(driver_id, None, None, 0)
#     if not events_df.empty and "ts" in events_df.columns:
#         # daily aggregates
#         daily = events_df.copy()
#         daily["date"] = daily["ts"].dt.date

#         charts_cols = st.columns(2)

#         # Daily avg speed
#         if "speed_kph" in daily.columns:
#             g = daily.groupby("date")["speed_kph"].mean().rename("avg_speed_kph")
#             charts_cols[0].line_chart(g, height=240)

#         # Daily hard-brake counts
#         if "hard_brake" in daily.columns:
#             gb = daily.groupby("date")["hard_brake"].sum().rename("hard_brakes")
#             charts_cols[1].bar_chart(gb, height=240)
#     else:
#         st.info("No events to plot trends yet.")

#     st.divider()

#     # 4) Premium scenarios — how premium changes with base premium
#     st.markdown("### Premium Scenarios")
#     if premium_now is not None:
#         low, high = st.slider(
#             "Choose base premium range for scenario ($/mo)",
#             min_value=40, max_value=300, value=(80, 200), step=5
#         )
#         steps = st.number_input("Number of points", min_value=5, max_value=40, value=15, step=1)

#         bases = np.linspace(low, high, int(steps))
#         premiums = []
#         for b in bases:
#             try:
#                 r = api_post("/price", json={"driver_id": driver_id, "base_premium": float(b)}).json()
#                 premiums.append(r.get("premium", np.nan))
#             except Exception:
#                 premiums.append(np.nan)

#         scen = pd.DataFrame({"base_premium": bases, "estimated_premium": premiums}).dropna()
#         if not scen.empty:
#             st.line_chart(scen.set_index("base_premium")["estimated_premium"], height=260)
#             st.caption("Shows how the estimated premium would change as you adjust the base premium.")
#         else:
#             st.info("Could not compute premium scenarios.")
#     else:
#         st.info("Premium scenarios require a current score/price — ingest events first.")


# # src/dashboard.py
# # import os, json, random
# # from datetime import datetime

# # import requests
# # import pandas as pd
# # import numpy as np
# # import streamlit as st

# # # -----------------------------
# # # Resolve API base: ENV > secrets > default
# # # -----------------------------
# # def _api_base() -> str:
# #     env = os.getenv("API_BASE")
# #     if env:
# #         return env
# #     try:
# #         return st.secrets["API_BASE"]  # raises if secrets are absent
# #     except Exception:
# #         return "http://127.0.0.1:8000"

# # API = _api_base()

# # # Optional: where the sample events file lives
# # def _events_path() -> str:
# #     env = os.getenv("EVENTS_FILE")
# #     if env:
# #         return env
# #     try:
# #         return st.secrets["EVENTS_FILE"]
# #     except Exception:
# #         return "data/events.ndjson"

# # st.set_page_config(page_title="Telematics UBI – Driver Dashboard", layout="wide")
# # st.title("Telematics UBI – Driver Dashboard")

# # # -----------------------------
# # # Helpers
# # # -----------------------------
# # def api_get(path, **kwargs):
# #     return requests.get(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)

# # def api_post(path, **kwargs):
# #     return requests.post(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)

# # def refresh_health_banner():
# #     """Fetch health and paint banner + update counters."""
# #     try:
# #         h = api_get("/health", timeout=3).json()
# #         ok = h.get("ok", False)
# #         st.session_state["csv_rows"] = int(h.get("csv_rows", 0))
# #         st.session_state["in_memory_events"] = int(h.get("in_memory_events", 0))
# #         st.session_state["model_loaded"] = bool(h.get("model_loaded", False))
# #         if ok:
# #             st.success(
# #                 f"Connected to API at {API} | "
# #                 f"Model loaded: {st.session_state['model_loaded']} | "
# #                 f"In-memory events: {st.session_state['in_memory_events']} | "
# #                 f"CSV rows: {st.session_state['csv_rows']}"
# #             )
# #         else:
# #             st.warning(f"API reachable but not healthy: {h}")
# #     except Exception as e:
# #         st.error(f"Cannot reach API at {API}. Error: {e}")

# # def fetch_kpis(driver_id: str, base_premium: float):
# #     """Return dict with risk_score, prob_incident, premium or a message."""
# #     try:
# #         s = api_get(f"/driver/{driver_id}/score").json()
# #         if "prob_incident" not in s:
# #             return {"message": s.get("message", "No data yet. Load events first.")}
# #         p = api_post("/price", json={"driver_id": driver_id, "base_premium": base_premium}).json()
# #         return {
# #             "risk_score": s.get("risk_score", 0),
# #             "prob_incident": s.get("prob_incident", 0.0),
# #             "premium": p.get("premium", base_premium),
# #         }
# #     except Exception as e:
# #         return {"message": f"API error: {e}"}

# # def load_sample_events(n=500):
# #     """Sample N lines from events.ndjson and POST /ingest for each."""
# #     path = _events_path()
# #     try:
# #         with open(path, "r", encoding="utf-8") as f:
# #             lines = f.read().splitlines()
# #         if not lines:
# #             st.warning(f"No lines in {path}.")
# #             return
# #         sample = random.sample(lines, k=min(n, len(lines)))
# #         sent = 0
# #         for line in sample:
# #             payload = json.loads(line)
# #             r = api_post("/ingest", json=payload)
# #             r.raise_for_status()
# #             sent += 1
# #         # refresh counters after ingest
# #         refresh_health_banner()
# #         st.success(f"Loaded {sent} events from {path}")
# #     except FileNotFoundError:
# #         st.error(f"Load failed: file not found → '{path}'. Put events.ndjson there or set EVENTS_FILE.")
# #     except Exception as e:
# #         st.error(f"Load failed: {e}")

# # def events_table(driver_id, start, end, limit):
# #     params = {}
# #     if start.strip():
# #         params["start"] = start.strip()
# #     if end.strip():
# #         params["end"] = end.strip()
# #     params["limit"] = limit

# #     try:
# #         r = api_get(f"/driver/{driver_id}/events", params=params).json()
# #         df = pd.DataFrame(r.get("events", []))
# #         st.caption(f"Returned rows: {r.get('count', 0)}")
# #         if df.empty:
# #             st.info("No events for this driver/time range.")
# #             return
# #         if "ts" in df.columns:
# #             df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
# #             df = df.sort_values("ts")
# #         st.dataframe(df, use_container_width=True, height=420)

# #         # Download CSV
# #         csv = df.to_csv(index=False).encode()
# #         st.download_button(
# #             label="Download CSV",
# #             data=csv,
# #             file_name=f"{driver_id}_events.csv",
# #             mime="text/csv",
# #         )

# #         # Quick charts
# #         st.markdown("### Quick Charts")
# #         if "ts" in df.columns and "speed_kph" in df.columns:
# #             st.line_chart(df.set_index("ts")["speed_kph"], height=220)
# #         if "ts" in df.columns and "hard_brake" in df.columns:
# #             hb = df.dropna(subset=["ts"]).set_index("ts")["hard_brake"].resample("5min").sum()
# #             st.bar_chart(hb, height=220)

# #     except Exception as e:
# #         st.error(f"API error: {e}")

# # def fetch_features(driver_id: str) -> dict | None:
# #     try:
# #         r = api_get(f"/driver/{driver_id}/features").json()
# #         return r.get("features")
# #     except Exception as e:
# #         st.error(f"API error: {e}")
# #         return None

# # def render_features_table(features: dict, driver_id: str):
# #     feats = features.copy()
# #     driver_row = feats.pop("driver_id", driver_id)
# #     fdf = pd.DataFrame(list(feats.items()), columns=["feature", "value"])
# #     # coerce numeric
# #     fdf["value"] = pd.to_numeric(fdf["value"], errors="coerce")
# #     st.caption(f"Driver: {driver_row}")
# #     st.dataframe(fdf, use_container_width=True)

# # def generate_suggestions(features: dict, risk_score: float | int | None):
# #     """Return a list of human-friendly tips from engineered features."""
# #     if not features:
# #         return ["No features available yet. Ingest events first."]

# #     # Fetch with safe defaults
# #     f = lambda k, d=0.0: float(features.get(k, d) or 0.0)
# #     tips = []

# #     # Speeding & variability
# #     avg_speed = f("avg_speed")
# #     p95_speed = f("p95_speed")
# #     speed_std = f("speed_std")
# #     if p95_speed > 110 or speed_std > 25:
# #         tips.append("Reduce high-speed episodes and speed variability; stay closer to posted limits.")
# #     elif avg_speed > 85:
# #         tips.append("Average speed is high; slower, steadier driving lowers risk and fuel use.")
# #     else:
# #         tips.append("Good speed discipline—keep it steady and within limits.")

# #     # Harsh maneuvers
# #     hb = f("hard_brakes_per_100km")
# #     ha = f("hard_accels_per_100km")
# #     if hb > 8 or ha > 8:
# #         tips.append("Many harsh accelerations/brakes detected—leave more following distance and anticipate traffic.")
# #     elif hb > 3 or ha > 3:
# #         tips.append("Moderate aggressive events; smooth inputs improve safety and comfort.")
# #     else:
# #         tips.append("Smooth inputs—few harsh events. Nice!")

# #     # Phone use
# #     phone = f("phone_use_min_per_100km")
# #     if phone > 10:
# #         tips.append("Phone use while driving is high—enable Do Not Disturb/Waze voice prompts to minimize distraction.")
# #     elif phone > 3:
# #         tips.append("Reduce intermittent phone interactions; voice control helps.")
# #     else:
# #         tips.append("Minimal phone use—keep it up.")

# #     # Night driving
# #     night = f("night_km_share")
# #     if night > 0.5:
# #         tips.append("Heavy night driving—plan daylight trips when possible, use rest breaks, and keep headlights clean.")
# #     elif night > 0.2:
# #         tips.append("Some night driving—ensure alertness and proper lighting.")
# #     else:
# #         tips.append("Low night-time exposure—generally safer.")

# #     # Rush-hour & urban mix
# #     rush = f("rush_km_share")
# #     urban = f("urban_km_share")
# #     if rush > 0.5 or urban > 0.6:
# #         tips.append("Frequent congested/urban trips—add buffer time, pick less busy routes to avoid stop-and-go.")
# #     elif rush > 0.25 or urban > 0.4:
# #         tips.append("Moderate congestion exposure—route planning can reduce stress and risk.")
# #     else:
# #         tips.append("Lower congestion exposure—good route choices.")

# #     # Tie to score if available
# #     if risk_score is not None:
# #         if risk_score >= 80:
# #             tips.append("Overall risk is elevated—follow the above tips to unlock potential premium discounts.")
# #         elif risk_score <= 40:
# #             tips.append("Strong overall profile—maintain habits to keep premiums low.")

# #     return tips

# # # -----------------------------
# # # State init & Top banner
# # # -----------------------------
# # for k, v in [("csv_rows", 0), ("in_memory_events", 0), ("model_loaded", False)]:
# #     st.session_state.setdefault(k, v)

# # refresh_health_banner()

# # # -----------------------------
# # # Controls
# # # -----------------------------
# # driver_id = st.text_input("Driver ID", "D_000")
# # base_premium = st.number_input("Base Premium ($/mo)", min_value=0.0, value=120.0, step=1.0)

# # tabs = st.tabs(["Overview", "Events", "Features", "Required"])

# # # -----------------------------
# # # OVERVIEW
# # # -----------------------------
# # with tabs[0]:
# #     c1, c2 = st.columns([1, 1])
# #     if c1.button("Load 500 sample events", key="load_overview"):
# #         load_sample_events(500)
# #     if c2.button("Refresh", key="refresh_overview"):
# #         refresh_health_banner()

# #     kpis = fetch_kpis(driver_id, base_premium)
# #     if "message" in kpis:
# #         st.info(kpis["message"])
# #     else:
# #         a, b, c = st.columns(3)
# #         a.metric("Risk Score", int(round(kpis["risk_score"])))
# #         b.metric("Incident Probability", f"{kpis['prob_incident']*100:.2f}%")
# #         c.metric("Estimated Premium", f"${kpis['premium']:.1f}")

# #         # Display personalized suggestions under Overview
# #         feats = fetch_features(driver_id)
# #         risk = kpis.get("risk_score")
# #         tips = generate_suggestions(feats or {}, risk)
# #         st.subheader("Personalized Suggestions")
# #         for t in tips:
# #             st.write(f"• {t}")

# # # -----------------------------
# # # EVENTS
# # # -----------------------------
# # with tabs[1]:
# #     st.subheader("Driver Events")
# #     colA, colB, colC = st.columns([1, 1, 1])
# #     start = colA.text_input("Start (ISO8601, optional)", value="")
# #     end = colB.text_input("End (ISO8601, optional)", value="")
# #     limit = colC.number_input("Max rows (0 = all)", min_value=0, value=0, step=100)
# #     events_table(driver_id, start, end, limit)

# # # -----------------------------
# # # FEATURES
# # # -----------------------------
# # with tabs[2]:
# #     st.subheader("Engineered Features")
# #     feats = fetch_features(driver_id)
# #     if feats:
# #         render_features_table(feats, driver_id)
# #     else:
# #         st.info("No features available for this driver yet.")

# # # -----------------------------
# # # # REQUIRED  (new tab)
# # # # -----------------------------
# # with tabs[3]:
# #     st.subheader("Required: Behavior, Scores & Premium Changes")

# #     # 1) Summary cards
# #     kpis = fetch_kpis(driver_id, base_premium)
# #     if "prob_incident" in kpis:
# #         premium_now = kpis["premium"]
# #         k1, k2, k3 = st.columns(3)
# #         k1.metric("Risk Score", int(round(kpis.get("risk_score", 0))))
# #         k2.metric("Incident Probability", f"{kpis.get('prob_incident', 0)*100:.2f}%")
# #         k3.metric("Current Estimated Premium", f"${premium_now:.2f}")
# #     else:
# #         st.info(kpis.get("message", "No score yet. Load/ingest events first."))
# #         premium_now = None

# #     st.divider()

# #     # 2) Driving behavior snapshot (from features)
# #     feats = fetch_features(driver_id) or {}
# #     if feats:
# #         st.markdown("### Driving Behavior Snapshot")
# #         cols = st.columns(4)
# #         def _fmt(v, default="0"):
# #             try:
# #                 if v is None:
# #                     return default
# #                 if isinstance(v, (int, float)):
# #                     return f"{v:.2f}"
# #                 return str(v)
# #             except Exception:
# #                 return default

# #         kmap = [
# #             ("avg_speed", "Avg speed (kph)"),
# #             ("p95_speed", "P95 speed (kph)"),
# #             ("hard_brakes_per_100km", "Hard brakes /100km"),
# #             ("hard_accels_per_100km", "Hard accels /100km"),
# #             ("phone_use_min_per_100km", "Phone use min /100km"),
# #             ("night_km_share", "Night km share"),
# #             ("rush_km_share", "Rush-hour km share"),
# #             ("urban_km_share", "Urban km share"),
# #         ]
# #         for i, (key, label) in enumerate(kmap[:4]):
# #             cols[i].metric(label, _fmt(feats.get(key)))
# #         cols2 = st.columns(4)
# #         for i, (key, label) in enumerate(kmap[4:]):
# #             val = feats.get(key)
# #             if "share" in key and isinstance(val, (int, float)):
# #                 cols2[i].metric(label, f"{val*100:.1f}%")
# #             else:
# #                 cols2[i].metric(label, _fmt(val))
# #     else:
# #         st.info("No features to summarize driving behavior.")

# #     st.divider()

# #     # 3) Trends from events (if any)
# #     st.markdown("### Recent Trends")
# #     # You already display events in Events tab, consider using similar logic
# #     # or implement fetch_events helper if you want to present a summary

# #     st.info("No events to plot trends yet (please implement this based on your event retrieval logic).")

# #     st.divider()

# #     # 4) Premium scenarios
# #     st.markdown("### Premium Scenarios")
# #     if premium_now is not None:
# #         low, high = st.slider(
# #             "Choose base premium range for scenario ($/mo)",
# #             min_value=40, max_value=300, value=(80, 200), step=5
# #         )
# #         steps = st.number_input("Number of points", min_value=5, max_value=40, value=15, step=1)

# #         bases = np.linspace(low, high, int(steps))
# #         premiums = []
# #         for b in bases:
# #             try:
# #                 r = api_post("/price", json={"driver_id": driver_id, "base_premium": float(b)}).json()
# #                 premiums.append(r.get("premium", np.nan))
# #             except Exception:
# #                 premiums.append(np.nan)

# #         scen = pd.DataFrame({"base_premium": bases, "estimated_premium": premiums}).dropna()
# #         if not scen.empty:
# #             st.line_chart(scen.set_index("base_premium")["estimated_premium"], height=260)
# #             st.caption("Shows how the estimated premium would change as you adjust the base premium.")
# #         else:
# #             st.info("Could not compute premium scenarios.")
# #     else:
# #         st.info("Premium scenarios require a current score/price — ingest events first.")

# # src/dashboard.py-----------final but no deplo
# import os, json, random
# from datetime import datetime

# import requests
# import pandas as pd
# import numpy as np
# import streamlit as st

# # -----------------------------
# # Resolve API base: ENV > secrets > default
# # -----------------------------
# def _api_base() -> str:
#     env = os.getenv("API_BASE")
#     if env:
#         return env
#     try:
#         return st.secrets["API_BASE"]  # raises if secrets are absent
#     except Exception:
#         return "http://127.0.0.1:8000"

# #API = _api_base()

# API = _api_base()  # which resolves to localhost

# def api_get(path, **kwargs):
#     return requests.get(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)

# def api_post(path, **kwargs):
#     return requests.post(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)


# # Optional: where the sample events file lives
# def _events_path() -> str:
#     env = os.getenv("EVENTS_FILE")
#     if env:
#         return env
#     try:
#         return st.secrets["EVENTS_FILE"]
#     except Exception:
#         return "data/events.ndjson"

# st.set_page_config(page_title="Telematics UBI – Driver Dashboard", layout="wide")
# st.title("Telematics UBI – Driver Dashboard")

# # -----------------------------
# # Helpers
# # -----------------------------
# def api_get(path, **kwargs):
#     return requests.get(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)

# def api_post(path, **kwargs):
#     return requests.post(f"{API}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)

# def refresh_health_banner():
#     """Fetch health and paint banner + update counters."""
#     try:
#         h = api_get("/health", timeout=3).json()
#         ok = h.get("ok", False)
#         st.session_state["csv_rows"] = int(h.get("csv_rows", 0))
#         st.session_state["in_memory_events"] = int(h.get("in_memory_events", 0))
#         st.session_state["model_loaded"] = bool(h.get("model_loaded", False))
#         if ok:
#             st.success(
#                 f"Connected to API at {API} | "
#                 f"Model loaded: {st.session_state['model_loaded']} | "
#                 f"In-memory events: {st.session_state['in_memory_events']} | "
#                 f"CSV rows: {st.session_state['csv_rows']}"
#             )
#         else:
#             st.warning(f"API reachable but not healthy: {h}")
#     except Exception as e:
#         st.error(f"Cannot reach API at {API}. Error: {e}")

# def fetch_kpis(driver_id: str, base_premium: float):
#     """Return dict with risk_score, prob_incident, premium or a message."""
#     try:
#         s = api_get(f"/driver/{driver_id}/score").json()
#         if "prob_incident" not in s:
#             return {"message": s.get("message", "No data yet. Load events first.")}
#         p = api_post("/price", json={"driver_id": driver_id, "base_premium": base_premium}).json()
#         return {
#             "risk_score": s.get("risk_score", 0),
#             "prob_incident": s.get("prob_incident", 0.0),
#             "premium": p.get("premium", base_premium),
#         }
#     except Exception as e:
#         return {"message": f"API error: {e}"}

# def load_sample_events(n=500):
#     """Sample N lines from events.ndjson and POST /ingest for each."""
#     path = _events_path()
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             lines = f.read().splitlines()
#         if not lines:
#             st.warning(f"No lines in {path}.")
#             return
#         sample = random.sample(lines, k=min(n, len(lines)))
#         sent = 0
#         for line in sample:
#             payload = json.loads(line)
#             r = api_post("/ingest", json=payload)
#             r.raise_for_status()
#             sent += 1
#         # refresh counters after ingest
#         refresh_health_banner()
#         st.success(f"Loaded {sent} events from {path}")
#     except FileNotFoundError:
#         st.error(f"Load failed: file not found → '{path}'. Put events.ndjson there or set EVENTS_FILE.")
#     except Exception as e:
#         st.error(f"Load failed: {e}")

# def events_table(driver_id, start, end, limit):
#     params = {}
#     if start.strip():
#         params["start"] = start.strip()
#     if end.strip():
#         params["end"] = end.strip()
#     params["limit"] = limit

#     try:
#         r = api_get(f"/driver/{driver_id}/events", params=params).json()
#         df = pd.DataFrame(r.get("events", []))
#         st.caption(f"Returned rows: {r.get('count', 0)}")
#         if df.empty:
#             st.info("No events for this driver/time range.")
#             return
#         if "ts" in df.columns:
#             df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
#             df = df.sort_values("ts")
#         st.dataframe(df, use_container_width=True, height=420)

#         # Download CSV
#         csv = df.to_csv(index=False).encode()
#         st.download_button(
#             label="Download CSV",
#             data=csv,
#             file_name=f"{driver_id}_events.csv",
#             mime="text/csv",
#         )

#         # Quick charts
#         st.markdown("### Quick Charts")
#         if "ts" in df.columns and "speed_kph" in df.columns:
#             st.line_chart(df.set_index("ts")["speed_kph"], height=220)
#         if "ts" in df.columns and "hard_brake" in df.columns:
#             hb = df.dropna(subset=["ts"]).set_index("ts")["hard_brake"].resample("5min").sum()
#             st.bar_chart(hb, height=220)

#     except Exception as e:
#         st.error(f"API error: {e}")

# def fetch_features(driver_id: str) -> dict | None:
#     try:
#         r = api_get(f"/driver/{driver_id}/features").json()
#         return r.get("features")
#     except Exception as e:
#         st.error(f"API error: {e}")
#         return None

# def render_features_table(features: dict, driver_id: str):
#     feats = features.copy()
#     driver_row = feats.pop("driver_id", driver_id)
#     fdf = pd.DataFrame(list(feats.items()), columns=["feature", "value"])
#     # coerce numeric
#     fdf["value"] = pd.to_numeric(fdf["value"], errors="coerce")
#     st.caption(f"Driver: {driver_row}")
#     st.dataframe(fdf, use_container_width=True)

# def generate_suggestions(features: dict, risk_score: float | int | None):
#     """Return a list of human-friendly tips from engineered features."""
#     if not features:
#         return ["No features available yet. Ingest events first."]

#     # Fetch with safe defaults
#     f = lambda k, d=0.0: float(features.get(k, d) or 0.0)
#     tips = []

#     # Speeding & variability
#     avg_speed = f("avg_speed")
#     p95_speed = f("p95_speed")
#     speed_std = f("speed_std")
#     if p95_speed > 110 or speed_std > 25:
#         tips.append("Reduce high-speed episodes and speed variability; stay closer to posted limits.")
#     elif avg_speed > 85:
#         tips.append("Average speed is high; slower, steadier driving lowers risk and fuel use.")
#     else:
#         tips.append("Good speed discipline—keep it steady and within limits.")

#     # Harsh maneuvers
#     hb = f("hard_brakes_per_100km")
#     ha = f("hard_accels_per_100km")
#     if hb > 8 or ha > 8:
#         tips.append("Many harsh accelerations/brakes detected—leave more following distance and anticipate traffic.")
#     elif hb > 3 or ha > 3:
#         tips.append("Moderate aggressive events; smooth inputs improve safety and comfort.")
#     else:
#         tips.append("Smooth inputs—few harsh events. Nice!")

#     # Phone use
#     phone = f("phone_use_min_per_100km")
#     if phone > 10:
#         tips.append("Phone use while driving is high—enable Do Not Disturb/Waze voice prompts to minimize distraction.")
#     elif phone > 3:
#         tips.append("Reduce intermittent phone interactions; voice control helps.")
#     else:
#         tips.append("Minimal phone use—keep it up.")

#     # Night driving
#     night = f("night_km_share")
#     if night > 0.5:
#         tips.append("Heavy night driving—plan daylight trips when possible, use rest breaks, and keep headlights clean.")
#     elif night > 0.2:
#         tips.append("Some night driving—ensure alertness and proper lighting.")
#     else:
#         tips.append("Low night-time exposure—generally safer.")

#     # Rush-hour & urban mix
#     rush = f("rush_km_share")
#     urban = f("urban_km_share")
#     if rush > 0.5 or urban > 0.6:
#         tips.append("Frequent congested/urban trips—add buffer time, pick less busy routes to avoid stop-and-go.")
#     elif rush > 0.25 or urban > 0.4:
#         tips.append("Moderate congestion exposure—route planning can reduce stress and risk.")
#     else:
#         tips.append("Lower congestion exposure—good route choices.")

#     # Tie to score if available
#     if risk_score is not None:
#         if risk_score >= 80:
#             tips.append("Overall risk is elevated—follow the above tips to unlock potential premium discounts.")
#         elif risk_score <= 40:
#             tips.append("Strong overall profile—maintain habits to keep premiums low.")

#     return tips

# # -----------------------------
# # State init & Top banner
# # -----------------------------
# for k, v in [("csv_rows", 0), ("in_memory_events", 0), ("model_loaded", False)]:
#     st.session_state.setdefault(k, v)

# refresh_health_banner()

# # -----------------------------
# # Controls
# # -----------------------------
# driver_id = st.text_input("Driver ID", "D_000")
# base_premium = st.number_input("Base Premium ($/mo)", min_value=0.0, value=120.0, step=1.0)

# tabs = st.tabs(["Overview", "Events", "Features", "Behaviour", "Suggestions"])

# # -----------------------------
# # OVERVIEW
# # -----------------------------
# with tabs[0]:
#     c1, c2 = st.columns([1, 1])
#     if c1.button("Load 500 sample events", key="load_overview"):
#         load_sample_events(500)
#     if c2.button("Refresh", key="refresh_overview"):
#         refresh_health_banner()
#     st.subheader("Behavior, Scores & Premium Changes")
#     kpis = fetch_kpis(driver_id, base_premium)
#     if "message" in kpis:
#         st.info(kpis["message"])
#     else:
#         a, b, c = st.columns(3)
#         a.metric("Risk Score", int(round(kpis["risk_score"])))
#         b.metric("Incident Probability", f"{kpis['prob_incident']*100:.2f}%")
#         c.metric("Estimated Premium", f"${kpis['premium']:.1f}")
    

# # -----------------------------
# # EVENTS
# # -----------------------------
# with tabs[1]:
#     st.subheader("Driver Events")
#     colA, colB, colC = st.columns([1, 1, 1])
#     start = colA.text_input("Start (ISO8601, optional)", value="")
#     end = colB.text_input("End (ISO8601, optional)", value="")
#     limit = colC.number_input("Max rows (0 = all)", min_value=0, value=0, step=100)
#     events_table(driver_id, start, end, limit)

# # -----------------------------
# # FEATURES
# # -----------------------------
# with tabs[2]:
#     st.subheader("Engineered Features")
#     feats = fetch_features(driver_id)
#     if feats:
#         render_features_table(feats, driver_id)
#     else:
#         st.info("No features available for this driver yet.")

# # -----------------------------
# # REQUIRED (mirror KPIs so reviewers see them exactly where asked)
# # -----------------------------
# with tabs[3]:
#     st.subheader("Behavior, Scores & Premium Changes")

#     # 1) Summary cards
#     kpis = fetch_kpis(driver_id, base_premium)
#     if "prob_incident" in kpis:
#         premium_now = kpis["premium"]
#         k1, k2, k3 = st.columns(3)
#         k1.metric("Risk Score", int(round(kpis.get("risk_score", 0))))
#         k2.metric("Incident Probability", f"{kpis.get('prob_incident', 0)*100:.2f}%")
#         k3.metric("Current Estimated Premium", f"${premium_now:.2f}")
#     else:
#         st.info(kpis.get("message", "No score yet. Load/ingest events first."))
#         premium_now = None

#     st.divider()

#     # 2) Driving behavior snapshot (from features)
#     feats = fetch_features(driver_id) or {}
#     if feats:
#         st.markdown("### Driving Behavior Snapshot")
#         cols = st.columns(4)
#         def _fmt(v, default="0"):
#             try:
#                 if v is None:
#                     return default
#                 if isinstance(v, (int, float)):
#                     return f"{v:.2f}"
#                 return str(v)
#             except Exception:
#                 return default

#         kmap = [
#             ("avg_speed", "Avg speed (kph)"),
#             ("p95_speed", "P95 speed (kph)"),
#             ("hard_brakes_per_100km", "Hard brakes /100km"),
#             ("hard_accels_per_100km", "Hard accels /100km"),
#             ("phone_use_min_per_100km", "Phone use min /100km"),
#             ("night_km_share", "Night km share"),
#             ("rush_km_share", "Rush-hour km share"),
#             ("urban_km_share", "Urban km share"),
#         ]
#         for i, (key, label) in enumerate(kmap[:4]):
#             cols[i].metric(label, _fmt(feats.get(key)))
#         cols2 = st.columns(4)
#         for i, (key, label) in enumerate(kmap[4:]):
#             val = feats.get(key)
#             if "share" in key and isinstance(val, (int, float)):
#                 cols2[i].metric(label, f"{val*100:.1f}%")
#             else:
#                 cols2[i].metric(label, _fmt(val))
#     else:
#         st.info("No features to summarize driving behavior.")

#     st.divider()

#     # 3) Trends from events (if any)
#     st.subheader("Personalized Suggestions")
#     feats = fetch_features(driver_id)
#     kpis = fetch_kpis(driver_id, base_premium)
#     risk = None if "message" in kpis else kpis.get("risk_score")
#     tips = generate_suggestions(feats or {}, risk)
#     for t in tips:
#         st.write(f"• {t}")


#     # 4) Premium scenarios
#     st.markdown("### Premium Scenarios")
#     if premium_now is not None:
#         low, high = st.slider(
#             "Choose base premium range for scenario ($/mo)",
#             min_value=40, max_value=300, value=(80, 200), step=5
#         )
#         steps = st.number_input("Number of points", min_value=5, max_value=40, value=15, step=1)

#         bases = np.linspace(low, high, int(steps))
#         premiums = []
#         for b in bases:
#             try:
#                 r = api_post("/price", json={"driver_id": driver_id, "base_premium": float(b)}).json()
#                 premiums.append(r.get("premium", np.nan))
#             except Exception:
#                 premiums.append(np.nan)

#         scen = pd.DataFrame({"base_premium": bases, "estimated_premium": premiums}).dropna()
#         if not scen.empty:
#             st.line_chart(scen.set_index("base_premium")["estimated_premium"], height=260)
#             st.caption("Shows how the estimated premium would change as you adjust the base premium.")
#         else:
#             st.info("Could not compute premium scenarios.")
#     else:
#         st.info("Premium scenarios require a current score/price — ingest events first.")



# # -----------------------------
# # SUGGESTIONS (personalized coaching based on features)
# # -----------------------------
# with tabs[4]:
#     st.subheader("Personalized Suggestions")
#     feats = fetch_features(driver_id)
#     kpis = fetch_kpis(driver_id, base_premium)
#     risk = None if "message" in kpis else kpis.get("risk_score")
#     tips = generate_suggestions(feats or {}, risk)
#     for t in tips:
#         st.write(f"• {t}")



#------try dashboard csv
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------
# API BASE
# --------------------------------------------------------------------
def _api_base() -> str:
    env = os.getenv("API_BASE")
    if env:
        return env.rstrip("/")
    try:
        return str(st.secrets["API_BASE"]).rstrip("/")
    except Exception:
        return "http://127.0.0.1:8000"

API = _api_base()

def api_get(path: str, **kwargs):
    return requests.get(f"{API}{path}", timeout=kwargs.pop("timeout", 8), **kwargs)

def api_post(path: str, **kwargs):
    return requests.post(f"{API}{path}", timeout=kwargs.pop("timeout", 12), **kwargs)

# --------------------------------------------------------------------
# STREAMLIT SETUP
# --------------------------------------------------------------------
st.set_page_config(page_title="Telematics UBI – Driver Dashboard", layout="wide")
st.title("Telematics UBI – Driver Dashboard")

for k, v in [("in_memory", 0), ("csv_rows", 0), ("source", ""), ("model_loaded", False)]:
    st.session_state.setdefault(k, v)

# --------------------------------------------------------------------
# BANNER / HEALTH
# --------------------------------------------------------------------
def refresh_health_banner():
    try:
        h = api_get("/health", timeout=4).json()
        ok = bool(h.get("ok", False))
        st.session_state["in_memory"] = int(h.get("in_memory_events", 0))
        st.session_state["csv_rows"] = int(h.get("csv_rows", 0))
        st.session_state["source"] = h.get("source", "")
        st.session_state["model_loaded"] = bool(h.get("model_loaded", False))

        msg = (
            f"Connected to API at {API} | Model loaded: {st.session_state['model_loaded']} | "
            f"In-memory events: {st.session_state['in_memory']} | CSV rows: {st.session_state['csv_rows']}"
        )
        if st.session_state["source"]:
            msg += f" | Source: {st.session_state['source']}"
        if ok:
            st.success(msg)
        else:
            st.warning(f"API reachable but not healthy: {h}")
    except Exception as e:
        st.error(f"Cannot reach API at {API}. Error: {e}")

refresh_health_banner()

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def fetch_kpis(driver_id: str, base_premium: float) -> Dict[str, Any]:
    try:
        s = api_get(f"/driver/{driver_id}/score", timeout=6).json()
        if "prob_incident" not in s:
            return {"message": s.get("message", "No data yet. Load CSV first.")}
        p = api_post("/price", json={"driver_id": driver_id, "base_premium": float(base_premium)}).json()
        return {
            "risk_score": float(s.get("risk_score", 0)),
            "prob_incident": float(s.get("prob_incident", 0.0)),
            "premium": float(p.get("premium", base_premium)),
        }
    except Exception as e:
        return {"message": f"API error: {e}"}

def fetch_events_df(driver_id: str, start: str, end: str, limit: int) -> pd.DataFrame:
    params = {}
    if start.strip():
        params["start"] = start.strip()
    if end.strip():
        params["end"] = end.strip()
    if limit and limit > 0:
        params["limit"] = int(limit)
    r = api_get(f"/driver/{driver_id}/events", params=params, timeout=10).json()
    df = pd.DataFrame(r.get("events", []))
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.sort_values("ts")
    return df

def fetch_features(driver_id: str) -> Optional[Dict[str, Any]]:
    try:
        j = api_get(f"/driver/{driver_id}/features", timeout=6).json()
        return j.get("features")
    except Exception as e:
        st.error(f"Features fetch failed: {e}")
        return None

def render_features_table(features: Dict[str, Any], driver_id: str):
    feats = features.copy()
    feats.pop("driver_id", None)
    fdf = pd.DataFrame(list(feats.items()), columns=["feature", "value"])
    fdf["value"] = pd.to_numeric(fdf["value"], errors="coerce")
    st.caption(f"Driver: {driver_id}")
    st.dataframe(fdf, use_container_width=True)

# --------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------
left, right = st.columns([2, 1])
with left:
    driver_id = st.text_input("Driver ID", "D_000")
with right:
    base_premium = st.number_input("Base Premium ($/mo)", min_value=0.0, value=120.0, step=1.0)

with st.expander("Data Controls (CSV is the source of truth)", expanded=False):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    # 1) Reset in-memory to CSV
    if c1.button("Load CSV (reset to csv-only)"):
        try:
            j = api_post("/load_csv", json={"clear": True}).json()
            if j.get("ok"):
                st.success(f"Loaded {j.get('loaded',0)} rows from {j.get('source')}. In-memory: {j.get('in_memory')}")
                refresh_health_banner()
            else:
                st.warning(j)
        except Exception as e:
            st.error(f"/load_csv failed: {e}")

    # 2) Generate + append N to CSV
    sim_n = c2.number_input("Rows to append", value=500, min_value=1, max_value=5000, step=50)
    if c2.button("Generate + append to CSV"):
        try:
            j = api_post("/simulate_and_append", json={"n": int(sim_n)}).json()
            if j.get("ok"):
                st.success(
                    f"Appended {j.get('added',0)} rows to CSV. CSV rows: {j.get('csv_rows')} | In-memory: {j.get('in_memory')}"
                )
                refresh_health_banner()
            else:
                st.warning(j)
        except Exception as e:
            st.error(f"/simulate_and_append failed: {e}")

    # 3) Save current memory -> CSV (overwrite)
    if c3.button("Save memory → CSV (overwrite)"):
        try:
            j = api_post("/persist_csv", json={"dedupe": True, "overwrite": True}).json()
            if j.get("ok"):
                st.success(f"Saved to CSV ({j.get('source')}). CSV rows: {j.get('csv_rows')}")
                refresh_health_banner()
            else:
                st.warning(j)
        except Exception as e:
            st.error(f"/persist_csv failed: {e}")

    # 4) Clear memory
    if c4.button("Clear in-memory events"):
        try:
            j = api_post("/clear").json()
            if j.get("ok"):
                st.success("Cleared in-memory events.")
                refresh_health_banner()
            else:
                st.warning(j)
        except Exception as e:
            st.error(f"/clear failed: {e}")

st.divider()

tabs = st.tabs(["Overview", "Events", "Features", "Behaviour", "Suggestions"])

# --------------------------------------------------------------------
# OVERVIEW
# --------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    if c1.button("Refresh"):
        refresh_health_banner()
    if c2.button("Recompute KPIs"):
        pass

    st.subheader("Behavior, Scores & Premium Changes")
    kpis = fetch_kpis(driver_id, base_premium)
    if "message" in kpis:
        st.info(kpis["message"])
    else:
        a, b, c = st.columns(3)
        a.metric("Risk Score", int(round(kpis["risk_score"])))
        b.metric("Incident Probability", f"{kpis['prob_incident']*100:.2f}%")
        c.metric("Estimated Premium", f"${kpis['premium']:.2f}")

# --------------------------------------------------------------------
# EVENTS (with plots)
# --------------------------------------------------------------------
with tabs[1]:
    st.subheader("Driver Events")
    colA, colB, colC = st.columns([1, 1, 1])
    start = colA.text_input("Start (ISO8601, optional)", value="")
    end = colB.text_input("End (ISO8601, optional)", value="")
    limit = colC.number_input("Max rows (0 = all)", min_value=0, value=0, step=100)

    df = fetch_events_df(driver_id, start, end, int(limit))
    st.caption(f"Returned rows: {len(df)}")
    if df.empty:
        st.info("No events for this driver/time range.")
    else:
        st.dataframe(df, use_container_width=True, height=420)

        st.markdown("### Quick Charts")

        # 1) Speed over time (line)
        if "ts" in df.columns and "speed_kph" in df.columns:
            st.line_chart(df.set_index("ts")["speed_kph"], height=220)

        # 2) Hard brakes per 5 minutes (bar)
        if "ts" in df.columns and "brake_mps2" in df.columns:
            work = df.copy()
            work["hard_brake_event"] = (work["brake_mps2"] > 3).astype(int)
            work = work.dropna(subset=["ts"]).set_index("ts").resample("5min")["hard_brake_event"].sum()
            st.bar_chart(work, height=220)

        # 3) Speed distribution (histogram-like)
        if "speed_kph" in df.columns:
            hist = pd.cut(df["speed_kph"].fillna(0), bins=[0, 20, 40, 60, 80, 100, 120, 140], include_lowest=True)
            vc = hist.value_counts().sort_index()
            st.bar_chart(pd.DataFrame({"count": vc.values}, index=vc.index.astype(str)), height=220)

        # 4) Road-type share
        if "road_type" in df.columns:
            share = df["road_type"].fillna("unknown").value_counts()
            st.bar_chart(pd.DataFrame({"count": share}), height=220)

        # 5) Daily aggregates: avg speed & count
        if "ts" in df.columns and "speed_kph" in df.columns:
            dailies = (
                df.dropna(subset=["ts"])
                .set_index("ts")
                .resample("1D")
                .agg(avg_speed=("speed_kph", "mean"), events=("speed_kph", "count"))
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Daily Avg Speed")
                st.line_chart(dailies["avg_speed"], height=220)
            with c2:
                st.markdown("#### Daily Events")
                st.bar_chart(dailies["events"], height=220)

# --------------------------------------------------------------------
# FEATURES
# --------------------------------------------------------------------
with tabs[2]:
    st.subheader("Engineered Features (quick view)")
    feats = fetch_features(driver_id)
    if feats:
        render_features_table(feats, driver_id)
    else:
        st.info("No features available yet for this driver.")

# --------------------------------------------------------------------
# BEHAVIOUR (Required) — KPIs + Premium scenarios plot
# --------------------------------------------------------------------
with tabs[3]:
    st.subheader("Behavior, Scores & Premium Changes")

    kpis = fetch_kpis(driver_id, base_premium)
    if "message" in kpis:
        st.info(kpis["message"])
        premium_now = None
    else:
        premium_now = kpis["premium"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Risk Score", int(round(kpis.get("risk_score", 0))))
        k2.metric("Incident Probability", f"{kpis.get('prob_incident', 0)*100:.2f}%")
        k3.metric("Current Estimated Premium", f"${premium_now:.2f}")

    st.divider()
    st.markdown("### Premium Scenarios")

    if premium_now is not None:
        low, high = st.slider(
            "Choose base premium range ($/mo)",
            min_value=40, max_value=300, value=(80, 200), step=5
        )
        steps = st.number_input("Number of points", min_value=5, max_value=40, value=15, step=1)

        bases = np.linspace(low, high, int(steps))
        premiums = []
        for b in bases:
            try:
                r = api_post("/price", json={"driver_id": driver_id, "base_premium": float(b)}).json()
                premiums.append(r.get("premium", np.nan))
            except Exception:
                premiums.append(np.nan)

        scen = pd.DataFrame({"base_premium": bases, "estimated_premium": premiums}).dropna()
        if not scen.empty:
            scen = scen.set_index("base_premium")
            st.line_chart(scen["estimated_premium"], height=260)
            st.caption("How estimated premium changes as you adjust the base premium.")
        else:
            st.info("Could not compute premium scenarios — ensure events/score exist.")
    else:
        st.info("Premium scenarios require a current score/price — load CSV first.")

# --------------------------------------------------------------------
# SUGGESTIONS
# --------------------------------------------------------------------
with tabs[4]:
    st.subheader("Personalized Suggestions")
    feats = fetch_features(driver_id) or {}
    if feats:
        tips = []
        if feats.get("avg_speed", 0) > 85:
            tips.append("Reduce average speed for lower risk and fuel use.")
        if feats.get("speed_std", 0) > 25:
            tips.append("Aim for steadier speed to reduce variability.")
        if feats.get("phone_use_min_per_100km", 0) > 5:
            tips.append("Use Do Not Disturb while driving to limit distraction.")
        if feats.get("hard_brakes_per_100km", 0) > 6:
            tips.append("Anticipate traffic and keep longer following distance.")
        if not tips:
            tips = ["Good habits detected—keep it up!"]
        for t in tips:
            st.write(f"• {t}")
    else:
        st.info("No features yet — load CSV to compute suggestions.")


