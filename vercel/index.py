"""Vercel serverless entry point for the Telematics UBI API."""
import shutil
from pathlib import Path

import src.api as api

# Only /tmp is writable on Vercel: work on a copy of the bundled sample data.
TMP = Path("/tmp/telematics")
TMP.mkdir(parents=True, exist_ok=True)
for name in ("ingested_events.csv", "lookup_context.csv"):
    source, target = Path(__file__).resolve().parent / "data" / name, TMP / name
    if source.exists() and not target.exists():
        shutil.copy(source, target)
api.DATA_DIR = TMP
api.CSV_PATH = TMP / "ingested_events.csv"
api.NDJSON_PATH = TMP / "events.ndjson"

# Each serverless instance keeps events in memory, so any request that finds memory empty
# loads the bundled sample data first (except the endpoints that manage the data themselves).
MANAGES_DATA = {"/clear", "/load_csv", "/bootstrap", "/simulate_and_append", "/ingest"}


@api.app.middleware("http")
async def ensure_sample_data(request, call_next):
    if not api.EVENTS and request.url.path not in MANAGES_DATA:
        api.bootstrap(api.BootstrapReq())
    return await call_next(request)

app = api.app
