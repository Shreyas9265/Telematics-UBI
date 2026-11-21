#!/usr/bin/env bash
# Deploy the FastAPI backend to Vercel (free Hobby plan): https://telematics-ubi-api.vercel.app/docs
# The Streamlit dashboard (src/dashboard.py) runs on Streamlit Community Cloud with the secret
#   API_BASE = "https://telematics-ubi-api.vercel.app"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
mkdir -p "$OUT/src" "$OUT/data"
cp "$ROOT"/src/{__init__.py,api.py,pricing.py,schemas.py} "$OUT/src/"
cp "$ROOT"/data/{ingested_events.csv,lookup_context.csv} "$OUT/data/"
cp "$ROOT/vercel/index.py" "$OUT/index.py"
printf "fastapi\npandas\nnumpy\npydantic\n" > "$OUT/requirements.txt"
echo "3.12" > "$OUT/.python-version"
cd "$OUT" && npx vercel deploy --prod --yes --name telematics-ubi-api
