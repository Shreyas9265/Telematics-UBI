#!/usr/bin/env bash
#python -m src.model --train data/events.ndjson

#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-data/ingested_events.csv}
OUT=${2:-models/risk_model.pkl}
PY=${PYTHON:-python}

echo "[*] Training from: $INPUT"
$PY -m src.model_train --input "$INPUT" --output "$OUT" --min_km_per_day 2.0 --cv 5
echo "[OK] Saved model to $OUT"
