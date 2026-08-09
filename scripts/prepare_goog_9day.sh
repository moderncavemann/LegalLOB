#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RAW_DIR="${RAW_DIR:-data/raw/lobster/GOOG}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/snapshot_l2/GOOG_9day}"
MAX_DAYS="${MAX_DAYS:-9}"

python scripts/prepare_lobster_snapshot_l2.py \
  --symbol GOOG \
  --dataset-id GOOG_9day \
  --raw-dir "$RAW_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --manifest data/manifests/datasets.yaml \
  --depth 10 \
  --max-days "$MAX_DAYS"

python scripts/validate_dataset_schema.py \
  --manifest data/manifests/datasets.yaml \
  --dataset GOOG_9day

python scripts/audit_dataset.py \
  --dataset GOOG_9day \
  --data-dir data \
  --json-out "$OUTPUT_DIR/audit.json"

python scripts/check_legal_lob_reconstruction.py \
  --dataset GOOG_9day \
  --data-dir data \
  --json-out "$OUTPUT_DIR/legal_lob_reconstruction.json"
