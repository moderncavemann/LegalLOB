#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/raw/nasdaq_itch/public}"
mkdir -p "$OUTPUT_DIR"

FILES=(
  "01302019.NASDAQ_ITCH50.gz"
  "03272019.NASDAQ_ITCH50.gz"
  "07302019.NASDAQ_ITCH50.gz"
  "08302019.NASDAQ_ITCH50.gz"
  "10302019.NASDAQ_ITCH50.gz"
  "12302019.NASDAQ_ITCH50.gz"
)

MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
RETRY_SECONDS="${RETRY_SECONDS:-30}"

for name in "${FILES[@]}"; do
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if python scripts/fetch_nasdaq_itch_public.py \
      --name "$name" \
      --output-dir "$OUTPUT_DIR"; then
      break
    fi
    if ((attempt == MAX_ATTEMPTS)); then
      printf 'Download failed after %s attempts: %s\n' "$MAX_ATTEMPTS" "$name" >&2
      exit 1
    fi
    printf 'Retrying %s in %s seconds (attempt %s/%s)\n' \
      "$name" "$RETRY_SECONDS" "$((attempt + 1))" "$MAX_ATTEMPTS" >&2
    sleep "$RETRY_SECONDS"
  done
done
