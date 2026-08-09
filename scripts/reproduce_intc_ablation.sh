#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="INTC_9day"
DATA_DIR="data"
RAW_CKPT="checkpoints_raw/intc_9day_full_50ep/best.pt"
LEGAL_CKPT="checkpoints_legal/intc_9day_full_50ep/best.pt"
OUT_DIR="${OUTPUT_ROOT:-results/ablation_projection}"
BATCH_SIZE="${BATCH_SIZE:-128}"
N_BATCHES="${N_BATCHES:-50}"
DDIM_STEPS="${DDIM_STEPS:-20}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUT_DIR="$2"
      shift 2
      ;;
    --n-batches)
      N_BATCHES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --ddim-steps)
      DDIM_STEPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "Writing reproduction outputs to: $OUT_DIR"

run_eval() {
  local seed="$1"
  local method="$2"
  local checkpoint="$3"
  local projection_mode="$4"
  local sampler_projection_mode="$5"
  local output_dir

  if [[ "$seed" == "0" ]]; then
    output_dir="$OUT_DIR/$method"
  else
    output_dir="$OUT_DIR/seeds/seed${seed}/$method"
  fi

  python scripts/evaluate_v2.py \
    --checkpoint "$checkpoint" \
    --weights ema \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --split test \
    --output-dir "$output_dir" \
    --n-batches "$N_BATCHES" \
    --batch-size "$BATCH_SIZE" \
    --ddim-steps "$DDIM_STEPS" \
    --projection-mode "$projection_mode" \
    --sampler-projection-mode "$sampler_projection_mode" \
    --representation auto \
    --seed "$seed"
}

for seed in 0 1 2; do
  run_eval "$seed" "intc_raw_no_projection" "$RAW_CKPT" "none" "none"
  run_eval "$seed" "intc_raw_posthoc_pava" "$RAW_CKPT" "exact" "none"
  run_eval "$seed" "intc_raw_inloop_dmpp" "$RAW_CKPT" "none" "exact"
  run_eval "$seed" "intc_legal_no_projection" "$LEGAL_CKPT" "none" "none"
done

python scripts/snowball_test.py \
  --genlob-ckpt "$LEGAL_CKPT" \
  --weights ema \
  --dataset "$DATASET" \
  --data-dir "$DATA_DIR" \
  --hist-len 50 \
  --pred-len 10 \
  --horizon 100 \
  --ddim-steps "$DDIM_STEPS" \
  --representation auto \
  --projection-mode none \
  --compute-projection-distance \
  --action-mode replay \
  --batch-size "$BATCH_SIZE" \
  --n-batches "$N_BATCHES" \
  --output-dir "$OUT_DIR/intc_legal_snowball_100_replay" \
  --seed 0

python scripts/write_ablation_provenance.py --results-dir "$OUT_DIR"
python scripts/summarize_projection_ablation.py --results-dir "$OUT_DIR"
python scripts/summarize_seed_ablation.py --results-dir "$OUT_DIR"

python -m pytest -q
