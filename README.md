# LegalLOB

LegalLOB is a constraint-native diffusion model for generating Level-2 limit
order book snapshots. It represents spreads, price-level gaps, and queue depths
in legal coordinates, so the decoder produces ordered price ladders with
positive spreads and non-negative depths by construction.

The repository provides the LegalLOB transform, diffusion model, matched-coordinate
and PAVA baselines, Public Nasdaq ITCH preprocessing, training, one-shot evaluation,
and recursive rollout evaluation. Generated snapshots can be consumed directly by
subsequent forecasting, risk, and control modules.

## How LegalLOB works

1. A Level-2 snapshot is encoded as midpoint, spread, adjacent price gaps, and
   queue depths.
2. The diffusion model learns the dynamics of these coordinates from recent book
   history.
3. The LegalLOB decoder maps generated coordinates to a valid order book through
   positivity-preserving links and cumulative price construction.
4. Evaluation measures snapshot quality, recursive stability, downstream
   acceptance, and the repair burden of projection-based alternatives.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch can be installed first with the CUDA, MPS, or CPU build appropriate for
the local platform. Training and evaluation commands accept `--device`.

## Repository structure

```text
genlob/
  models/                    diffusion backbone, SDE, and sampling code
  utils/legal_lob.py         LegalLOB and matched-coordinate transforms
  utils/physics.py           validity checks and PAVA projection
  data/                      dataset loaders and normalization
  eval/metrics.py            evaluation metrics
scripts/
  train_v2.py                model training
  evaluate_v2.py             one-shot generation evaluation
  snowball_test.py           recursive rollout evaluation
  run_intermediate_consumer_panel.py
                              downstream-consumer evaluation
experiments/
  multi_asset_event_time/    fixed multi-asset event-time replication
tests/                        unit and workflow tests
```

## Prepare the data

The experiments use six Public Nasdaq TotalView-ITCH sample days. The download
and preprocessing utilities build the required datasets locally under `data/`.

```bash
bash scripts/download_nasdaq_2019_samples.sh

python scripts/prepare_public_itch_multisymbol.py \
  --itch-file data/raw/nasdaq_itch/public/01302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/03272019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/07302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/08302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/10302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/12302019.NASDAQ_ITCH50.gz \
  --symbols AAPL,AMZN,GOOG,INTC \
  --sampling both \
  --purge-gap 100 \
  --update-manifest
```

Processed arrays follow the layout documented in
[`data/README.md`](data/README.md). Market data, checkpoints, and generated
outputs remain in local ignored directories and follow their providers' terms.

## Train LegalLOB

This example trains a GOOG event-time model:

```bash
python scripts/train_v2.py \
  --dataset PUBLIC_ITCH_2019_SAMPLES_GOOG_event \
  --data-dir data \
  --hist-len 50 \
  --pred-len 10 \
  --batch-size 128 \
  --epochs 50 \
  --cfg-drop 0.1 \
  --representation legal_lob \
  --action-conditioning none \
  --physics-loss-weight 0 \
  --action-contrastive-weight 0 \
  --save-dir checkpoints/goog_legal_seed0 \
  --num-workers 0 \
  --seed 0 \
  --device mps
```

Set `--representation matched_lob` to train the matched-coordinate comparator.

## Evaluate generation

One-shot evaluation:

```bash
python scripts/evaluate_v2.py \
  --checkpoint checkpoints/goog_legal_seed0/final.pt \
  --weights ema \
  --dataset PUBLIC_ITCH_2019_SAMPLES_GOOG_event \
  --data-dir data \
  --split test \
  --n-batches 5 \
  --batch-size 128 \
  --ddim-steps 20 \
  --projection-mode none \
  --representation auto \
  --output-dir results/goog_legal_seed0/evaluation \
  --device mps
```

Recursive rollout evaluation:

```bash
python scripts/snowball_test.py \
  --genlob-ckpt checkpoints/goog_legal_seed0/final.pt \
  --weights ema \
  --dataset PUBLIC_ITCH_2019_SAMPLES_GOOG_event \
  --data-dir data \
  --hist-len 50 \
  --pred-len 10 \
  --horizon 100 \
  --ddim-steps 20 \
  --projection-mode none \
  --compute-projection-distance \
  --action-mode hold \
  --n-batches 5 \
  --batch-size 128 \
  --output-dir results/goog_legal_seed0/rollout \
  --device mps
```

The fixed AAPL, AMZN, GOOG, and INTC event-time study is organized in
[`experiments/multi_asset_event_time/`](experiments/multi_asset_event_time/).

```bash
python experiments/multi_asset_event_time/run_training.py
python experiments/multi_asset_event_time/run_evaluation.py
```

## Tests

```bash
python -m compileall -q genlob scripts experiments tests
python -m pytest -q
```

## License

LegalLOB is released under the MIT License.
