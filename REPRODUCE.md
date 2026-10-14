# Reproducing the results of LegalLOB

This repository contains LegalLOB, its baselines, the data preparation for the paper's four settings, and the training,
evaluation and summary scripts behind every result table and figure of the ICAIF '26 paper. Market data, trained
checkpoints and generated outputs are not included, so reproducing the results means preparing the data, training the
models and running the evaluations below. Back to the [README](README.md).

## Requirements

Python 3.10+ with the packages in `requirements.txt`, and PyTorch 2 with CUDA, MPS or CPU support (install the build for
your platform first). The paper's runs used PyTorch 2.8 on an Apple M3 Ultra (MPS). The GOOG-Event setting needs a
Databento account: `scripts/fetch_databento_sample.py` reads `DATABENTO_API_KEY` from `.env` and stops at a spending cap
(`--max-total-cost-usd`, default 5).

## Settings and variants

| Paper | Dataset ID | Source and split |
|---|---|---|
| AAPL-1Hz, AMZN-1Hz, GOOG-1Hz | `PUBLIC_ITCH_2019_SAMPLES_{AAPL,AMZN,GOOG}_1s` | Nasdaq public TotalView-ITCH samples; last valid regular-hours book in each second; training Jan 30, Mar 27 and Jul 30, validation Aug 30, test Oct 30 and Dec 30, 2019 |
| GOOG-Event | `GOOG_databento_20190130_mbp10_rth_v2` | Databento XNAS.ITCH MBP-10, GOOG, Jan 30, 2019, regular hours; every valid book update; chronological 70/15/15 split with 100-row purged gaps |

| Paper | Code |
|---|---|
| Raw | `--representation raw` |
| Raw+Penalty | `--representation raw --physics-loss-weight 1e-5` (`raw_physics`) |
| Coord-only | `--representation matched_lob` (the code calls this variant "matched") |
| LegalLOB | `--representation legal_lob` |
| +Post | `--projection-mode exact`: exact PAVA after decoding |
| +In-loop | `--sampler-projection-mode exact`: exact PAVA at every DDIM step (method IDs `raw_inloop_pava` and `matched_repeated_pava`) |

Every learned variant is trained with three seeds (0, 1, 2) for 50 epochs and evaluated with the EMA weights of its best
validation checkpoint and three sampling seeds (0, 1, 2), using DDIM-20. The defaults of the scripts below are set to the
paper's configuration.

## 1. Data

```bash
# AAPL/AMZN/GOOG-1Hz: download the six public ITCH sample days (about 25 GB) and build the 1-Hz datasets
bash scripts/download_nasdaq_2019_samples.sh
python scripts/prepare_public_itch_multisymbol.py \
  --itch-file data/raw/nasdaq_itch/public/01302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/03272019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/07302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/08302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/10302019.NASDAQ_ITCH50.gz \
  --itch-file data/raw/nasdaq_itch/public/12302019.NASDAQ_ITCH50.gz \
  --symbols AAPL,AMZN,GOOG --sampling both --purge-gap 100 --update-manifest

# GOOG-Event: fetch the GOOG MBP-10 file of Jan 30, 2019 and build the event-time dataset
python scripts/fetch_databento_sample.py --schemas mbp-10
python scripts/prepare_databento_mbp10_snapshot_l2.py \
  --dbn-file data/raw/databento/XNAS.ITCH/GOOG/2019-01-30/GOOG_XNAS-ITCH_2019-01-30_mbp-10_0000-2100UTC.dbn.zst \
  --symbol GOOG --dataset-id GOOG_databento_20190130_mbp10_rth_v2 \
  --regular-hours-only --purge-gap 100
```

`--sampling both` also builds event-time versions of the public ITCH data; the paper uses only the `_1s` datasets from
this source. Each processed dataset is written to `data/processed/snapshot_l2/<dataset-id>/` (layout in
[`data/README.md`](data/README.md)).

## 2. Training and single-window evaluation

```bash
# AAPL/AMZN/GOOG-1Hz: trains Raw, Raw+Penalty, Coord-only and LegalLOB and runs their single-window evaluations
python scripts/run_public_itch_panel_experiments.py --device mps

# GOOG-Event
python scripts/run_snapshot_training_seeds.py \
  --dataset GOOG_databento_20190130_mbp10_rth_v2 \
  --output-root results/goog_databento_rth_v2_full \
  --checkpoint-root checkpoints_training_seeds/goog_databento_rth_v2_50ep \
  --include-matched-baseline --device mps
```

Each evaluation generates ten future snapshots for 6,400 test histories. Then summarize the runs:

```bash
python scripts/summarize_feasibility_evidence.py      # violations, point MSE and temporal diagnostics
python scripts/summarize_financial_sensitivity.py     # snapshot financial functionals
```

The size of the repairs (Section 5.4) comes from an independent audit of the saved arrays:

```bash
python scripts/audit_saved_generation.py \
  --results-root results/public_itch_1s_panel_full/AAPL \
  --normalization data/processed/snapshot_l2/PUBLIC_ITCH_2019_SAMPLES_AAPL_1s/normalization.json \
  --dataset PUBLIC_ITCH_2019_SAMPLES_AAPL_1s \
  --output-dir results/submission_audits/AAPL
# repeat for AMZN, GOOG and GOOG-Event (results/goog_databento_rth_v2_full -> results/submission_audits/GOOG_databento_RTH_v2)
python scripts/verify_pava_evidence.py
```

## 3. GOOG-Event rollout, intermediate requests and probabilistic scores

```bash
# 100-step rollout of every method (Table 4, Figure 3), one run per sampling seed
for s in 0 1 2; do
  python scripts/run_all_method_rollout_panel.py --eval-seed $s \
    --output-root results/all_method_rollout_goog_event_640_eval$s --device mps
done
python scripts/summarize_all_method_rollout.py \
  --results-roots results/all_method_rollout_goog_event_640_eval0 \
                  results/all_method_rollout_goog_event_640_eval1 \
                  results/all_method_rollout_goog_event_640_eval2 \
  --output-prefix paper_data/all_method_rollout_goog_event_640_3eval \
  --table-path paper_tables/table_all_method_rollout_3eval.md

# Fail-closed monitor of every intermediate request (Table 5)
python scripts/run_intermediate_consumer_panel.py --device mps
python scripts/summarize_intermediate_consumer_outcome.py --results-root results/intermediate_consumer_goog_event

# 32 samples for each of 512 histories (Table 7)
python scripts/run_conditional_score_panel.py --device mps
python scripts/summarize_conditional_scores.py
```

The intermediate-request study uses the same 640 histories, sampling seeds and checkpoints as the rollout study and also
runs Coord-only+In-loop, which the rollout panel does not include. The Coord-only+In-loop row of Table 4 comes from it;
its values for LegalLOB and Coord-only+Post equal those of the rollout study.

## 4. Generation cost

```bash
python scripts/benchmark_method_efficiency.py --device mps    # Table 8: 1,280 future snapshots, seed-0 checkpoints
```

## Where the results are written

| Result | Script | File |
|---|---|---|
| Table 3 (violations before repair) | `summarize_feasibility_evidence.py` | `paper_tables/table_cross_asset_feasibility.md` |
| Size of the repairs (Section 5.4) | `audit_saved_generation.py`, `verify_pava_evidence.py` | `results/submission_audits/*/saved_generation_audit.json`, `paper_tables/table_pava_evidence_audit.md` |
| Table 4 and Figure 3 (rollout) | `summarize_all_method_rollout.py` | `paper_tables/table_all_method_rollout_3eval.md`, `paper_data/all_method_rollout_goog_event_640_3eval_*` |
| Table 5 (rejected intermediate requests) | `summarize_intermediate_consumer_outcome.py` | `results/intermediate_consumer_goog_event/intermediate_consumer_outcome.md` |
| Table 6 (fidelity) | `summarize_feasibility_evidence.py`, `summarize_financial_sensitivity.py` | point MSE: `paper_tables/table_cross_asset_feasibility.md`; temporal diagnostics: `paper_tables/table_temporal_distribution.md`; financial functionals: `paper_tables/table_financial_functional_sensitivity.md` |
| Table 7 (probabilistic scores) | `summarize_conditional_scores.py` | `paper_tables/table_conditional_scores.md` |
| Table 8 (generation cost) | `benchmark_method_efficiency.py` | `paper_tables/table_method_efficiency.md` |
| Figure 4 (GOOG-Event marginals) | `generate_paper_visualizations.py` | `paper_figures/figure_structural_distributions.*` |

All summaries first average the sampling seeds within each training seed and then report the mean and the sample standard
deviation across the three training seeds.

## Not included

Market data (the Nasdaq ITCH samples and the Databento file follow their providers' terms), trained checkpoints, generated
arrays and result files. Every step above re-creates them locally; `data/`, `checkpoints*/`, `results/`, `paper_data/`,
`paper_tables/` and `paper_figures/` are local working directories.

## Separate study: multi-asset event time

[`experiments/multi_asset_event_time/`](experiments/multi_asset_event_time/) holds a later fixed-budget study on AAPL, AMZN,
GOOG and INTC event-time books that is not part of the ICAIF paper:

```bash
python experiments/multi_asset_event_time/run_training.py
python experiments/multi_asset_event_time/run_evaluation.py
```
