# Multi-Asset Event-Time Study

This directory contains the fixed-budget replication of LegalLOB on AAPL,
AMZN, GOOG, and INTC Public ITCH event-time Level-2 snapshots. The study compares
LegalLOB with matched-coordinate generation followed by endpoint PAVA and with
PAVA applied throughout recursive sampling.

The workflow reports immediate state acceptance, downstream-consumer acceptance,
endpoint validity, projection calls, correction magnitude, cumulative repair,
and a fidelity guardrail under paired training and evaluation seeds.

## Files

- `protocol.md` defines the asset set, split, training budget, pairing, metrics,
  and integrity checks.
- `metric_definition.md` fixes the denominator used for correction per call.
- `train_fixed_steps.py` applies the registered fixed-step training settings.
- `run_training.py` executes the 24 training cells.
- `run_evaluation.py` executes the paired evaluation workflow.
- `summarize_results.py` validates the complete grid and creates the compact
  multi-asset result panel.

The internal campaign identifier beginning with `p4_` is retained in generated
ledgers and checkpoints so artifacts remain traceable to the frozen study.

## Run

From the repository root:

```bash
python experiments/multi_asset_event_time/run_training.py
python experiments/multi_asset_event_time/run_evaluation.py
```

Use `python experiments/multi_asset_event_time/run_evaluation.py --prepare-only`
to create and inspect the paired evaluation plan before launching evaluation.
