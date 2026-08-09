# P4 Event-Time Robustness Frozen Run Contract

Campaign ID: `p4_public_itch_event_time_20260807_v1`

This document fixes the P4 protocol before any formal denominator is launched.
P4 is a separately trained robustness replication on event-time sampled Level-2
snapshots. It is not message-level generation, order replay, queue/fill modelling,
causal action evaluation, unseen-asset transfer, or cross-venue generalization.

## Fixed datasets and split contract

The asset set is fixed before observing P4 outcomes:

1. `PUBLIC_ITCH_2019_SAMPLES_AAPL_event`
2. `PUBLIC_ITCH_2019_SAMPLES_AMZN_event`
3. `PUBLIC_ITCH_2019_SAMPLES_GOOG_event`
4. `PUBLIC_ITCH_2019_SAMPLES_INTC_event`

All four datasets must identify the same `nasdaq_public_itch_samples` source,
`sampling=event`, and the same six dates. The frozen day split is:

- train: 2019-01-30, 2019-03-27, 2019-07-30;
- validation: 2019-08-30;
- test: 2019-10-30, 2019-12-30;
- purge gap: 100 source rows.

Each dataset must have true timestamps and day identifiers, no schema errors,
and normalization metadata with `fit_split=train_only` and
`price_reference_protocol=history_last_mid`. Dataset normalization and the
additional representation-coordinate statistics may access only that asset's
training split. Representation-coordinate statistics use a deterministic
uniform sample of 548 batches across the full group-safe training-window index,
not the chronological head.

## Fixed training denominator and compute budget

The formal denominator is exactly:

`4 assets x 3 training seeds x 2 representations = 24 training cells`.

- training seeds: 0, 1, 2;
- representations: `legal_lob`, `matched_lob`;
- each asset is trained independently;
- no 1-Hz or other existing checkpoint may be reused;
- `epochs=50`, `max_train_batches=548`, `batch_size=128`;
- exact optimizer-update budget: `50 x 548 = 27,400` per cell;
- `max_val_batches=183`, sampled deterministically and uniformly from the
  corresponding validation split;
- `early_stopping_patience=0`;
- `validation_seed=10000`;
- `hist_len=50`, `pred_len=10`, `cfg_drop=0.1`, learning rate `1e-4`;
- `action_conditioning=none`, physics and action-contrastive losses both zero;
- EMA rate 0.999, MPS device, zero data-loader workers;
- formal evaluation uses EMA weights from `final.pt`, whose Adam state must
  prove exactly 27,400 optimizer updates.

All 24 cells are attempted at most once. A started cell that crashes, is
interrupted, has a non-finite skipped update, fails an artifact gate, or sees
source/data drift is terminal failed and is never silently retried. Pending,
never-started cells may be launched by an explicit resume of the same frozen
ledger. Failed cells and all denominators remain in the final record.

## Fixed paired evaluation

For every asset, the completed fixed-step checkpoints are evaluated with:

- training seeds: 0, 1, 2;
- evaluation seeds: 0, 1, 2;
- 5 batches x 128 histories = 640 histories per train/evaluation-seed cell;
- horizon 100, prediction chunk 10, 20 DDIM steps per chunk;
- hold action, identical test histories and real futures;
- identical evaluation seed and initial DDIM noise across methods;
- EMA weights from `final.pt` only.

The evaluation denominator is 36 train/evaluation-seed cells and 108 method
rows. The three fixed methods are:

1. LegalLOB, with no external projection;
2. Matched+Post, with one exact PAVA repair per generated 10-step chunk;
3. Matched+Repeated PAVA, with exact PAVA inside every reverse step.

Within each evaluation cell, history starts, real futures, post-load RNG state,
and initial-noise hashes must match across methods. Matched+Post must also
reproduce its independently generated reference rollout within the frozen
numeric tolerance.

## Primary metrics and directions

Only the following P4 outcomes are promoted to the compact robustness panel:

| Metric | Definition | Direction |
|---|---|---:|
| Pre-projection invalidity | 1 minus pre-projection frame acceptance over the recursive path | down |
| Intermediate consumer rejection | 1 minus pre-projection block acceptance over the recursive path | down |
| Endpoint invalidity | 1 minus delivered endpoint frame acceptance | down |
| Projection calls/path | external projection calls over a 100-step path | down |
| Correction/call | decoded-state L2 change per state-changing projection call | down |
| Cumulative repair | summed decoded-state projection L2 over a path | down |
| Fidelity guardrail | market-order slippage Wasserstein-1 distance in ticks | down |

Correction/call is `N/A`, not zero, when a method makes no projection call.
Calls/path and cumulative repair are zero for LegalLOB by construction.

## Integrity and scientific decision rules

Integrity success requires all of the following before unified freezing:

- 24/24 training cells terminal and exactly accounted for;
- each launched cell has `attempt_count=1` and `subprocess_launch_count=1`;
- every successful training cell has 50 finite history rows, no early stop, and
  every Adam parameter state at step 27,400 in `final.pt`;
- representation statistics are finite, positive, and identical across the
  three seeds for a fixed asset/representation;
- 36/36 evaluation cells and 108/108 method rows are accounted for;
- all method-pairing, denominator, implementation, source, data, checkpoint,
  command, log, and output hash checks pass;
- all four pre-fixed assets remain in the report, including negative results.

An asset obtains strict mechanism GO only if LegalLOB has zero pre-projection
invalidity, zero intermediate block rejection, zero endpoint invalidity, and
zero calls/path; Matched+Post has non-zero intermediate rejection but zero
delivered endpoint invalidity; and repeated PAVA delivers valid intermediate
states while making state-changing calls. A four-asset headline GO requires all
four pre-fixed assets to pass. Finite-precision LegalLOB failures are retained
and classified as a directional or scientific no-GO result, never removed.

The fidelity guardrail is descriptive rather than a selection rule. Its value
must be finite for all methods. If LegalLOB is worse than both Matched
comparators for an asset, that asset receives a fidelity warning and the paper
must not claim absence of fidelity degradation. No outcome can trigger a
training retry, threshold change, seed change, or asset exclusion.

Allowed final classifications are `verified_go`,
`verified_scientific_no_go`, and `terminal_with_declared_failures`. Formal
results are written to new P4 roots and never alter the frozen P0--P3 evidence.
