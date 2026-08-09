# P4 Event-Time Evaluation Metric Clarification

Status: outcome-blind clarification issued before any formal P4 evaluation.

Campaign: `p4_public_itch_event_time_20260807_v1`

The frozen run contract describes `Correction/call` as decoded-state L2 per
state-changing projection call. The audited implementation in
`ProjectionTotals.mean_correction_per_call` uses a different, unambiguous
denominator:

`sum of per-trajectory decoded-state projection L2 / all scheduled projection-trajectory call pairs`.

This denominator includes scheduled calls whose correction is zero. P4 therefore
reports the metric as **decoded-state L2 per scheduled projection call**. The
fraction of calls that change state remains a separate diagnostic,
`state_changing_call_rate`. For LegalLOB, which schedules no external projection,
Correction/call is `N/A`; projection calls/path and cumulative repair are zero.

This clarification changes no training cell, checkpoint, representation,
projection schedule, evaluation history, seed, noise, threshold, metric value,
or outcome-selection rule. The P4 evaluation plan and final receipt must bind
this file by content hash.
