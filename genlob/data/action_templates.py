"""Diagnostic action templates matched to snapshot action conventions.

These helpers are not evidence of real Buy/Sell market impact. The current
snapshot datasets use heuristic/proxy action channels; formal action causality
requires message-level aggressor-side labels from Databento MBO, ITCH, or
LOBSTER event data.
"""

from __future__ import annotations

import torch


NASDAQ_DEFAULT_TEMPLATE = {
    "abs_direction": 0.72,
    "size": 0.59,
    "aggressiveness": 0.05,
    "duration": 0.50,
    "urgency": 0.32,
}


def make_counterfactual_action(
    direction: float,
    batch_size: int,
    device: torch.device | str,
    dataset: str,
    action_dim: int = 5,
) -> torch.Tensor:
    """Build an in-distribution diagnostic intervention vector.

    The function name is kept for backward compatibility with old experiment
    scripts. In the current protocol this is a diagnostic template, not a
    formal counterfactual action with observed causal ground truth.

    The NASDAQ action extractor does not train on one-hot direction-only
    interventions. Non-hold actions also carry size/aggressiveness/urgency
    fields, so diagnostic probes should use a representative template instead
    of zero-filling those dimensions. Do not interpret the returned vector as a
    true aggressor-side Buy/Sell causal label.
    """
    action = torch.zeros(batch_size, action_dim, device=device)

    if dataset.startswith("nasdaq_") or dataset.count("_") == 1:
        action[:, 3] = NASDAQ_DEFAULT_TEMPLATE["duration"]
        if direction != 0:
            sign = 1.0 if direction > 0 else -1.0
            action[:, 0] = sign * NASDAQ_DEFAULT_TEMPLATE["abs_direction"]
            action[:, 1] = NASDAQ_DEFAULT_TEMPLATE["size"]
            action[:, 2] = NASDAQ_DEFAULT_TEMPLATE["aggressiveness"]
            action[:, 4] = NASDAQ_DEFAULT_TEMPLATE["urgency"]
        return action

    raise ValueError(f"Unsupported dataset in slim checkout: {dataset}")


def summarize_action_templates(actions) -> dict:
    """Summarize empirical non-hold action magnitudes for audit reports."""
    actions = torch.as_tensor(actions)
    if actions.numel() == 0:
        return {}

    actions = actions.detach().float().cpu()
    direction = actions[:, 0]
    out = {
        "rows": int(actions.shape[0]),
        "non_hold_rate": float((direction.abs() > 1e-8).float().mean()),
    }
    for name, mask in (("buy", direction > 1e-8), ("sell", direction < -1e-8)):
        if not bool(mask.any()):
            continue
        subset = actions[mask]
        out[name] = {
            "count": int(subset.shape[0]),
            "mean": [float(v) for v in subset.mean(dim=0)],
            "median": [float(v) for v in subset.median(dim=0).values],
        }
    return out
