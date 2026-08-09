from pathlib import Path
import sys

import pytest
import torch

from experiments.multi_asset_event_time import run_training as p4
from experiments.multi_asset_event_time import train_fixed_steps as p4_entry


def _dataset_contracts():
    return {
        symbol: {
            "normalization": {
                "min_spread_normalized": 0.1,
                "min_level_gap_normalized": 0.1,
            }
        }
        for symbol, _ in p4.ASSETS
    }


def test_build_cell_plans_freezes_exact_24_cell_matrix(tmp_path: Path):
    cells = p4.build_cell_plans(tmp_path, "checkpoints/p4", _dataset_contracts())

    p4.validate_exact_cell_denominator(cells)
    assert len(cells) == 24
    assert cells[0]["cell_id"] == "aapl__train_seed0__legal_lob"
    assert cells[-1]["cell_id"] == "intc__train_seed2__matched_lob"
    assert {cell["train_seed"] for cell in cells} == {0, 1, 2}
    assert {cell["representation"] for cell in cells} == {
        "legal_lob",
        "matched_lob",
    }
    assert {cell["symbol"] for cell in cells} == {"AAPL", "AMZN", "GOOG", "INTC"}


def test_training_command_and_expected_args_freeze_fixed_steps(tmp_path: Path):
    cell = p4.build_cell_plans(
        tmp_path, "checkpoints/p4", _dataset_contracts()
    )[0]

    assert cell["expected_checkpoint_args"]["epochs"] == 50
    assert cell["expected_checkpoint_args"]["max_train_batches"] == 548
    assert cell["expected_checkpoint_args"]["early_stopping_patience"] == 0
    assert cell["expected_checkpoint_args"]["p4_uniform_representation_stats"] is True
    assert cell["expected_checkpoint_args"]["p4_uniform_validation"] is True
    assert p4.TARGET_OPTIMIZER_STEPS == 27_400
    assert "--max-train-batches 548" in cell["command_shell"]
    assert "--save-every 1000" in cell["command_shell"]
    assert (
        "experiments/multi_asset_event_time/train_fixed_steps.py"
        in cell["command_shell"]
    )


def test_training_command_resolves_to_frozen_wrapper_namespace(
    tmp_path: Path, monkeypatch
):
    cell = p4.build_cell_plans(
        tmp_path, "checkpoints/p4", _dataset_contracts()
    )[0]
    monkeypatch.setattr(sys, "argv", cell["command"][1:])

    actual = vars(p4_entry.parse_and_validate_args())
    actual["price_reference"] = "history_last_mid"
    actual["min_spread"] = 0.1
    actual["min_gap"] = 0.1

    assert actual == cell["expected_checkpoint_args"]


def test_optimizer_step_gate_accepts_exact_budget():
    checkpoint = {
        "optimizer_state_dict": {
            "state": {
                0: {"step": torch.tensor(float(p4.TARGET_OPTIMIZER_STEPS))},
                1: {"step": p4.TARGET_OPTIMIZER_STEPS},
            }
        }
    }

    assert p4._optimizer_steps(checkpoint) == [27_400, 27_400]


def test_optimizer_step_gate_rejects_non_integral_counter():
    checkpoint = {
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(1.5)}}}
    }

    with pytest.raises(p4.CampaignError, match="Non-integral Adam step"):
        p4._optimizer_steps(checkpoint)


def test_denominator_gate_rejects_asset_removal(tmp_path: Path):
    cells = p4.build_cell_plans(tmp_path, "checkpoints/p4", _dataset_contracts())

    with pytest.raises(p4.CampaignError, match="24-cell"):
        p4.validate_exact_cell_denominator(cells[:-1])


def test_terminal_summary_is_null_safe_before_final_integrity(tmp_path: Path):
    cells = p4.build_cell_plans(tmp_path, "checkpoints/p4", _dataset_contracts())
    for cell in cells:
        cell.update(
            {
                "status": "completed",
                "attempt_count": 1,
                "subprocess_launch_count": 1,
            }
        )
    ledger = {
        "cells": cells,
        "final_integrity": None,
        "protocol_drift": {"latched": False},
        "revision": 1,
        "updated_at_utc": "2026-01-01T00:00:00Z",
    }

    summary = p4.ledger_summary(ledger)

    assert summary["completed"] == 24
    assert summary["status"] == "terminal_with_declared_failures"
