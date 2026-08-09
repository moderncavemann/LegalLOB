import sys

import torch

from scripts.run_snapshot_ablation import main as ablation_main
from scripts.train_v2 import fixed_validation_rng


def test_fixed_validation_rng_reuses_noise_and_restores_training_rng():
    torch.manual_seed(123)
    state_before = torch.random.get_rng_state().clone()
    with fixed_validation_rng(999):
        first = torch.randn(8)
    assert torch.equal(state_before, torch.random.get_rng_state())
    with fixed_validation_rng(999):
        second = torch.randn(8)
    assert torch.equal(first, second)
    assert torch.equal(state_before, torch.random.get_rng_state())


def test_ablation_runs_paired_bootstrap_after_baseline_evaluation(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_snapshot_ablation.py",
            "--dataset",
            "fixture",
            "--output-dir",
            str(tmp_path / "results"),
            "--raw-save-dir",
            str(tmp_path / "raw"),
            "--legal-save-dir",
            str(tmp_path / "legal"),
            "--skip-training",
            "--skip-snowball",
            "--eval-seeds",
            "0",
            "--n-batches",
            "1",
            "--dry-run",
        ],
    )

    assert ablation_main() == 0
    output = capsys.readouterr().out
    persistence_eval = output.index("evaluate_snapshot_baseline.py")
    persistence_bootstrap = output.index("block_bootstrap_persistence_vs_legal.json")
    moving_block_eval = output.index("--baseline moving_block_bootstrap")
    moving_block_bootstrap = output.index("block_bootstrap_moving_block_bootstrap_vs_legal.json")

    assert persistence_eval < persistence_bootstrap
    assert moving_block_eval < moving_block_bootstrap
