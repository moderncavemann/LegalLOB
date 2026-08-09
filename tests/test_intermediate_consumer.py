from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.run_intermediate_consumer_panel as consumer_runner
from genlob.utils.legal_lob import MatchedLOBTransform
from genlob.utils.physics import project_to_valid_lob
from scripts.run_intermediate_consumer_panel import (
    ConsumerTrace,
    ProjectionTotals,
    RiskContract,
    StageTotals,
    exact_decoded_projection_feedback,
    freeze_risk_contract,
    load_variant_for_legacy_stream,
    observed_ddim_sample,
    set_seed,
    strict_admissibility,
    torch_rng_sha256,
)


NORMALIZATION = {
    "min_spread_normalized": 0.1,
    "min_level_gap_normalized": 0.1,
    "ticks_per_price_unit": 10.0,
    "volume_scale": 15.0,
}


def valid_lob(batch=1, frames=1):
    lob = np.zeros((batch, 2, 20, frames), dtype=np.float32)
    # Keep a margin above the 0.1 contract so the zero-tolerance assertion is
    # not a float32 representation test for the decimal value 0.1.
    asks = np.arange(1, 11, dtype=np.float32) * 0.2
    bids = -np.arange(1, 11, dtype=np.float32) * 0.2
    lob[:, 0, :, :] = np.concatenate([asks, bids])[None, :, None]
    lob[:, 1, :, :] = 1.0
    return lob


def test_strict_admissibility_is_fail_closed():
    lob = valid_lob(batch=4)
    lob[1, 0, 0, 0] = lob[1, 0, 10, 0]
    lob[2, 1, 3, 0] = -1.0
    lob[3, 0, 5, 0] = np.nan

    masks = strict_admissibility(lob, NORMALIZATION)

    assert masks["accepted"].tolist() == [True, False, False, False]
    assert masks["spread"].sum() == 1
    assert masks["negative_depth"].sum() == 1
    assert masks["nonfinite"].sum() == 1


def test_risk_contract_uses_training_depth_and_deterministic_indices():
    train = np.transpose(valid_lob(batch=12), (0, 3, 1, 2)).reshape(12, 2, 20)
    train[:, 1, 0] = np.linspace(0.5, 1.5, 12)
    train[:, 1, 10] = np.linspace(0.75, 1.75, 12)

    left = freeze_risk_contract(train, NORMALIZATION, max_threshold_snapshots=8)
    right = freeze_risk_contract(train, NORMALIZATION, max_threshold_snapshots=8)

    assert left.target_size_shares > 0
    assert np.isfinite(left.slippage_alarm_ticks)
    assert np.isfinite(left.liquidity_alarm_target_units)
    assert left.threshold_train_snapshots == 8
    assert left.threshold_index_sha256 == right.threshold_index_sha256


def test_consumer_trace_separates_pre_rejection_from_delivered_acceptance():
    contract = RiskContract(100.0, 10.0, 0.1, 1, "fixture")
    pre = valid_lob(batch=2, frames=2)
    pre[0, 1, 0, 0] = -1.0
    delivered = valid_lob(batch=2, frames=2)
    trace = ConsumerTrace(NORMALIZATION, contract)
    trace.start_batch(2)
    trace.observe(pre, delivered, call_index=0)
    trace.end_batch()

    summary = trace.summary()
    assert summary["pre"]["frame_acceptance_rate"] == 0.75
    assert summary["delivered"]["frame_acceptance_rate"] == 1.0
    assert summary["delivered"]["zero_tolerance_frame_acceptance_rate"] == 1.0
    assert summary["pre_completed_without_rejection_rate"] == 0.5
    assert summary["delivered_completed_without_rejection_rate"] == 1.0


def test_stage_totals_gates_before_risk_and_skips_all_rejected(monkeypatch):
    calls = []

    def fake_risk(lob, normalization, contract):
        calls.append(lob.copy())
        count = lob.shape[0] * lob.shape[-1]
        return {
            "max_slippage": np.zeros(count),
            "liquidity_5tick": np.ones(count),
            "underfill": np.zeros(count, dtype=bool),
            "alarm": np.zeros(count, dtype=bool),
        }

    monkeypatch.setattr(consumer_runner, "_algebraic_risk_arrays", fake_risk)
    contract = RiskContract(100.0, 10.0, 0.1, 1, "fixture")
    mixed = valid_lob(batch=3)
    mixed[1, 1, 0, 0] = -1.0
    mixed[2, 0, 0, 0] = np.nan
    totals = StageTotals()
    totals.update(mixed, NORMALIZATION, contract)

    assert len(calls) == 1
    assert calls[0].shape == (1, 2, 20, 1)
    assert np.isfinite(calls[0]).all()
    assert totals.summary()["accepted_risk_snapshots"] == 1

    rejected = mixed[1:]
    rejected_totals = StageTotals()
    rejected_totals.update(rejected, NORMALIZATION, contract)
    rejected_summary = rejected_totals.summary()
    assert len(calls) == 1
    assert rejected_summary["accepted_risk_snapshots"] == 0
    assert rejected_summary["risk_alarm_rate_among_accepted"] is None
    assert rejected_summary["finite_max_slippage_mean_ticks"] is None


def test_projection_totals_reports_calls_per_path_and_incidence():
    totals = ProjectionTotals()
    totals.register_paths(2)
    pre = torch.zeros(2, 2, 20, 1)
    projected = pre.clone()
    projected[0, 0, 0, 0] = 1.0
    totals.observe(pre, projected)
    totals.observe(pre, pre)

    summary = totals.summary()
    assert summary["calls_per_trajectory"] == 2.0
    assert summary["state_changing_call_rate"] == 0.25
    assert summary["mean_cumulative_correction_per_trajectory"] == 0.5


def test_exact_projection_feedback_keeps_pava_output_separate_from_roundtrip():
    transform = MatchedLOBTransform()
    lob = torch.from_numpy(valid_lob(batch=2, frames=2))
    lob[0, 0, 3, 0] = lob[0, 0, 2, 0] - 0.5
    lob[1, 1, 7, 1] = -0.25
    model_state = transform.encode(lob)

    feedback = exact_decoded_projection_feedback(
        model_state,
        "matched_lob",
        transform,
        None,
        min_spread=0.1,
        min_gap=0.1,
        device=torch.device("cpu"),
    )
    direct = project_to_valid_lob(
        lob, min_spread=0.1, min_gap=0.1, use_gpu_approx=False
    )

    assert torch.equal(feedback.pre_lob, lob)
    assert torch.equal(feedback.exact_projected_lob, direct)
    assert strict_admissibility(
        feedback.exact_projected_lob.numpy(), NORMALIZATION
    )["accepted"].all()
    assert torch.equal(
        feedback.roundtrip_lob, transform.decode(feedback.feedback_state)
    )


def test_legacy_loader_preserves_seed_constructor_sampling_order(monkeypatch):
    def fake_load(_path, _device, _weights):
        torch.rand(17)
        return torch.nn.Identity(), object(), {"arch": "fixture"}, {}

    monkeypatch.setattr(consumer_runner, "load_genlob", fake_load)
    set_seed(73)
    torch.rand(17)
    expected_hash = torch_rng_sha256()

    _, _, _, _, observed_hash = load_variant_for_legacy_stream(
        Path("fixture.pt"), torch.device("cpu"), 73
    )

    assert observed_hash == expected_hash


def test_legacy_loader_matches_real_checkpoint_rng_when_available():
    checkpoint = Path(
        "checkpoints_training_seeds/goog_databento_rth_v2_50ep/"
        "train_seed0/matched_lob/best.pt"
    )
    if not checkpoint.exists():
        pytest.skip("Frozen integration checkpoint is not available")
    set_seed(19)
    model, _, _, _ = consumer_runner.load_genlob(
        checkpoint, torch.device("cpu"), "ema"
    )
    expected_hash = torch_rng_sha256()
    del model

    model, _, _, _, observed_hash = load_variant_for_legacy_stream(
        checkpoint, torch.device("cpu"), 19
    )
    del model

    assert observed_hash == expected_hash


def test_observed_ddim_calls_observer_and_projection_each_step():
    class DummySDE:
        N = 4
        alphas_cumprod = torch.tensor([0.95, 0.85, 0.75, 0.65])

        @staticmethod
        def prior_sampling(shape):
            return torch.zeros(shape)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, x, labels, history, action):
            self.inputs.append(x.detach().clone())
            return torch.zeros_like(x)

    calls = {"project": 0, "observe": 0}

    def project(x):
        calls["project"] += 1
        return x + 0.01

    def observe(pre, delivered, reverse_index, timestep):
        calls["observe"] += 1
        assert delivered.shape == pre.shape
        assert reverse_index in (0, 1)
        assert timestep in (0, 2)

    model = DummyModel()
    output, last_pre, noise_hash = observed_ddim_sample(
        model,
        DummySDE(),
        (2, 2, 20, 3),
        torch.zeros(2, 2, 20, 5),
        None,
        ddim_steps=2,
        device=torch.device("cpu"),
        project_fn=project,
        observer_fn=observe,
    )

    assert output.shape == last_pre.shape == (2, 2, 20, 3)
    assert len(noise_hash) == 64
    assert calls == {"project": 2, "observe": 2}
    assert torch.allclose(
        model.inputs[1],
        torch.full_like(model.inputs[1], 0.01 * np.sqrt(0.95)),
    )
