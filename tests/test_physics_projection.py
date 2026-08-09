import torch

from genlob.utils.legal_lob import (
    BoundaryAwareLegalLOBTransform,
    LegalLOBTransform,
    MatchedLOBTransform,
    decode_lob,
    encode_lob,
)
from genlob.utils.physics import (
    _isotonic_regression_pava,
    compute_violation_rate,
    project_to_valid_lob,
)


def test_single_chain_projection_enforces_lob_constraints():
    x = torch.zeros(2, 2, 20, 3)
    x[:, 0, :10, :] = torch.randn(2, 10, 3)
    x[:, 0, 10:, :] = torch.randn(2, 10, 3)
    x[:, 1, :, :] = -1.0

    y = project_to_valid_lob(x, min_spread=1e-4, use_gpu_approx=False)
    report = compute_violation_rate(y, min_spread=1e-4)

    assert report["crossed_book_rate"] == 0.0
    assert report["ask_ladder_violation"] == 0.0
    assert report["bid_ladder_violation"] == 0.0
    assert report["negative_volume_rate"] == 0.0


def test_single_chain_projection_enforces_minimum_level_gap():
    x = torch.randn(2, 2, 20, 3)
    y = project_to_valid_lob(
        x,
        min_spread=0.2,
        min_gap=0.1,
        use_gpu_approx=False,
    )
    report = compute_violation_rate(y, min_spread=0.2, min_gap=0.1)
    assert report["crossed_book_rate"] == 0.0
    assert report["ask_ladder_violation"] == 0.0
    assert report["bid_ladder_violation"] == 0.0


def test_pava_satisfies_isotonic_projection_kkt_conditions():
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(16, 20, generator=generator, dtype=torch.float64)
    projected = _isotonic_regression_pava(values, increasing=True)

    assert torch.all(projected[:, 1:] >= projected[:, :-1] - 1e-12)
    multipliers = torch.cumsum(values - projected, dim=-1)
    assert torch.all(multipliers[:, :-1] >= -1e-10)
    assert torch.allclose(multipliers[:, -1], torch.zeros(16, dtype=torch.float64), atol=1e-10)
    inactive = projected[:, :-1] < projected[:, 1:] - 1e-10
    assert torch.all(torch.abs(multipliers[:, :-1][inactive]) < 1e-10)


def test_compute_violation_rate_uses_min_spread():
    x = torch.zeros(1, 2, 20, 1)
    x[:, 0, :10, :] = torch.arange(10, dtype=torch.float32).view(1, 10, 1) + 1.0
    x[:, 0, 10:, :] = torch.arange(10, 0, -1, dtype=torch.float32).view(1, 10, 1)
    x[:, 0, 0, :] = 1.0000005
    x[:, 0, 10, :] = 1.0

    report = compute_violation_rate(x, min_spread=1e-4)

    assert report["crossed_book_rate"] == 1.0


def test_legal_lob_transform_decode_is_always_valid():
    transform = LegalLOBTransform(min_spread=1e-4)
    params = torch.randn(4, 2, 20, 5)

    lob = transform.decode(params)
    report = compute_violation_rate(lob, min_spread=1e-4)

    assert lob.shape == params.shape
    assert report["crossed_book_rate"] == 0.0
    assert report["ask_ladder_violation"] == 0.0
    assert report["bid_ladder_violation"] == 0.0
    assert report["negative_volume_rate"] == 0.0


def test_legal_lob_transform_remains_valid_for_large_float32_midpoints():
    transform = LegalLOBTransform(min_spread=0.1, gap_eps=0.1)
    params = torch.randn(4, 2, 20, 5) * 100
    lob = transform.decode(params)
    report = compute_violation_rate(lob, min_spread=0.1, min_gap=0.1)
    assert report["crossed_book_rate"] == 0.0
    assert report["ask_ladder_violation"] == 0.0
    assert report["bid_ladder_violation"] == 0.0


def test_legal_lob_transform_round_trips_positive_valid_books():
    transform = LegalLOBTransform(min_spread=1e-4, gap_eps=1e-8)
    params = torch.randn(2, 2, 20, 4)
    lob = transform.decode(params)

    recovered = transform.decode(transform.encode(lob))

    assert torch.allclose(recovered, lob, atol=2e-5, rtol=2e-5)


def test_legal_lob_functional_wrappers_round_trip():
    lob = decode_lob(torch.randn(2, 2, 20, 3), min_spread=1e-4)
    recovered = decode_lob(encode_lob(lob, min_spread=1e-4), min_spread=1e-4)

    assert torch.allclose(recovered, lob, atol=2e-5, rtol=2e-5)


def test_boundary_aware_transform_represents_gap_boundary_with_finite_latent():
    transform = BoundaryAwareLegalLOBTransform(min_spread=0.1, gap_eps=0.1)
    params = torch.zeros(2, 2, 20, 3)
    params[:, 0, 0] = torch.tensor([[-0.4], [0.7]])
    params[:, 0, 1] = 0.25
    params[:, 1] = 0.2

    lob = transform.decode(params)
    encoded = transform.encode(lob)
    report = compute_violation_rate(lob, min_spread=0.1, min_gap=0.1)
    ask_gaps = lob[:, 0, 1:10] - lob[:, 0, :9]
    bid_gaps = lob[:, 0, 10:19] - lob[:, 0, 11:20]

    assert torch.isfinite(encoded).all()
    assert torch.all(ask_gaps <= 0.1 + 2e-5)
    assert torch.all(bid_gaps <= 0.1 + 2e-5)
    assert report["crossed_book_rate"] == 0.0
    assert report["ask_ladder_violation"] == 0.0
    assert report["bid_ladder_violation"] == 0.0
    assert report["negative_volume_rate"] == 0.0


def test_boundary_aware_transform_round_trips_closed_gap_books():
    transform = BoundaryAwareLegalLOBTransform(min_spread=0.1, gap_eps=0.1)
    params = torch.randn(2, 2, 20, 4)
    params[:, 0, 2:20] = torch.where(
        torch.arange(18).view(1, 18, 1) % 2 == 0,
        torch.zeros_like(params[:, 0, 2:20]),
        params[:, 0, 2:20].abs(),
    )
    lob = transform.decode(params)
    recovered = transform.decode(transform.encode(lob))

    assert torch.allclose(recovered, lob, atol=2e-5, rtol=2e-5)


def test_matched_lob_transform_round_trips_without_enforcing_constraints():
    legal = LegalLOBTransform(min_spread=0.1, gap_eps=0.1)
    matched = MatchedLOBTransform()
    lob = legal.decode(torch.randn(2, 2, 20, 3))
    recovered = matched.decode(matched.encode(lob))
    assert torch.allclose(recovered, lob, atol=2e-5, rtol=2e-5)

    params = matched.encode(lob)
    params[:, 0, 1] = -1.0
    params[:, 0, 2] = -0.5
    params[:, 1, 0] = -1.0
    invalid = matched.decode(params)
    report = compute_violation_rate(invalid, min_spread=0.1, min_gap=0.1)
    assert report["crossed_book_rate"] > 0
    assert report["ask_ladder_violation"] > 0
    assert report["negative_volume_rate"] > 0
