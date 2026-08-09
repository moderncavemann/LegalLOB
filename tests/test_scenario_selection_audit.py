import numpy as np

from scripts.audit_scenario_selection import _kendall_tau_b, compare_rankings


def test_kendall_tau_b_handles_order_and_ties():
    assert _kendall_tau_b(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 1.0
    assert _kendall_tau_b(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == -1.0
    assert np.isfinite(
        _kendall_tau_b(np.array([np.inf, np.inf, 1.0]), np.array([np.inf, 2.0, 1.0]))
    )


def test_compare_rankings_detects_selection_change():
    pre = np.array([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    post = np.array([[3.0, 4.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    result = compare_rankings(pre, post, top_k=2)
    assert result["worst_candidate_flip_rate"] == 0.5
    assert result["top_k_membership_change_rate"] == 0.0
    assert result["top_k_jaccard_mean"] == 1.0


def test_compare_rankings_treats_equal_infinite_selection_as_zero_shift():
    scores = np.array([[np.inf, np.inf, 1.0]])
    result = compare_rankings(scores, scores.copy(), top_k=2)
    assert result["worst_candidate_flip_rate"] == 0.0
    assert result["selected_final_risk_gain_mean"] == 0.0
    assert result["selected_repair_shift_mean"] == 0.0
