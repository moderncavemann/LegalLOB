import torch

from genlob.data.action_templates import make_counterfactual_action


def test_nasdaq_diagnostic_buy_sell_templates_are_informative_not_direction_only():
    buy = make_counterfactual_action(1.0, 3, torch.device("cpu"), "nasdaq_INTC_9day")
    sell = make_counterfactual_action(-1.0, 3, torch.device("cpu"), "nasdaq_INTC_9day")

    assert buy.shape == (3, 5)
    assert sell.shape == (3, 5)
    assert torch.all(buy[:, 0] > 0)
    assert torch.all(sell[:, 0] < 0)
    assert torch.all(buy[:, 1:] > 0)
    assert torch.all(sell[:, 1:] > 0)


def test_nasdaq_diagnostic_hold_template_keeps_duration_anchor_only():
    hold = make_counterfactual_action(0.0, 2, torch.device("cpu"), "nasdaq_INTC_9day")

    assert torch.allclose(hold[:, 0], torch.zeros(2))
    assert torch.allclose(hold[:, 1], torch.zeros(2))
    assert torch.allclose(hold[:, 2], torch.zeros(2))
    assert torch.allclose(hold[:, 3], torch.full((2,), 0.5))
    assert torch.allclose(hold[:, 4], torch.zeros(2))
