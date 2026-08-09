from experiments.multi_asset_event_time.train_fixed_steps import (
    deterministic_uniform_indices,
)


def test_deterministic_uniform_indices_span_and_are_unique():
    indices = deterministic_uniform_indices(101, 11)

    assert indices[0] == 0
    assert indices[-1] == 100
    assert indices == sorted(set(indices))
    assert len(indices) == 11


def test_deterministic_uniform_indices_clip_to_dataset():
    assert deterministic_uniform_indices(3, 10) == [0, 1, 2]


def test_deterministic_uniform_indices_singleton_uses_middle():
    assert deterministic_uniform_indices(10, 1) == [5]
