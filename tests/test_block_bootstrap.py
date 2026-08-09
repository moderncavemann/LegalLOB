import numpy as np
import pytest

from scripts.block_bootstrap_ablation import moving_block_means


def test_grouped_bootstrap_preserves_symbol_day_composition():
    diff = np.concatenate([np.zeros(8), np.full(8, 10.0)])
    groups = np.concatenate([np.zeros(8, dtype=int), np.ones(8, dtype=int)])

    boot = moving_block_means(diff, block_size=3, n_boot=100, seed=7, groups=groups)

    assert np.all(boot == pytest.approx(5.0))


def test_grouped_bootstrap_is_reproducible_and_validates_alignment():
    diff = np.arange(12, dtype=float)
    groups = np.repeat(np.arange(3), 4)

    first = moving_block_means(diff, block_size=2, n_boot=25, seed=11, groups=groups)
    second = moving_block_means(diff, block_size=2, n_boot=25, seed=11, groups=groups)

    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="groups must align"):
        moving_block_means(diff, block_size=2, n_boot=1, seed=0, groups=groups[:-1])
