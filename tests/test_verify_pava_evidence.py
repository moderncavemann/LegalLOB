import numpy as np
import torch

from genlob.utils.physics import project_to_valid_lob
from scripts.verify_pava_evidence import (
    independent_invalid_rate,
    price_kkt_residuals,
)


def test_exact_pava_passes_independent_validity_and_kkt_checks():
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(6, 2, 20, 4, generator=generator, dtype=torch.float64)
    projected = project_to_valid_lob(
        raw,
        min_spread=0.2,
        min_gap=0.1,
        use_gpu_approx=False,
    )
    raw_array = raw.numpy()
    projected_array = projected.numpy()

    assert independent_invalid_rate(projected_array, 0.2, 0.1, 1e-10) == 0.0
    residuals = price_kkt_residuals(raw_array, projected_array, 0.2, 0.1)
    assert residuals["primal_violation_max"] < 1e-12
    assert residuals["dual_violation_max"] < 1e-12
    assert residuals["stationarity_sum_abs_max"] < 1e-12
    assert residuals["complementarity_scaled_max"] < 1e-12

    real = project_to_valid_lob(
        torch.randn(6, 2, 20, 4, generator=generator, dtype=torch.float64),
        min_spread=0.2,
        min_gap=0.1,
        use_gpu_approx=False,
    )
    raw_error = torch.square(raw - real).mean()
    projected_error = torch.square(projected - real).mean()
    assert projected_error <= raw_error + 1e-12
