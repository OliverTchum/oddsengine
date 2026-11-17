"""Risk maths on hand-computable cases: each one can be worked out with a pen."""

import numpy as np
import pytest

from oddsengine.portfolio.risk import cvar


def test_cvar_on_a_clean_boundary_is_the_mean_of_the_worst_slice() -> None:
    # losses 1..10 at alpha 0.80: the worst 20% is exactly {9, 10}, mean 9.5
    assert cvar(np.arange(1.0, 11.0), alpha=0.80) == pytest.approx(9.5)


def test_cvar_gives_the_boundary_sample_partial_weight() -> None:
    # alpha 0.85 cuts between samples: (10/10 + (9/10 - 0.85) * 9) / 0.15 = 9.6667
    assert cvar(np.arange(1.0, 11.0), alpha=0.85) == pytest.approx(9.666667, abs=1e-5)


def test_cvar_at_alpha_zero_is_the_mean() -> None:
    assert cvar(np.arange(1.0, 11.0), alpha=0.0) == pytest.approx(5.5)


def test_cvar_near_alpha_one_is_the_worst_loss() -> None:
    assert cvar(np.array([1.0, 5.0, 100.0]), alpha=0.999) == pytest.approx(100.0)


def test_cvar_is_at_least_var() -> None:
    losses = np.random.default_rng(0).normal(size=5000)
    assert cvar(losses, alpha=0.95) >= float(np.quantile(losses, 0.95))


def test_cvar_ignores_sample_order() -> None:
    losses = np.arange(1.0, 11.0)
    assert cvar(losses[::-1].copy(), alpha=0.9) == pytest.approx(cvar(losses, alpha=0.9))


def test_cvar_rejects_alpha_one() -> None:
    with pytest.raises(ValueError):
        cvar(np.arange(5.0), alpha=1.0)


def test_cvar_never_punishes_diversification() -> None:
    # subadditivity, the property VaR lacks
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=20_000), rng.normal(size=20_000)
    combined = cvar(0.5 * a + 0.5 * b, alpha=0.95)
    assert combined <= 0.5 * cvar(a, alpha=0.95) + 0.5 * cvar(b, alpha=0.95) + 1e-9
