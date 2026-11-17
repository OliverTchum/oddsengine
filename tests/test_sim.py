"""Simulator tests: input checks, leg win rates, group independence, the all-lose tail and the
matrix formula.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from oddsengine.portfolio.sim import SimLeg, simulate_returns


def leg(prob: float, event_id: str, *, price: float = 1.909, group: str = "") -> SimLeg:
    return SimLeg(prob=prob, price=price, event_id=event_id, group=group)


def win_corr(returns: np.ndarray, i: int, j: int) -> float:
    wins = (returns > 0).astype(np.float64)
    return float(np.corrcoef(wins[:, i], wins[:, j])[0, 1])


def test_returns_have_one_column_per_leg_with_win_and_loss_values() -> None:
    legs = [leg(0.55, "e1", price=2.0), leg(0.40, "e2", price=3.5)]
    returns = simulate_returns(legs, n_sims=500, seed=1)
    assert returns.shape == (500, 2)
    assert set(np.unique(returns[:, 0])) <= {1.0, -1.0}
    assert set(np.unique(returns[:, 1])) <= {2.5, -1.0}


def test_the_same_seed_gives_the_same_matrix() -> None:
    legs = [leg(0.5, "e1"), leg(0.5, "e2")]
    a = simulate_returns(legs, n_sims=2000, seed=42)
    b = simulate_returns(legs, n_sims=2000, seed=42)
    c = simulate_returns(legs, n_sims=2000, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        simulate_returns([], n_sims=100, seed=0)
    with pytest.raises(ValueError):
        simulate_returns([leg(0.5, "e1")], n_sims=0, seed=0)
    with pytest.raises(ValueError):
        simulate_returns([leg(0.5, "e1")], n_sims=100, seed=0, rho=1.0)
    with pytest.raises(ValidationError):
        leg(1.5, "e1")
    with pytest.raises(ValidationError):
        leg(0.5, "e1", price=0.8)


def test_an_event_can_only_appear_once() -> None:
    # two legs on one event is an upstream bug
    with pytest.raises(ValueError, match="more than once"):
        simulate_returns([leg(0.5, "e1"), leg(0.4, "e1")], n_sims=10, seed=0)


def test_each_leg_wins_at_its_own_probability() -> None:
    # correlation must never change a single leg's win rate
    legs = [leg(0.60, "e1", group="a"), leg(0.25, "e2", group="b"), leg(0.85, "e3", group="a")]
    returns = simulate_returns(legs, n_sims=20_000, seed=7, rho=0.3)
    assert (returns > 0).mean(axis=0).tolist() == [0.60, 0.25, 0.85]


def test_sampling_noise_never_turns_an_edge_negative() -> None:
    # p = 0.1269 at 8.00 is a 1.5% edge, but a Phi(z) comparison won only 12.27% of worlds, below 1
    # / 8
    returns = simulate_returns([leg(0.1269, "e1", price=8.0)], n_sims=10_000, seed=7)
    assert returns.mean() == pytest.approx(0.1269 * 8.0 - 1.0)


def test_rho_zero_gives_independent_legs() -> None:
    legs = [leg(0.5, "e1"), leg(0.5, "e2")]
    returns = simulate_returns(legs, n_sims=100_000, seed=11, rho=0.0)
    assert win_corr(returns, 0, 1) == pytest.approx(0.0, abs=0.02)


def test_legs_in_one_group_move_together() -> None:
    legs = [leg(0.5, "e1"), leg(0.5, "e2")]
    returns = simulate_returns(legs, n_sims=100_000, seed=11, rho=0.9)
    assert win_corr(returns, 0, 1) > 0.5


def test_different_groups_are_independent_even_at_high_rho() -> None:
    legs = [leg(0.5, "e1", group="nba"), leg(0.5, "e2", group="epl")]
    returns = simulate_returns(legs, n_sims=100_000, seed=11, rho=0.9)
    assert abs(win_corr(returns, 0, 1)) < 0.02


def test_small_rho_multiplies_the_all_lose_tail_severalfold() -> None:
    # ten legs at p = 0.55: independent all-lose is 0.45^10 = 3.4e-4, at rho 0.07 it is 2.00e-3
    legs = [leg(0.55, f"e{i}") for i in range(10)]
    indep = simulate_returns(legs, n_sims=400_000, seed=17, rho=0.0)
    linked = simulate_returns(legs, n_sims=400_000, seed=17, rho=0.07)
    p_indep = float(np.all(indep < 0, axis=1).mean())
    p_linked = float(np.all(linked < 0, axis=1).mean())
    assert p_indep == pytest.approx(0.45**10, rel=0.35)
    assert p_linked == pytest.approx(2.001e-3, rel=0.20)
    assert 4.0 < p_linked / p_indep < 8.0


def test_the_matrix_is_exactly_the_copula_formula() -> None:
    # rebuilds the matrix by hand with the same seed and two groups
    legs = [
        leg(0.55, "e1", group="nba"),
        leg(0.40, "e2", group="nba"),
        leg(0.70, "e3", group="epl"),
    ]
    n, seed, rho = 5000, 9, 0.2
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n, 2))
    private = rng.standard_normal((n, 3))
    z = np.sqrt(rho) * factors[:, [0, 0, 1]] + np.sqrt(1.0 - rho) * private
    # a leg wins in its round(p * n) lowest worlds
    cutoff = np.sort(z, axis=0)[np.round(np.array([0.55, 0.40, 0.70]) * n).astype(int), [0, 1, 2]]
    expected = np.where(z < cutoff, 1.909 - 1.0, -1.0)
    assert np.array_equal(simulate_returns(legs, n_sims=n, seed=seed, rho=rho), expected)
