"""Optimiser tests: Kelly and drawdown behaviour, limit checks on the returned stakes, then the
solver guard.
"""

import cvxpy as cp
import numpy as np
import pytest
from pydantic import ValidationError

from oddsengine.config import RiskConfig
from oddsengine.portfolio.optimiser import optimise_slate, solve
from oddsengine.portfolio.risk import cvar
from oddsengine.portfolio.sim import SimLeg, simulate_returns
from oddsengine.types import FloatArray


def slate(n_legs: int = 5, prob: float = 0.58, rho: float = 0.05) -> FloatArray:
    legs = [SimLeg(prob=prob, price=2.0, event_id=f"e{i}") for i in range(n_legs)]
    return simulate_returns(legs, n_sims=4000, seed=4, rho=rho)


def exact_bet(prob: float, price: float, n: int = 200) -> FloatArray:
    # one leg winning in exactly prob of the worlds, so it matches closed-form Kelly
    column = np.full((n, 1), -1.0)
    column[: round(prob * n), 0] = price - 1.0
    return column


def test_with_no_drawdown_promise_one_bet_gets_full_kelly() -> None:
    # Kelly stake: (0.55 * 2 - 1) / (2 - 1) = 0.10
    stakes = optimise_slate(exact_bet(0.55, 2.0), RiskConfig(drawdown_prob=0.999999))
    assert stakes[0] == pytest.approx(0.10, abs=0.005)


def test_a_tighter_drawdown_promise_stakes_less() -> None:
    bet = exact_bet(0.55, 2.0)
    loose, mid, tight = (
        optimise_slate(bet, RiskConfig(drawdown_prob=b))[0] for b in (0.5, 0.1, 0.01)
    )
    assert loose > mid > tight >= 0.0


def test_a_bet_with_no_edge_is_not_staked() -> None:
    assert optimise_slate(exact_bet(0.40, 2.0), RiskConfig())[0] == pytest.approx(0.0, abs=1e-6)
    assert optimise_slate(slate(prob=0.40), RiskConfig()).sum() == pytest.approx(0.0, abs=1e-6)


def test_one_nonnegative_stake_per_leg_within_the_budget() -> None:
    bets = np.hstack([exact_bet(0.80, 2.0), exact_bet(0.80, 2.0)])
    stakes = optimise_slate(bets, RiskConfig(drawdown_prob=0.5, max_total_stake=0.10))
    assert stakes.shape == (2,)
    assert np.all(stakes >= 0.0)
    assert stakes.sum() <= 0.10 + 1e-6


def test_correlated_legs_get_less_total_stake() -> None:
    calm = optimise_slate(slate(rho=0.0), RiskConfig()).sum()
    linked = optimise_slate(slate(rho=0.6), RiskConfig()).sum()
    assert linked < calm


def test_a_budget_that_allows_total_ruin_is_rejected() -> None:
    # wealth must stay positive when every leg loses
    with pytest.raises(ValidationError):
        RiskConfig(max_total_stake=1.0)


def test_a_cvar_limit_is_honoured_by_the_solution() -> None:
    returns = slate()
    stakes = optimise_slate(returns, RiskConfig(drawdown_prob=0.5, cvar_limit=0.05))
    assert cvar(-(returns @ stakes), 0.95) <= 0.05 + 1e-3


def test_a_variance_limit_is_honoured_by_the_solution() -> None:
    returns = slate()
    stakes = optimise_slate(returns, RiskConfig(drawdown_prob=0.5, variance_limit=0.0008))
    assert float(np.var(returns @ stakes)) <= 0.0008 + 1e-6


def test_tightening_a_limit_reduces_exposure() -> None:
    returns = slate()
    loose = optimise_slate(returns, RiskConfig(drawdown_prob=0.5, cvar_limit=0.20)).sum()
    tight = optimise_slate(returns, RiskConfig(drawdown_prob=0.5, cvar_limit=0.03)).sum()
    assert tight < loose


def test_limits_compose() -> None:
    returns = slate()
    one = optimise_slate(returns, RiskConfig(drawdown_prob=0.5, cvar_limit=0.05))
    both = optimise_slate(
        returns, RiskConfig(drawdown_prob=0.5, cvar_limit=0.05, variance_limit=0.0008)
    )
    assert both.sum() <= one.sum() + 1e-6


def test_solve_returns_stakes_that_meet_the_constraints() -> None:
    w = cp.Variable(2, nonneg=True)
    problem = cp.Problem(cp.Maximize(cp.sum(w)), [cp.sum(w) <= 0.3])
    assert solve(problem, w).sum() == pytest.approx(0.3, abs=1e-6)


def test_an_infeasible_problem_raises_rather_than_returning_junk() -> None:
    w = cp.Variable(1, nonneg=True)
    problem = cp.Problem(cp.Maximize(cp.sum(w)), [cp.sum(w) <= 0.1, cp.sum(w) >= 0.5])
    with pytest.raises(RuntimeError, match="feasible"):
        solve(problem, w)
