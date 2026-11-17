"""CLV tests on hand-computable cases, including that economic CLV carries the margin paid."""

import pytest

from oddsengine.execution.clv import clv_econ, clv_prob


def test_economic_clv_worked_examples() -> None:
    # took 2.0 / 2.2 / 1.8 on something the close says is a fair 50%
    assert clv_econ(2.0, 0.5) == pytest.approx(0.0)
    assert clv_econ(2.2, 0.5) == pytest.approx(0.10)
    assert clv_econ(1.8, 0.5) == pytest.approx(-0.10)


def test_probability_clv_measures_the_market_moving_towards_us() -> None:
    # fair 50% when taken, 55% at the close: the market came our way by 10%
    assert clv_prob(0.50, 0.55) == pytest.approx(0.10)
    assert clv_prob(0.55, 0.55) == pytest.approx(0.0)


def test_the_two_definitions_differ_by_the_margin_paid() -> None:
    # a margined price on a market that never moves: only economic CLV shows the fee
    assert clv_econ(1.91, 0.5) == pytest.approx(1.91 * 0.5 - 1.0)
    assert clv_econ(1.91, 0.5) < 0.0
    assert clv_prob(0.5, 0.5) == pytest.approx(0.0)


def test_probability_clv_rejects_an_impossible_taken_probability() -> None:
    with pytest.raises(ValueError):
        clv_prob(0.0, 0.5)
