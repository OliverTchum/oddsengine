"""Devig tests: worked examples first, then hypothesis properties for any valid book."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from oddsengine.models.devig import DEVIG_METHODS, devig_multiplicative, devig_power

METHOD_IDS = sorted(DEVIG_METHODS)
METHODS = [DEVIG_METHODS[name] for name in METHOD_IDS]

# any valid book: 2 to 8 prices between 1.05 and 50
ODDS_BOOKS = st.lists(
    st.floats(min_value=1.05, max_value=50.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=8,
)


def test_multiplicative_worked_example() -> None:
    # reciprocals [0.6667, 0.4000] sum to 1.0667, so they scale to [0.625, 0.375]
    assert devig_multiplicative([1.50, 2.50]) == pytest.approx([0.625, 0.375], abs=1e-9)


def test_multiplicative_three_way_book() -> None:
    # a football 1X2 book
    probs = devig_multiplicative([2.70, 3.25, 2.90])
    assert probs == pytest.approx([0.36208, 0.30081, 0.33711], abs=1e-4)


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_a_symmetric_book_is_fifty_fifty(method) -> None:  # type: ignore[no-untyped-def]
    assert method([1.91, 1.91]) == pytest.approx([0.5, 0.5], abs=1e-9)


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
def test_a_fair_book_comes_back_unchanged(method) -> None:  # type: ignore[no-untyped-def]
    # implied probabilities already sum to 1
    assert method([4.0, 4.0 / 3.0]) == pytest.approx([0.25, 0.75], abs=1e-9)


def test_power_shades_the_longshot_harder_than_multiplicative() -> None:
    # power gives a heavy favourite more than multiplicative does
    odds = [1.15, 5.50]
    p_mult = devig_multiplicative(odds)
    p_power = devig_power(odds)
    assert p_power[0] > p_mult[0] + 1e-6
    assert p_power[1] < p_mult[1] - 1e-6


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize(
    "bad", [[1.5], [0.9, 2.0], [2.0, 1.0], []], ids=["single", "below_one", "one", "empty"]
)
def test_invalid_input_raises(method, bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        method(bad)


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
@settings(deadline=None, max_examples=50)
@given(odds=ODDS_BOOKS)
def test_output_is_a_probability_vector(method, odds) -> None:  # type: ignore[no-untyped-def]
    probs = method(odds)
    assert probs.shape == (len(odds),)
    assert float(np.sum(probs)) == pytest.approx(1.0, abs=1e-6)
    assert np.all((probs > 0.0) & (probs < 1.0))


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
@settings(deadline=None, max_examples=50)
@given(odds=ODDS_BOOKS)
def test_shorter_odds_never_get_less_probability(method, odds) -> None:  # type: ignore[no-untyped-def]
    probs = method(odds)
    for i in range(len(odds)):
        for j in range(len(odds)):
            if odds[i] < odds[j] - 1e-9:
                assert probs[i] > probs[j] - 1e-9


@pytest.mark.parametrize("method", METHODS, ids=METHOD_IDS)
@settings(deadline=None, max_examples=50)
@given(
    weights=st.lists(
        st.floats(min_value=0.05, max_value=1.0, allow_nan=False), min_size=2, max_size=6
    )
)
def test_any_fair_book_comes_back_unchanged(method, weights) -> None:  # type: ignore[no-untyped-def]
    true_probs = np.asarray(weights, dtype=np.float64)
    true_probs /= true_probs.sum()
    assert method(list(1.0 / true_probs)) == pytest.approx(list(true_probs), abs=1e-6)


@settings(deadline=None, max_examples=50)
@given(odds=ODDS_BOOKS)
def test_power_output_is_a_true_power_of_the_implieds(odds) -> None:  # type: ignore[no-untyped-def]
    # log(p) / log(1 / o) is the same k for every outcome
    probs = devig_power(odds)
    raw = 1.0 / np.asarray(odds, dtype=np.float64)
    k = np.log(probs) / np.log(raw)
    assert float(np.max(k) - np.min(k)) == pytest.approx(0.0, abs=1e-5)
