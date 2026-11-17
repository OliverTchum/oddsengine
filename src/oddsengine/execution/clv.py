"""Closing-line value: did we get a better price than the market settled on?

Economic CLV is what the bet was worth at the closing fair probability, margin
and commission included, and is the headline. Probability CLV is how far the
fair probability moved towards us.
"""


def clv_econ(price_taken: float, p_close: float) -> float:
    # the bet's worth at the closing probability
    return price_taken * p_close - 1.0


def clv_prob(p_taken: float, p_close: float) -> float:
    # how far the probability moved from the one taken
    if p_taken <= 0.0:
        raise ValueError("p_taken must be positive")
    return p_close / p_taken - 1.0
