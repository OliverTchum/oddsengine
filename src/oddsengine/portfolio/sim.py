"""Monte Carlo simulation of a slate's joint outcomes.

Legs in one correlation group (one sport on one date) move together through a
shared factor, and legs in different groups are independent. Each leg wins in
exactly its own share of the simulated worlds.
"""

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from oddsengine.types import FloatArray


# one bet in the slate: its win probability, price, event and correlation group
class SimLeg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prob: float = Field(gt=0.0, lt=1.0)
    price: float = Field(gt=1.0)
    event_id: str
    group: str = ""


def simulate_returns(
    legs: Sequence[SimLeg], *, n_sims: int, seed: int, rho: float = 0.05
) -> FloatArray:
    # returns per unit staked, price minus 1 on a win and -1 on a loss
    if not legs:
        raise ValueError("need at least one leg")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be at least 0 and below 1")

    # we bet at most one selection per event, so an event can only appear once
    events = [leg.event_id for leg in legs]
    if len(set(events)) < len(events):
        raise ValueError("an event appears more than once; a slate has one selection per event")

    # the distinct groups in first-seen order, and which factor column each leg reads from
    groups = list(dict.fromkeys(leg.group for leg in legs))
    column = [groups.index(leg.group) for leg in legs]

    # common factors first, then one private factor per leg
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n_sims, len(groups)))
    private = rng.standard_normal((n_sims, len(legs)))
    z = np.sqrt(rho) * factors[:, column] + np.sqrt(1.0 - rho) * private

    # each leg wins in exactly its share of worlds: those where its score is lowest
    probs = np.array([leg.prob for leg in legs])
    prices = np.array([leg.price for leg in legs])

    # ranks each world within a leg's column, lowest score first
    rank = z.argsort(axis=0).argsort(axis=0)
    return np.where(rank < np.round(probs * n_sims), prices - 1.0, -1.0)
