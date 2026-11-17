"""The portfolio stage: candidate bets in, a staked and risk-checked slate out.

It simulates the candidates' joint outcomes, chooses every stake at once under
the risk limits and each book's cap, then drops legs left at zero.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime

from oddsengine.config import AppConfig
from oddsengine.portfolio.optimiser import optimise_slate
from oddsengine.portfolio.sim import SimLeg, simulate_returns
from oddsengine.types import BetProposal, Candidate, SlateDecision

# stakes at or below this are too small to count
MIN_STAKE = 1e-6


def build_slate(
    candidates: Sequence[Candidate],
    *,
    config: AppConfig,
    as_of: datetime,
    room: Mapping[str, float] | None = None,
) -> SlateDecision:
    # room is how much more each book may take before its cap

    # no candidates: empty slate
    if not candidates:
        return SlateDecision(as_of=as_of)

    # simulates joint outcomes of the candidates
    legs = [
        SimLeg(prob=c.fair_prob, price=c.quoted_price, event_id=c.event_id, group=c.group)
        for c in candidates
    ]
    sim = config.sim
    returns = simulate_returns(legs, n_sims=sim.n_sims, seed=sim.seed, rho=sim.rho)

    # chooses every stake at once
    books = [c.book for c in candidates]
    stakes = optimise_slate(returns, config.risk, books=books, room=room or {})

    # builds the bets, dropping zero stakes
    proposals = tuple(
        BetProposal(
            market_id=c.market_id,
            selection=c.selection,
            fair_prob=c.fair_prob,
            quoted_price=c.quoted_price,
            edge=c.fair_prob * c.quoted_price - 1.0,
            stake=float(stake),
        )
        for c, stake in zip(candidates, stakes, strict=True)
        if stake > MIN_STAKE
    )
    return SlateDecision(as_of=as_of, proposals=proposals)
