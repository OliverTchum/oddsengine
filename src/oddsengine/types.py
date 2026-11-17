"""Data passed between the stages of the engine.

Frozen pydantic models: validated when created, fields cannot be reassigned,
unknown fields rejected.
"""

from datetime import datetime

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

FloatArray = npt.NDArray[np.float64]


# base class: immutable, unknown fields rejected
class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# every book's prices for one market when we captured them. observed_at_utc is
# when we captured it, quoted_at_utc is when each book quoted it (empty if unknown)
class QuoteSnapshot(_Frozen):
    market_id: str
    event_id: str
    observed_at_utc: datetime
    prices: dict[str, dict[str, float]] = Field(default_factory=dict)  # {book: {selection: price}}
    quoted_at_utc: dict[str, datetime] = Field(default_factory=dict)  # {book: quote time}


# one bet the portfolio stage may choose to stake
class Candidate(_Frozen):
    market_id: str
    event_id: str
    group: str = ""  # correlation group for the simulator, e.g. "basketball_nba:2026-11-06"
    book: str = ""  # where the bet would be placed, for the per-book cap
    selection: str
    fair_prob: float = Field(gt=0.0, lt=1.0)  # consensus probability
    quoted_price: float = Field(gt=1.0)  # the price we can take, after commission and haircut


# a candidate that made the slate, with its stake
class BetProposal(_Frozen):
    market_id: str
    selection: str
    fair_prob: float
    quoted_price: float
    edge: float
    stake: float = Field(ge=0.0)  # fraction of bankroll


# one slate: the bets the optimiser staked
class SlateDecision(_Frozen):
    as_of: datetime
    proposals: tuple[BetProposal, ...] = ()


# one scheduled event; the team names are the Odds API's
class EventInfo(_Frozen):
    event_id: str
    sport: str  # e.g. "basketball_nba"
    commence_utc: datetime
    home_team: str = ""
    away_team: str = ""


# how an event ended: the selection that won, e.g. a team name or "Draw"
class Result(_Frozen):
    event_id: str
    winner: str
