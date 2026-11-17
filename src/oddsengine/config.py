"""Settings for the whole engine, loaded from configs/default.yaml.

Every tunable number lives here rather than in the code. Unknown keys are
rejected.
"""

from datetime import timedelta
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# base class that rejects unknown keys
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# settings for the simulation of the slate
class SimConfig(_Strict):
    n_sims: int = Field(default=10_000, gt=0)
    seed: int = 7
    rho: float = Field(default=0.05, ge=0.0, lt=1.0)  # correlation within one group


# limits the optimiser must respect, as fractions of bankroll
class RiskConfig(_Strict):
    max_total_stake: float = Field(default=0.5, gt=0.0, lt=1.0)
    drawdown_floor: float = Field(default=0.7, gt=0.0, lt=1.0)  # never fall below this ...
    drawdown_prob: float = Field(default=0.05, gt=0.0, lt=1.0)  # ... with more than this prob
    tail: float = Field(default=0.95, ge=0.0, lt=1.0)  # tail level for CVaR
    cvar_limit: float | None = Field(default=None, gt=0.0)  # None means no limit
    variance_limit: float | None = Field(default=None, gt=0.0)


# which candidates are worth staking
class DecisionConfig(_Strict):
    ev_min: float = Field(default=0.02, ge=0.0)  # smallest edge we bet
    max_lead: timedelta = timedelta(hours=12)  # latest start we bet on
    # where the fair probability comes from: a name in models.registry.FAIR_PROB_SOURCES
    fair_prob: str = "consensus"


# what the odds archiver polls from the Odds API
class ArchiverConfig(_Strict):
    root: str = "data_local/odds"
    sports: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=lambda: ["uk"])
    markets: list[str] = Field(default_factory=lambda: ["h2h"])


# how one book is used: reference books set the fair probability, venue books
# may be bet into, and both does both. A book missing from config is ignored
class VenueConfig(_Strict):
    role: Literal["reference", "venue", "both"] = "venue"
    # books in one family share one vote; None means its own family
    family: str | None = None
    weight: float = Field(default=1.0, gt=0.0)  # the family's vote, before normalising
    commission: float = Field(default=0.0, ge=0.0, lt=1.0)  # on net winnings (exchanges)
    max_stake: float = Field(default=0.05, gt=0.0, le=1.0)  # per-book cap


# building the consensus and selecting candidates
class ConsensusConfig(_Strict):
    devig_method: Literal["multiplicative", "power"] = "power"
    max_age: timedelta = timedelta(minutes=10)  # oldest usable quote, by its quote time
    max_skew: timedelta = timedelta(seconds=60)  # clock disagreement we tolerate
    min_reference_families: int = Field(default=2, ge=1)  # independent opinions needed
    min_prob: float = Field(default=0.10, ge=0.0, lt=1.0)  # floor against bad longshot estimates
    max_edge: float = Field(default=0.10, gt=0.0)  # edges above this are flagged, never bet
    max_disagreement: float = Field(default=0.08, gt=0.0)  # widest gap from the consensus
    # flat cut to every price for latency
    latency_haircut: float = Field(default=0.01, ge=0.0, lt=1.0)
    max_candidates: int = Field(default=6, ge=1)  # keeps the optimiser fast


# paper trading settings, with stakes and P&L in units of the starting bankroll
class PaperConfig(_Strict):
    bankroll: float = Field(default=1.0, gt=0.0)
    verify_after: timedelta = timedelta(minutes=5)  # re-check a decision's price this much later
    verify_within: timedelta = timedelta(minutes=30)  # ... and no later than this after that
    close_from: timedelta = timedelta(minutes=15)  # closing window opens this long before start
    close_until: timedelta = timedelta(minutes=2)  # ... and shuts this long before start
    min_close_gap: timedelta = timedelta(minutes=10)  # closer closes are ignored


# the live loop (scripts/tick.py): what it polls, how often, and where it writes
class LiveConfig(_Strict):
    sports: list[str] = Field(default_factory=list)  # sports we decide on
    # sports settled with overtime, so draw prices are dropped
    no_draws: list[str] = Field(default_factory=list)
    horizon: timedelta = timedelta(hours=6)  # full polls only while a game starts this soon
    decide_every: timedelta = timedelta(hours=1)  # gap between full polls of one sport
    close_lead: timedelta = timedelta(minutes=10)  # close poll in this last stretch of the window
    settle_after: timedelta = timedelta(hours=4)  # ask for results this long after the start
    settle_every: timedelta = timedelta(hours=6)  # gap between results calls for one sport
    credit_reserve: int = Field(default=30, ge=0)  # below this, no new full polls
    ledger: str = "records/paper_ledger.jsonl"
    flags: str = "data_local/flags.jsonl"  # flagged and disputed picks, never the ledger
    results: str = "data_local/results"


# one sport's Betfair market, the one that settles like the Odds API h2h
class BetfairMarket(_Strict):
    event_type: str  # Betfair event type id: basketball 7522, ice hockey 7524, soccer 1
    market_type: str  # Betfair market type code, e.g. MATCH_ODDS
    countries: list[str] = Field(default_factory=list)  # market countries asked for; empty = any


# Betfair Exchange settings: read-only market data, archived every tick, with a
# trigger for targeted polls of games whose price just moved
class BetfairConfig(_Strict):
    root: str = "data_local/betfair"
    session: str = "data_local/betfair_session.json"  # cached session token, never committed
    markets: dict[str, BetfairMarket] = Field(default_factory=dict)  # by Odds API sport key
    start_tolerance: timedelta = timedelta(minutes=15)  # start times may disagree this much
    aliases: dict[str, str] = Field(default_factory=dict)  # Betfair name -> Odds API name
    keep_alive: timedelta = timedelta(hours=1)  # refresh the cached session this often
    horizon: timedelta = timedelta(hours=6)  # capture (and trigger on) games starting this soon
    max_spread: float = Field(default=0.02, gt=0.0)  # widest back-lay gap trusted, in probability
    move: float = Field(default=0.02, gt=0.0)  # a runner moving this much triggers its game
    compare_within: timedelta = timedelta(minutes=15)  # the capture compared with is this recent
    compare_back: timedelta = timedelta(seconds=270)  # ... and at least this old
    max_triggers: int = Field(default=3, ge=0)  # trigger polls per sport in any 24 hours ...
    hot_reserve: int = Field(default=1, ge=0)  # ... the last of which only go to games ...
    hot_zone: timedelta = timedelta(hours=2)  # ... starting this soon, where team news lands


# the whole config file
class AppConfig(_Strict):
    sim: SimConfig = SimConfig()
    risk: RiskConfig = RiskConfig()
    decision: DecisionConfig = DecisionConfig()
    archiver: ArchiverConfig = ArchiverConfig()
    consensus: ConsensusConfig = ConsensusConfig()
    paper: PaperConfig = PaperConfig()
    live: LiveConfig = LiveConfig()
    betfair: BetfairConfig = BetfairConfig()
    venues: dict[str, VenueConfig] = Field(default_factory=dict)

    # checks that books in one family agree on its weight
    @model_validator(mode="after")
    def _family_weights_agree(self) -> Self:
        seen: dict[str, float] = {}
        for book, venue in self.venues.items():
            family = venue.family or book
            # the weight the family's first member gave it
            first = seen.setdefault(family, venue.weight)
            if first != venue.weight:
                raise ValueError(f"family {family!r} has members with different weights")
        return self


def load_config(path: str | Path) -> AppConfig:
    # read and validate the YAML file
    return AppConfig.model_validate(yaml.safe_load(Path(path).read_text()))


# secrets come from the environment or a local .env file
class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    odds_api_key: SecretStr
    # optional, without them the live loop skips Betfair
    betfair_app_key: SecretStr | None = None
    betfair_username: SecretStr | None = None
    betfair_password: SecretStr | None = None

    def betfair_login(self) -> tuple[str, str, str] | None:
        # returns the app key, username and password, or None if any is missing
        key, user, password = self.betfair_app_key, self.betfair_username, self.betfair_password
        if key is None or user is None or password is None:
            return None
        return key.get_secret_value(), user.get_secret_value(), password.get_secret_value()
