"""Model tests: select and decide price games with whichever model they are given, and
the config picks it by name.
"""

from datetime import UTC, datetime, timedelta

import pytest

from oddsengine.config import AppConfig, ConsensusConfig, DecisionConfig, VenueConfig
from oddsengine.execution.trader import TraderState, decide
from oddsengine.models.base import Pooled
from oddsengine.models.registry import fair_prob_source
from oddsengine.selection.select import select
from oddsengine.types import EventInfo, QuoteSnapshot

T0 = datetime(2026, 11, 6, 19, 0, tzinfo=UTC)
VENUES = {
    "pinnacle": VenueConfig(role="reference"),
    "ref2": VenueConfig(role="reference"),
    "bookA": VenueConfig(role="venue"),
}
EVENTS = {
    "e1": EventInfo(event_id="e1", sport="basketball_nba", commence_utc=T0 + timedelta(hours=2))
}


def config(fair_prob: str = "consensus") -> AppConfig:
    # margin-free references, so each devigs to exactly the probability it was built from
    return AppConfig(
        decision=DecisionConfig(ev_min=0.01, fair_prob=fair_prob),
        consensus=ConsensusConfig(devig_method="multiplicative", latency_haircut=0.0),
        venues=VENUES,
    )


def fair(p_home: float) -> dict[str, float]:
    return {"home": 1.0 / p_home, "away": 1.0 / (1.0 - p_home)}


def snap() -> QuoteSnapshot:
    # pinnacle says 0.54, ref2 says 0.46 (an even panel gives 0.50); bookA offers home at 2.00
    prices = {"pinnacle": fair(0.54), "ref2": fair(0.46), "bookA": {"home": 2.00, "away": 1.80}}
    return QuoteSnapshot(
        market_id="e1:h2h",
        event_id="e1",
        observed_at_utc=T0,
        prices=prices,
        quoted_at_utc=dict.fromkeys(prices, T0),
    )


# a stand-in model that says the same thing about every game, and remembers what it saw
class Fixed:
    def __init__(self, p_home: float) -> None:
        self.p_home = p_home
        self.seen: list[EventInfo] = []

    def fair_probs(
        self, snapshot: QuoteSnapshot, event: EventInfo, *, exclude: str | None = None
    ) -> Pooled | None:
        self.seen.append(event)
        return Pooled(probs={"home": self.p_home, "away": 1 - self.p_home}, n_families=2)


def picks(source: Fixed | None = None) -> list[tuple[str, float]]:
    # (selection, fair probability) of every pick select makes on snap()
    cfg = config()
    found = select(
        [snap()],
        events=EVENTS,
        venues=cfg.venues,
        cfg=cfg.consensus,
        ev_min=0.01,
        as_of=T0,
        source=source,
    )
    return [(p.selection, p.p_consensus) for p in found.picks]


def test_the_default_model_is_the_panel_consensus() -> None:
    # panel 0.50 at 2.00: no edge on home, and away at 1.80 is worse
    assert picks() == []


def test_select_prices_games_with_the_model_it_is_given() -> None:
    # a model saying 0.53 makes home at 2.00 a 6% edge
    assert picks(Fixed(0.53)) == [("home", 0.53)]


def test_a_model_is_given_the_games_event() -> None:
    model = Fixed(0.53)
    picks(model)
    assert model.seen == [EVENTS["e1"]]


def test_decide_uses_the_model_the_config_names() -> None:
    # pinnacle alone says 0.54, so home at 2.00 is an 8% edge; the panel sees none
    empty = TraderState(bankroll=1)
    assert decide([snap()], events=EVENTS, state=empty, config=config(), as_of=T0).orders == []
    [order] = decide(
        [snap()], events=EVENTS, state=empty, config=config("pinnacle"), as_of=T0
    ).orders
    assert (order.selection, order.p_consensus, order.strategy) == (
        "home",
        pytest.approx(0.54),
        "pinnacle",
    )


def test_an_unknown_model_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown fair_prob"):
        fair_prob_source(config("coin_flip"))
