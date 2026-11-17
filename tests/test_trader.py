"""Trader tests in the order an order lives: decide, verify, close, settle, summarise, then
bankroll properties.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from oddsengine.config import (
    AppConfig,
    ConsensusConfig,
    DecisionConfig,
    PaperConfig,
    RiskConfig,
    SimConfig,
    VenueConfig,
)
from oddsengine.execution.ledger import Ledger
from oddsengine.execution.trader import (
    STAKE_STEP,
    Closed,
    Decided,
    Filled,
    Missed,
    Position,
    Settled,
    TraderState,
    close_payload,
    decidable,
    decide,
    digest,
    fold,
    settle_payload,
    summarise,
    verify,
)
from oddsengine.selection.select import net_price
from oddsengine.types import EventInfo, QuoteSnapshot, Result

T0 = datetime(2026, 11, 6, 19, 0, tzinfo=UTC)  # the decision tick
START = T0 + timedelta(hours=2)  # every event's start unless told otherwise
MIN = timedelta(minutes=1)

# two independent reference families, two soft books and an exchange
VENUES = {
    "ref1": VenueConfig(role="reference"),
    "ref2": VenueConfig(role="reference"),
    "bookA": VenueConfig(role="venue"),
    "bookB": VenueConfig(role="venue"),
    "exch": VenueConfig(role="venue", commission=0.05),
}

# bookA's home price against a fair 50%: edge 0.5 * 2.15 * 0.99 - 1 = 0.06425
STALE = {"bookA": {"home": 2.15, "away": 1.75}}


def config(*, risk: RiskConfig | None = None, **venues: VenueConfig) -> AppConfig:
    # hand-computable consensus and a small fast simulation
    return AppConfig(
        sim=SimConfig(n_sims=2000, seed=11),
        risk=risk or RiskConfig(),
        decision=DecisionConfig(ev_min=0.01),
        consensus=ConsensusConfig(devig_method="multiplicative", min_reference_families=2),
        paper=PaperConfig(),
        venues={**VENUES, **venues},
    )


def fair(p_home: float) -> dict[str, float]:
    # margin-free prices that devig back to exactly p_home
    return {"home": 1.0 / p_home, "away": 1.0 / (1.0 - p_home)}


def snap(
    event: str = "e1",
    *,
    at: datetime = T0,
    venue: dict[str, dict[str, float]] | None = None,
    p: float = 0.5,
    refs: tuple[str, ...] = ("ref1", "ref2"),
) -> QuoteSnapshot:
    # one h2h market with reference books at fair p plus venue prices, all quoted at `at`
    prices = {ref: fair(p) for ref in refs} | (STALE if venue is None else venue)
    return QuoteSnapshot(
        market_id=f"{event}:h2h",
        event_id=event,
        observed_at_utc=at,
        prices=prices,
        quoted_at_utc=dict.fromkeys(prices, at),
    )


def events(*ids: str, start: datetime = START) -> dict[str, EventInfo]:
    return {e: EventInfo(event_id=e, sport="basketball_nba", commence_utc=start) for e in ids}


def decision(
    order_id: str = "o1",
    *,
    event: str = "e9",
    book: str = "bookA",
    selection: str = "home",
    price: float = 2.20,
    price_net: float | None = None,
    p: float = 0.5,
    stake: float | Decimal = 0.02,
    decided_at: datetime = T0,
    start: datetime = START,
) -> Decided:
    # hand-made decision record, for tests that start after the decision
    net = price if price_net is None else price_net
    return Decided(
        order_id=order_id,
        event_id=event,
        market_id=f"{event}:h2h",
        selection=selection,
        book=book,
        price=price,
        price_net=net,
        p_consensus=p,
        n_families=2,
        edge=p * net * 0.99 - 1,
        latency_haircut=0.01,
        stake=stake,
        decided_at=decided_at,
        commence_utc=start,
        config_digest="c" * 64,
        snapshot_digest="s" * 64,
    )


def filled(d: Decided) -> Filled:
    # verified five minutes later at the decision price
    return Filled(
        order_id=d.order_id,
        observed_at=d.decided_at + 5 * MIN,
        price=d.price,
        price_net=d.price_net,
    )


def missed(d: Decided) -> Missed:
    return Missed(order_id=d.order_id, observed_at=d.decided_at + 5 * MIN, reason="no_quote")


def state(*positions: Position) -> TraderState:
    # the state rebuilt from these positions, with nothing settled
    return TraderState(
        bankroll=1.0,
        positions={p.decision.order_id: p for p in positions},
        reserved=sum(p.decision.stake for p in positions if p.fill is None and p.miss is None),
        at_risk=sum(p.decision.stake for p in positions if p.fill is not None and p.settle is None),
    )


def payloads(*records: Decided | Filled | Missed | Closed | Settled) -> list[dict[str, object]]:
    # records exactly as the ledger stores them
    return [r.model_dump(mode="json") for r in records]


def test_a_decision_records_everything_needed_to_audit_it() -> None:
    cfg, s = config(), snap()
    [d] = decide([s], events=events("e1"), state=state(), config=cfg, as_of=T0).orders
    payload = d.model_dump(mode="json")
    assert payload["kind"] == "decision"
    assert (d.event_id, d.market_id, d.book, d.selection) == ("e1", "e1:h2h", "bookA", "home")
    assert (d.price, d.price_net, d.latency_haircut) == (2.15, 2.15, 0.01)
    assert d.p_consensus == pytest.approx(0.5) and d.n_families == 2
    # the edge can be recomputed from the record
    assert d.edge == pytest.approx(d.p_consensus * d.price_net * (1 - d.latency_haircut) - 1)
    assert d.edge == pytest.approx(0.06425)
    assert 0.0 < d.stake <= cfg.venues["bookA"].max_stake
    assert (d.decided_at, d.commence_utc) == (T0, START)
    assert (d.config_digest, d.snapshot_digest) == (digest(cfg), digest(s))
    assert d.order_id


# start times relative to the tick; verify_after is 5 minutes, max_lead is 12 hours
@pytest.mark.parametrize(
    ("starts_in", "decidable_now"),
    [
        (-MIN, False),  # started a minute ago
        (timedelta(0), False),  # starts this instant
        (5 * MIN, False),  # no time left to verify
        (5 * MIN + timedelta(seconds=1), True),
        (timedelta(hours=12), True),
        (timedelta(hours=12, seconds=1), False),  # too far ahead
    ],
)
def test_only_games_inside_the_decision_window_are_decided(
    starts_in: timedelta, decidable_now: bool
) -> None:
    # decide and decidable agree on every case, so in-play games are never decided on or logged
    schedule, cfg = events("e1", start=T0 + starts_in), config()
    out = decide([snap("e1")], events=schedule, state=state(), config=cfg, as_of=T0).orders
    assert [d.event_id for d in out] == (["e1"] if decidable_now else [])
    assert decidable(snap("e1"), schedule, as_of=T0, config=cfg) is decidable_now


def test_games_with_no_known_start_get_no_orders() -> None:
    schedule, cfg = events("e1"), config()
    snaps = [snap("unlisted"), snap("e1")]
    out = decide(snaps, events=schedule, state=state(), config=cfg, as_of=T0).orders
    assert [d.event_id for d in out] == ["e1"]
    assert not decidable(snap("unlisted"), schedule, as_of=T0, config=cfg)


def test_a_decision_at_or_after_the_start_cannot_be_recorded() -> None:
    assert decision(decided_at=START - timedelta(seconds=1)).decided_at < START
    for late in (START, START + MIN):
        with pytest.raises(ValueError):
            decision(decided_at=late)


def test_a_second_tick_does_not_duplicate_orders() -> None:
    cfg, snaps, schedule = config(), [snap("e1"), snap("e2")], events("e1", "e2")
    first = decide(snaps, events=schedule, state=state(), config=cfg, as_of=T0).orders
    assert {d.event_id for d in first} == {"e1", "e2"}
    after = state(*(Position(decision=d) for d in first))
    assert decide(snaps, events=schedule, state=after, config=cfg, as_of=T0 + MIN).orders == []


def test_a_reference_only_book_is_never_ordered() -> None:
    # ref3's 2.20 beats bookA's 2.15 against the consensus of 0.4848, but reference books are never
    # bet
    venue = {"ref3": {"home": 2.20, "away": 2.20 / 1.20}} | STALE
    cfg = config(ref3=VenueConfig(role="reference"))
    out = decide(
        [snap(venue=venue)], events=events("e1"), state=state(), config=cfg, as_of=T0
    ).orders
    assert out and {d.book for d in out} == {"bookA"}


def test_pending_and_open_orders_reserve_the_risk_budget() -> None:
    cfg = config(risk=RiskConfig(max_total_stake=0.03))
    snaps, schedule = [snap("e1"), snap("e2"), snap("e3")], events("e1", "e2", "e3")

    # one pending and one filled order already use the whole budget
    pending = Position(decision=decision("o1", event="e8", stake=0.015))
    d = decision("o2", event="e9", stake=0.015)
    full = state(pending, Position(decision=d, fill=filled(d)))
    assert decide(snaps, events=schedule, state=full, config=cfg, as_of=T0).orders == []

    # with 0.02 reserved, the new stakes fit in the 0.01 left
    free = decide(snaps, events=schedule, state=state(), config=cfg, as_of=T0).orders
    part = state(Position(decision=decision("o1", event="e8", stake=0.02)))
    squeezed = decide(snaps, events=schedule, state=part, config=cfg, as_of=T0).orders
    assert sum(x.stake for x in free) > 0.01
    assert squeezed and sum(x.stake for x in squeezed) <= 0.01 + 1e-6


def test_a_miss_frees_its_budget_and_its_event_can_be_decided_again() -> None:
    # the miss frees the whole budget, so e1 and e2 can be staked again
    cfg = config(risk=RiskConfig(max_total_stake=0.03))
    d = decision("o1", event="e1", stake=0.03)
    after_miss = state(Position(decision=d, miss=missed(d)))
    later = T0 + 10 * MIN
    snaps = [snap("e1", at=later), snap("e2", at=later)]
    out = decide(snaps, events=events("e1", "e2"), state=after_miss, config=cfg, as_of=later).orders
    assert sorted(x.event_id for x in out) == ["e1", "e2"]
    assert all(x.stake > 0.0 for x in out)


def test_a_retry_is_a_new_order_with_its_own_id() -> None:
    # deciding the same bet again after a miss needs a new order id
    cfg, schedule = config(), events("e1")
    [first] = decide([snap()], events=schedule, state=state(), config=cfg, as_of=T0).orders
    after_miss = state(Position(decision=first, miss=missed(first)))
    later = T0 + 10 * MIN
    [retry] = decide(
        [snap(at=later)], events=schedule, state=after_miss, config=cfg, as_of=later
    ).orders
    assert (retry.event_id, retry.book, retry.selection) == ("e1", first.book, first.selection)
    assert retry.order_id != first.order_id


def test_new_stakes_at_one_book_stay_inside_its_cap() -> None:
    # two games at bookA, with the cap at half the uncapped stake, so the optimiser fills the cap
    snaps, schedule = [snap("e1"), snap("e2")], events("e1", "e2")
    loose = decide(
        snaps,
        events=schedule,
        state=state(),
        config=config(bookA=VenueConfig(max_stake=0.5)),
        as_of=T0,
    ).orders
    cap = float(sum(d.stake for d in loose)) / 2
    tight = decide(
        snaps,
        events=schedule,
        state=state(),
        config=config(bookA=VenueConfig(max_stake=cap)),
        as_of=T0,
    ).orders
    assert float(sum(d.stake for d in tight)) == pytest.approx(cap, abs=1e-5)
    assert float(sum(d.stake for d in tight)) <= cap


def test_a_capped_book_frees_budget_for_another_book() -> None:
    # capping bookA hard moves stake to bookB
    snaps = [snap("e1"), snap("e2", venue={"bookB": STALE["bookA"]})]
    schedule = events("e1", "e2")
    risk = RiskConfig(max_total_stake=0.04)
    free = decide(snaps, events=schedule, state=state(), config=config(risk=risk), as_of=T0).orders
    capped_cfg = config(risk=risk, bookA=VenueConfig(max_stake=0.005))
    capped = decide(snaps, events=schedule, state=state(), config=capped_cfg, as_of=T0).orders
    at = {d.book: d.stake for d in capped}
    assert at["bookA"] <= Decimal("0.005")
    assert at["bookB"] > {d.book: d.stake for d in free}["bookB"]


def test_exposure_already_at_a_book_counts_against_its_cap() -> None:
    cfg = config(bookA=VenueConfig(max_stake=0.015))
    d = decision("o1", event="e9", book="bookA", stake=0.01)
    held = state(Position(decision=d, fill=filled(d)))
    out = decide(
        [snap("e1"), snap("e2")], events=events("e1", "e2"), state=held, config=cfg, as_of=T0
    ).orders
    assert out and sum(x.stake for x in out) == pytest.approx(Decimal("0.005"), abs=Decimal("2e-6"))


def test_digests_ignore_key_order_and_see_every_change() -> None:
    a = snap(venue={"bookA": {"home": 2.15, "away": 1.75}, "bookB": {"away": 1.7, "home": 2.2}})
    b = QuoteSnapshot.model_validate(
        {**a.model_dump(), "prices": dict(reversed(list(a.prices.items())))}
    )
    assert digest(a) == digest(b)
    assert len(digest(a)) == 64 and int(digest(a), 16) >= 0
    assert digest(a) != digest(snap(venue={"bookA": {"home": 2.16, "away": 1.75}}))
    assert digest(config()) != digest(config(bookA=VenueConfig(max_stake=0.04)))


@pytest.mark.parametrize(
    ("decided_at_price", "then", "outcome"),
    [
        (2.20, 2.20, "fill"),
        (2.20, 2.25, "fill"),
        (2.20, 2.05, "fill"),  # 7% worse but still above the bar
        (2.20, 2.01, "no_edge"),
        (2.20, None, "no_quote"),
        (2.04, 2.02, "fill"),  # 0.5 * 2.02 - 1 is exactly ev_min
        (2.04, 2.01, "no_edge"),  # 0.5 * 2.01 - 1 = 0.5% is short of it
    ],
)
def test_fill_verification_needs_the_edge_to_still_clear_ev_min(
    decided_at_price: float, then: float | None, outcome: str
) -> None:
    # fair 50% and ev_min 1%, so the price must stay at or above 2.02, whatever bookB pays
    venue = {"bookB": {"home": 2.40, "away": 1.65}}
    if then is not None:
        venue["bookA"] = {"home": then, "away": 1.75}
    s2 = snap(at=T0 + 5 * MIN, venue=venue, refs=())
    pos = Position(decision=decision(event="e1", price=decided_at_price))
    record = verify(pos, [s2], config=config(), as_of=T0 + 5 * MIN)
    if outcome == "fill":
        assert isinstance(record, Filled)
        assert (record.price, record.price_net, record.observed_at) == (then, then, T0 + 5 * MIN)
    else:
        assert isinstance(record, Missed) and record.reason == outcome
        assert record.price == then and record.observed_at == T0 + 5 * MIN


def test_a_fill_is_judged_net_of_the_books_commission() -> None:
    # 1 + (3 - 1) * 0.95 = 2.90
    d = decision(event="e1", book="exch", price=3.0, price_net=2.90)
    s2 = snap(at=T0 + 5 * MIN, venue={"exch": {"home": 3.0, "away": 1.45}})
    record = verify(Position(decision=d), [s2], config=config(), as_of=T0 + 5 * MIN)
    assert isinstance(record, Filled) and record.price_net == pytest.approx(2.90)

    # at p 0.36, 2.85 clears ev_min gross (+2.6%) but nets 1 + 1.85 * 0.95 = 2.7575, an edge of
    # -0.7%
    d = decision(event="e1", book="exch", price=3.0, price_net=2.90, p=0.36)
    s2 = snap(at=T0 + 5 * MIN, venue={"exch": {"home": 2.85, "away": 1.45}})
    record = verify(Position(decision=d), [s2], config=config(), as_of=T0 + 5 * MIN)
    assert isinstance(record, Missed) and record.reason == "no_edge"


def test_verification_uses_the_first_snapshot_at_least_verify_after_later() -> None:
    def at(minutes: int, home: float, event: str = "e1") -> QuoteSnapshot:
        return snap(event, at=T0 + minutes * MIN, venue={"bookA": {"home": home, "away": 1.75}})

    # +4m is too early, +6m is too late, and the other market is ignored
    snaps = [at(6, 2.10), at(5, 2.19, event="e2"), at(5, 2.19), at(4, 2.30)]
    record = verify(
        Position(decision=decision(event="e1")), snaps, config=config(), as_of=T0 + 7 * MIN
    )
    assert isinstance(record, Filled)
    assert (record.price, record.observed_at) == (2.19, T0 + 5 * MIN)


def test_an_order_waits_for_its_verification_snapshot_without_looking_ahead() -> None:
    pos, cfg = Position(decision=decision(event="e1")), config()
    early = snap(at=T0 + 4 * MIN)
    assert verify(pos, [early], config=cfg, as_of=T0 + 4 * MIN) is None
    # a snapshot captured after as_of is ignored
    future = snap(at=T0 + 5 * MIN, venue={"bookA": {"home": 2.20, "away": 1.75}})
    assert verify(pos, [early, future], config=cfg, as_of=T0 + 4 * MIN) is None


def test_a_game_that_starts_before_verification_is_a_miss() -> None:
    pos = Position(decision=decision(event="e1", start=T0 + 10 * MIN))
    cfg, kickoff = config(), T0 + 10 * MIN
    assert verify(pos, [], config=cfg, as_of=kickoff - timedelta(seconds=1)) is None
    # an in-play snapshot is not a verification
    in_play = snap(at=kickoff + MIN, venue={"bookA": {"home": 2.20, "away": 1.75}})
    for as_of in (kickoff, kickoff + 2 * MIN):
        record = verify(pos, [in_play], config=cfg, as_of=as_of)
        assert isinstance(record, Missed)
        assert (record.reason, record.observed_at) == ("started", as_of)


def test_the_verification_snapshot_is_never_used_to_decide_again() -> None:
    cfg, schedule = config(), events("e1")
    [d] = decide([snap()], events=schedule, state=state(), config=cfg, as_of=T0).orders

    # snapshot 2: bookA still at 2.15, and bookB now offers an even bigger edge
    s2 = snap(
        at=T0 + 5 * MIN,
        venue={"bookA": {"home": 2.15, "away": 1.75}, "bookB": {"home": 2.25, "away": 1.7}},
    )
    record = verify(Position(decision=d), [s2], config=cfg, as_of=T0 + 5 * MIN)
    assert isinstance(record, Filled)
    after = state(Position(decision=d, fill=record))
    assert decide([s2], events=schedule, state=after, config=cfg, as_of=T0 + 5 * MIN).orders == []


def test_a_verification_long_after_it_was_due_is_a_late_miss() -> None:
    # the check is due at T+5m and allowed until T+35m, so a fill at T+40m is refused
    pos, cfg = Position(decision=decision(event="e1")), config()
    late = snap(at=T0 + 40 * MIN, venue={"bookA": {"home": 2.20, "away": 1.75}})
    record = verify(pos, [late], config=cfg, as_of=T0 + 40 * MIN)
    assert isinstance(record, Missed)
    assert (record.reason, record.observed_at) == ("late", T0 + 40 * MIN)
    # a snapshot at the window's last moment still counts
    on_time = snap(at=T0 + 35 * MIN, venue={"bookA": {"home": 2.20, "away": 1.75}})
    assert isinstance(verify(pos, [on_time, late], config=cfg, as_of=T0 + 40 * MIN), Filled)


def closing(*, decided_at: datetime = START - 60 * MIN) -> Position:
    # filled at net 2.0 with a decision-time consensus of 0.52
    d = decision(event="e1", price=2.0, p=0.52, decided_at=decided_at)
    return Position(decision=d, fill=filled(d))


def test_close_clv_worked_example() -> None:
    # the latest snapshot inside [T-15m, T-2m] is the close, moving p from 0.52 to 0.55
    snaps = [
        snap(at=START - 20 * MIN, p=0.51),  # before the window
        snap(at=START - 14 * MIN, p=0.53),
        snap(at=START - 3 * MIN, p=0.55),
        snap(at=START - 1 * MIN, p=0.60),  # after the window
    ]
    record = close_payload(closing(), snaps, config=config())
    assert record.status == "ok" and record.observed_at == START - 3 * MIN
    assert record.p_close == pytest.approx(0.55)
    assert record.clv_econ == pytest.approx(0.10)  # 2.0 * 0.55 - 1
    assert record.clv_prob == pytest.approx(0.0577, abs=1e-4)  # 0.55 / 0.52 - 1


def test_economic_clv_uses_the_net_price_taken() -> None:
    # gross 2.0 on an exchange charging 5% is net 1.95: 1.95 * 0.55 - 1 = 0.0725
    d = decision(
        event="e1", book="exch", price=2.0, price_net=1.95, p=0.52, decided_at=START - 60 * MIN
    )
    pos = Position(decision=d, fill=filled(d))
    record = close_payload(pos, [snap(at=START - 3 * MIN, p=0.55)], config=config())
    assert record.clv_econ == pytest.approx(0.0725)


@pytest.mark.parametrize(
    "snaps",
    [
        [],  # absent
        [snap(at=START - 30 * MIN, p=0.55)],  # only before the window
        [snap(at=START - 1 * MIN, p=0.55)],  # late: after the window shuts
        [snap(at=START - 5 * MIN, p=0.55, refs=("ref1",))],  # one family: no consensus
    ],
    ids=["absent", "early", "late", "no-consensus"],
)
def test_a_close_without_a_usable_snapshot_is_missing(snaps: list[QuoteSnapshot]) -> None:
    record = close_payload(closing(), snaps, config=config())
    assert record.status == "missing"
    assert record.clv_econ is None and record.clv_prob is None


def test_the_close_skips_a_later_snapshot_with_no_panel() -> None:
    # a late verification poll holds only the bet book, so the close is the snapshot before it
    panel = snap(at=START - 10 * MIN, p=0.55)
    bet_book_only = snap(at=START - 5 * MIN, refs=())
    record = close_payload(closing(), [panel, bet_book_only], config=config())
    assert record.status == "ok" and record.observed_at == START - 10 * MIN
    assert record.p_close == pytest.approx(0.55)


def test_a_close_too_soon_after_the_decision_is_degenerate() -> None:
    # a 9 minute gap is too short, exactly 10 is fine
    pos = closing(decided_at=START - 14 * MIN)
    soon = close_payload(pos, [snap(at=START - 5 * MIN, p=0.55)], config=config())
    assert soon.status == "degenerate" and soon.clv_econ is None and soon.clv_prob is None
    enough = close_payload(pos, [snap(at=START - 4 * MIN, p=0.55)], config=config())
    assert enough.status == "ok"


def settling(selection: str = "home") -> Position:
    # stake 0.02 filled at 1.95 on an exchange charging 5%: net 1.9025
    d = decision(
        event="e1", book="exch", selection=selection, price=1.95, price_net=1.9025, stake=0.02
    )
    return Position(decision=d, fill=filled(d))


def test_settle_worked_example() -> None:
    record = settle_payload(settling(), Result(event_id="e1", winner="home"))
    assert record.won and record.winner == "home"
    assert record.pnl == Decimal("0.01805")  # 0.02 * (1.9025 - 1)


@pytest.mark.parametrize("winner", ["away", "draw"])
def test_any_other_result_loses_the_stake(winner: str) -> None:
    # a draw on a home pick is a loss, not a void
    record = settle_payload(settling(), Result(event_id="e1", winner=winner))
    assert not record.won and record.pnl == Decimal("-0.02")


def test_lifecycle_misuse_raises() -> None:
    d = decision(event="e1")
    pending, done, gone = (
        Position(decision=d),
        Position(decision=d, fill=filled(d)),
        Position(decision=d, miss=missed(d)),
    )
    cfg, result = config(), Result(event_id="e1", winner="home")
    with pytest.raises(ValueError):
        verify(done, [], config=cfg, as_of=T0 + 5 * MIN)
    with pytest.raises(ValueError):
        verify(gone, [], config=cfg, as_of=T0 + 5 * MIN)
    with pytest.raises(ValueError):
        close_payload(gone, [], config=cfg)
    with pytest.raises(ValueError):
        settle_payload(pending, result)
    with pytest.raises(ValueError):
        settle_payload(done, Result(event_id="e2", winner="home"))


def history() -> list[dict[str, object]]:
    # A filled, closed ok and won; B missed; C pending; D filled, close missing, unsettled
    a, b = decision("A", event="eA", stake=0.02), decision("B", event="eB", stake=0.01)
    c, d = decision("C", event="eC", stake=0.03), decision("D", event="eD", stake=0.04)
    close_a = Closed(
        order_id="A",
        status="ok",
        observed_at=START - 3 * MIN,
        p_close=0.55,
        clv_econ=0.1,
        clv_prob=0.05,
    )
    won_a = Settled(order_id="A", winner="home", won=True, pnl=0.024)  # 0.02 * (2.20 - 1)
    no_close_d = Closed(order_id="D", status="missing")
    return payloads(a, b, c, d, filled(a), missed(b), filled(d), close_a, no_close_d, won_a)


def test_fold_rebuilds_positions_bankroll_and_exposure() -> None:
    s = fold(history(), bankroll=1.0)
    assert list(s.positions) == ["A", "B", "C", "D"]
    assert s.bankroll == Decimal("1.024")
    assert s.reserved == Decimal("0.03")  # C
    assert s.at_risk == Decimal("0.04")  # D
    a = s.positions["A"]
    assert a.fill is not None and a.close is not None and a.settle is not None and a.miss is None
    assert s.positions["B"].miss is not None and s.positions["C"].fill is None


def test_summarise_reports_the_fill_rate_pnl_and_mean_clv() -> None:
    summary = summarise(fold(history(), bankroll=1.0))
    assert (summary.decisions, summary.pending, summary.filled, summary.missed) == (4, 1, 2, 1)
    assert summary.fill_rate == pytest.approx(2 / 3)
    assert summary.settled == 1 and summary.pnl == Decimal("0.024")
    assert summary.closes == {"ok": 1, "missing": 1, "degenerate": 0}
    assert summary.clv_econ == pytest.approx(0.1) and summary.clv_prob == pytest.approx(0.05)


def test_the_shortfall_is_the_mean_fill_price_against_the_decision_price() -> None:
    # A filled at its decision price, B 5% worse (2.09 against 2.20): mean -2.5%
    a, b = decision("A", event="eA"), decision("B", event="eB")
    worse = Filled(order_id="B", observed_at=T0 + 5 * MIN, price=2.09, price_net=2.09)
    summary = summarise(fold(payloads(a, b, filled(a), worse), bankroll=1.0))
    assert summary.shortfall == pytest.approx(-0.025)


def test_nothing_resolved_means_no_fill_rate_and_no_clv() -> None:
    summary = summarise(fold(payloads(decision("A"), decision("B", event="e8")), bankroll=1.0))
    assert (summary.decisions, summary.pending, summary.fill_rate) == (2, 2, None)
    assert summary.shortfall is None
    assert summary.closes == {"ok": 0, "missing": 0, "degenerate": 0}
    assert summary.clv_econ is None and summary.clv_prob is None and summary.pnl == 0.0


@pytest.mark.parametrize(
    "case",
    ["unknown order", "decided twice", "resolved twice", "settled unfilled", "closed missed"],
)
def test_fold_rejects_an_impossible_history(case: str) -> None:
    a = decision("A")
    win = Settled(order_id="A", winner="home", won=True, pnl=0.024)
    records = {
        "unknown order": [filled(a)],
        "decided twice": [a, a],
        "resolved twice": [a, missed(a), filled(a)],
        "settled unfilled": [a, win],
        "closed missed": [a, missed(a), Closed(order_id="A", status="missing")],
    }[case]
    with pytest.raises(ValueError):
        fold(payloads(*records), bankroll=1.0)


def full_cycle(path: Path) -> Ledger:
    # decides two events, fills one, misses the other, then closes and settles the fill
    cfg, ledger, schedule = config(), Ledger(path), events("e1", "e2")

    def now() -> TraderState:
        return fold((e["payload"] for e in ledger.entries()), bankroll=cfg.paper.bankroll)

    def write(*records: Decided | Filled | Missed | Closed | Settled | None) -> None:
        for r in records:
            if r is not None:
                ledger.append(r.model_dump(mode="json"))

    # snapshot 2 comes five minutes later, with e2's book having pulled its price
    t1 = START - 60 * MIN
    write(
        *decide(
            [snap("e1", at=t1), snap("e2", at=t1)],
            events=schedule,
            state=now(),
            config=cfg,
            as_of=t1,
        ).orders
    )
    s2 = [snap("e1", at=t1 + 5 * MIN), snap("e2", at=t1 + 5 * MIN, venue={})]
    write(*(verify(p, s2, config=cfg, as_of=t1 + 5 * MIN) for p in now().positions.values()))

    # the closing line, then the result
    close = [snap("e1", at=START - 3 * MIN, p=0.55)]
    fills = [p for p in now().positions.values() if p.fill is not None]
    write(*(close_payload(p, close, config=cfg) for p in fills))
    write(*(settle_payload(p, Result(event_id="e1", winner="home")) for p in fills))
    return ledger


def test_a_full_cycle_ledger_verifies_and_summarises(tmp_path: Path) -> None:
    ledger = full_cycle(tmp_path / "paper.jsonl")
    assert ledger.verify() == 6  # 2 decisions, 1 fill, 1 miss, 1 close, 1 settle
    kinds = [e["payload"]["kind"] for e in ledger.entries()]
    assert kinds[:2] == ["decision", "decision"] and sorted(kinds[2:4]) == ["fill", "miss"]
    assert kinds[4:] == ["close", "settle"]

    # the state rebuilt from the file alone
    state_ = fold((e["payload"] for e in ledger.entries()), bankroll=1.0)
    summary = summarise(state_)
    assert (summary.decisions, summary.filled, summary.missed, summary.pending) == (2, 1, 1, 0)
    assert summary.fill_rate == pytest.approx(0.5)
    won = next(p for p in state_.positions.values() if p.settle is not None)
    assert summary.pnl == (won.decision.stake * Decimal("1.15")).quantize(STAKE_STEP)  # at 2.15
    assert summary.closes["ok"] == 1 and summary.clv_econ == pytest.approx(2.15 * 0.55 - 1)
    assert state_.bankroll == 1 + summary.pnl
    assert state_.reserved == 0.0 and state_.at_risk == 0.0


def test_a_replay_writes_the_same_ledger_twice(tmp_path: Path) -> None:
    # same snapshots in, same order ids and digests out
    first, second = full_cycle(tmp_path / "a.jsonl"), full_cycle(tmp_path / "b.jsonl")
    assert first.head_hash() == second.head_hash()


ORDER = st.tuples(
    st.floats(0.001, 0.05),  # stake
    st.floats(1.2, 5.0),  # gross price
    st.sampled_from([0.0, 0.02, 0.05]),  # commission
    st.sampled_from(["pending", "missed", "open", "won", "lost"]),  # where it ends up
)


@settings(max_examples=100, deadline=None)
@given(orders=st.lists(ORDER, min_size=1, max_size=6), data=st.data())
def test_money_is_neither_created_nor_destroyed(
    orders: list[tuple[float, float, float, str]], data: st.DataObject
) -> None:
    # each order's records in life order, with the money they move
    lives: list[list[tuple[Decided | Filled | Missed | Settled, Decimal, Decimal, Decimal]]] = []
    zero = Decimal(0)
    for i, (raw, price, commission, end) in enumerate(orders):
        stake = Decimal(raw).quantize(STAKE_STEP)
        net = net_price(price, commission)
        d = decision(f"o{i}", event=f"e{i}", price=price, price_net=net, stake=stake)
        life = [(d, stake, zero, zero)]  # (record, change in reserved, in at_risk, in bankroll)
        if end == "missed":
            life.append((missed(d), -stake, zero, zero))
        elif end != "pending":
            f = filled(d)
            life.append((f, -stake, stake, zero))
            if end in ("won", "lost"):
                pos = Position(decision=d, fill=f)
                settle = settle_payload(
                    pos, Result(event_id=f"e{i}", winner="home" if end == "won" else "away")
                )
                won = (stake * (Decimal(str(net)) - 1)).quantize(STAKE_STEP)
                gain = won if end == "won" else -stake
                life.append((settle, zero, -stake, gain))
        lives.append(life)

    # interleaves the orders while keeping each one's own sequence
    records: list[Decided | Filled | Missed | Settled] = []
    reserved = at_risk = bankroll = zero
    while any(lives):
        life = data.draw(st.sampled_from([life for life in lives if life]))
        record, dr, da, db = life.pop(0)
        records.append(record)
        reserved, at_risk, bankroll = reserved + dr, at_risk + da, bankroll + db

        # after every record the rebuilt state matches the books kept here
        s = fold(payloads(*records), bankroll=1.0)
        assert s.reserved == reserved
        assert s.at_risk == at_risk
        assert s.bankroll == 1 + bankroll
