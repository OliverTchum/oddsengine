"""Schedule tests: when each kind of poll is due, on hand-made clocks."""

from datetime import UTC, datetime, timedelta

import pytest

from oddsengine.config import AppConfig, BetfairConfig, LiveConfig
from oddsengine.data.schedule import (
    capture_window,
    close_poll_due,
    close_ready,
    full_poll_due,
    moved,
    results_call_due,
    settle_due,
    trigger_due,
    trigger_games,
    verify_due,
)
from oddsengine.execution.trader import Closed, Decided, Filled, Missed, Position, Settled
from oddsengine.types import EventInfo

NOW = datetime(2026, 11, 8, 20, 0, tzinfo=UTC)
MIN = timedelta(minutes=1)
CFG = AppConfig()  # verify_after 5m, close window [T-15m, T-2m], horizon 6h, decide_every 1h
# betfair: move 0.02, max_triggers 3 (the last for games within 2h)


def game(start: datetime) -> EventInfo:
    return EventInfo(event_id="e1", sport="basketball_nba", commence_utc=start)


def position(*, start: datetime, decided_at: datetime = NOW - 60 * MIN, stage: str) -> Position:
    # one order at a given stage of its life
    d = Decided(
        order_id="o1",
        event_id="e1",
        market_id="e1:h2h",
        selection="home",
        book="bookA",
        price=2.0,
        price_net=2.0,
        p_consensus=0.52,
        n_families=2,
        edge=0.03,
        latency_haircut=0.01,
        stake=0.01,
        decided_at=decided_at,
        commence_utc=start,
        config_digest="c" * 64,
        snapshot_digest="s" * 64,
    )
    if stage == "pending":
        return Position(decision=d)
    if stage == "missed":
        return Position(decision=d, miss=Missed(order_id="o1", observed_at=NOW, reason="no_quote"))
    fill = Filled(order_id="o1", observed_at=decided_at + 5 * MIN, price=2.0, price_net=2.0)
    if stage == "filled":
        return Position(decision=d, fill=fill)
    close = Closed(order_id="o1", status="missing")
    if stage == "closed":
        return Position(decision=d, fill=fill, close=close)
    settle = Settled(order_id="o1", winner="home", won=True, pnl=0.01)
    return Position(decision=d, fill=fill, close=close, settle=settle)


def test_a_full_poll_is_due_for_a_decidable_game_within_the_horizon() -> None:
    assert full_poll_due([game(NOW + 3 * 60 * MIN)], last=None, now=NOW, config=CFG)


def test_no_full_poll_for_games_beyond_the_horizon_or_too_close_to_decide() -> None:
    far, close = game(NOW + 7 * 60 * MIN), game(NOW + 5 * MIN)
    assert not full_poll_due([far, close], last=None, now=NOW, config=CFG)
    assert not full_poll_due([], last=None, now=NOW, config=CFG)


def test_the_exchange_is_captured_further_out_than_full_polls() -> None:
    # full polls within 3h, Betfair capture (and so triggers) within 6h
    cfg = AppConfig(
        live=LiveConfig(horizon=timedelta(hours=3)),
        betfair=BetfairConfig(horizon=timedelta(hours=6)),
    )
    five_out = game(NOW + 5 * 60 * MIN)
    assert not full_poll_due([five_out], last=None, now=NOW, config=cfg)
    assert capture_window(NOW, config=cfg) == (NOW + 5 * MIN, NOW + 6 * 60 * MIN)


def test_a_full_poll_waits_decide_every_after_the_last_one() -> None:
    games = [game(NOW + 3 * 60 * MIN)]
    assert not full_poll_due(games, last=NOW - 59 * MIN, now=NOW, config=CFG)
    assert full_poll_due(games, last=NOW - 60 * MIN, now=NOW, config=CFG)


def test_a_pending_order_is_verified_once_verify_after_has_passed() -> None:
    start = NOW + 60 * MIN
    assert verify_due(
        position(start=start, decided_at=NOW - 5 * MIN, stage="pending"), now=NOW, config=CFG
    )
    assert not verify_due(
        position(start=start, decided_at=NOW - 4 * MIN, stage="pending"), now=NOW, config=CFG
    )
    assert not verify_due(position(start=start, stage="filled"), now=NOW, config=CFG)
    assert not verify_due(position(start=start, stage="missed"), now=NOW, config=CFG)


def test_every_game_gets_one_close_poll_late_in_its_window() -> None:
    # close window is [T-15m, T-2m] but polling runs [T-12m, T-2m]
    g = game(NOW + 10 * MIN)
    assert close_poll_due(g, last_close=None, now=NOW, config=CFG)  # T-10m
    assert not close_poll_due(g, last_close=NOW - MIN, now=NOW, config=CFG)  # polled at T-11m
    assert close_poll_due(g, last_close=NOW - 3 * MIN, now=NOW, config=CFG)  # last was T-13m
    early, shut = game(NOW + 13 * MIN), game(NOW + MIN)
    assert not close_poll_due(early, last_close=None, now=NOW, config=CFG)  # T-13m: too early
    assert not close_poll_due(shut, last_close=None, now=NOW, config=CFG)  # T-1m: window shut


def test_a_close_is_recorded_once_the_window_shuts() -> None:
    assert not close_ready(position(start=NOW + 2 * MIN, stage="filled"), now=NOW, config=CFG)
    assert close_ready(
        position(start=NOW + 2 * MIN - timedelta(seconds=1), stage="filled"), now=NOW, config=CFG
    )
    assert not close_ready(
        position(start=NOW - 60 * MIN, decided_at=NOW - 120 * MIN, stage="closed"),
        now=NOW,
        config=CFG,
    )


def test_settlement_is_due_settle_after_the_start() -> None:
    assert settle_due(
        position(start=NOW - 4 * 60 * MIN, decided_at=NOW - 5 * 60 * MIN, stage="closed"),
        now=NOW,
        config=CFG,
    )
    assert not settle_due(
        position(start=NOW - 3 * 60 * MIN, decided_at=NOW - 4 * 60 * MIN, stage="closed"),
        now=NOW,
        config=CFG,
    )
    assert not settle_due(
        position(start=NOW - 5 * 60 * MIN, decided_at=NOW - 6 * 60 * MIN, stage="settled"),
        now=NOW,
        config=CFG,
    )


def test_results_are_asked_for_at_most_once_per_settle_every() -> None:
    assert results_call_due(last=None, now=NOW, config=CFG)
    assert not results_call_due(last=NOW - 5 * 60 * MIN, now=NOW, config=CFG)
    assert results_call_due(last=NOW - 6 * 60 * MIN, now=NOW, config=CFG)


def test_a_game_is_triggered_when_a_runner_moves_at_least_the_threshold() -> None:
    before = {"e1": {"A": 0.50, "B": 0.50}, "e2": {"C": 0.40, "D": 0.60}}
    after = {"e1": {"A": 0.53, "B": 0.47}, "e2": {"C": 0.41, "D": 0.59}}
    assert moved(before, after, config=CFG) == {"e1": pytest.approx(0.03)}


def test_a_runner_or_game_seen_in_only_one_capture_is_not_a_move() -> None:
    before = {"e1": {"A": 0.50}}
    after = {"e1": {"B": 0.30}, "e2": {"C": 0.90}}
    assert moved(before, after, config=CFG) == {}
    assert moved({}, after, config=CFG) == {}


def test_a_trigger_poll_is_due_only_before_the_closing_window_opens() -> None:
    # at T-15m the window is open, so a poll there would become the close
    since = NOW - 5 * MIN
    assert trigger_due(game(NOW + 16 * MIN), last_seen=None, since=since, now=NOW, config=CFG)
    assert not trigger_due(game(NOW + 15 * MIN), last_seen=None, since=since, now=NOW, config=CFG)
    assert not trigger_due(game(NOW - MIN), last_seen=None, since=since, now=NOW, config=CFG)


def test_a_trigger_is_held_only_by_books_captured_after_the_move_began() -> None:
    # the move is measured from the exchange capture 5 minutes back, so only books captured after it
    # can show it
    g, since = game(NOW + 60 * MIN), NOW - 5 * MIN
    assert not trigger_due(g, last_seen=NOW - 2 * MIN, since=since, now=NOW, config=CFG)
    assert trigger_due(g, last_seen=since, since=since, now=NOW, config=CFG)
    assert trigger_due(g, last_seen=NOW - 10 * MIN, since=since, now=NOW, config=CFG)


def test_the_last_trigger_of_the_day_is_kept_for_games_near_their_start() -> None:
    # two open triggers go to any game; the third only to a game within 2h
    near, far = game(NOW + 120 * MIN), game(NOW + 121 * MIN)
    for used in (0, 1):
        assert trigger_games([near, far], used=used, now=NOW, config=CFG) == [near, far]
    assert trigger_games([near, far], used=2, now=NOW, config=CFG) == [near]
    assert trigger_games([far], used=2, now=NOW, config=CFG) == []
    assert trigger_games([near], used=3, now=NOW, config=CFG) == []
