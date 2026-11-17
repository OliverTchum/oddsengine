"""Tick tests: scripts/tick.py run end to end on a temporary folder with the Odds API faked."""

import fcntl
import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from oddsengine.data.odds_api import OddsApiError
from oddsengine.execution.ledger import Ledger
from oddsengine.types import EventInfo

NOW = datetime(2026, 11, 8, 23, 0, tzinfo=UTC)
START = NOW + timedelta(hours=2)  # inside the 3 hour horizon, so a full poll is due
SPORTS = ["basketball_nba", "soccer_epl"]

CONFIG = """
consensus: {devig_method: multiplicative, latency_haircut: 0.0}
decision: {ev_min: 0.01}
archiver: {root: data/odds, regions: [uk, eu], markets: [h2h]}
live:
  sports: [basketball_nba, soccer_epl]
  ledger: records/ledger.jsonl
  flags: data/flags.jsonl
  results: data/results
betfair: {root: data/betfair}
venues:
  ref1: {role: reference}
  ref2: {role: reference}
  bookA: {role: venue}
"""


def game(sport: str) -> EventInfo:
    return EventInfo(
        event_id=f"{sport}-1", sport=sport, commence_utc=START, home_team="Home", away_team="Away"
    )


def book(key: str, home: float, away: float) -> dict:
    # one bookmaker's h2h prices, quoted at the tick time
    stamp = NOW.isoformat()
    outcomes = [{"name": "Home", "price": home}, {"name": "Away", "price": away}]
    return {
        "key": key,
        "last_update": stamp,
        "markets": [{"key": "h2h", "last_update": stamp, "outcomes": outcomes}],
    }


def payload(sport: str) -> list[dict]:
    # the references say 50/50 but bookA still pays 2.15 on home
    return [
        {
            "id": f"{sport}-1",
            "sport_key": sport,
            "commence_time": START.isoformat(),
            "home_team": "Home",
            "away_team": "Away",
            "bookmakers": [
                book("ref1", 2.0, 2.0),
                book("ref2", 2.0, 2.0),
                book("bookA", 2.15, 1.75),
            ],
        }
    ]


@pytest.fixture
def tick(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # loads scripts/tick.py as a module with every path in tmp_path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("tick", root / "scripts" / "tick.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "tick", module)
    spec.loader.exec_module(module)

    (tmp_path / "config.yaml").write_text(CONFIG)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "LOCK", tmp_path / "tick.lock")
    monkeypatch.setenv("ODDS_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)

    # fake Odds API that records the sports asked for, and no Betfair
    module.odds_calls = []
    monkeypatch.setattr(module, "fetch_events", lambda *, api_key, sport: ([game(sport)], 500))

    def fake_odds(*, api_key, sport, **_):
        module.odds_calls.append(sport)
        return payload(sport), 498

    monkeypatch.setattr(module, "fetch_odds", fake_odds)
    monkeypatch.setattr(module, "fetch_scores", lambda **_: ([], 500))
    monkeypatch.setattr(module, "betfair_session", lambda run: None)
    return module


def run(tick, tmp_path: Path, *extra: str) -> int:
    return tick.main(["--config", str(tmp_path / "config.yaml"), "--now", NOW.isoformat(), *extra])


def test_dry_run_polls_nothing_and_writes_nothing(tick, tmp_path):
    # a dry run makes no paid call and writes no ledger or archive
    assert run(tick, tmp_path, "--dry-run") == 0
    assert tick.odds_calls == []
    assert not (tmp_path / "records" / "ledger.jsonl").exists()
    assert not (tmp_path / "data" / "odds").exists()


def test_live_tick_records_a_decision_on_the_stale_book(tick, tmp_path):
    # bookA's 2.15 against a fair 50% clears ev_min, so it is ordered
    assert run(tick, tmp_path) == 0
    records = [e["payload"] for e in Ledger(tmp_path / "records" / "ledger.jsonl").entries()]
    decisions = [r for r in records if r["kind"] == "decision"]
    assert decisions
    assert {d["book"] for d in decisions} == {"bookA"}


def test_held_lock_skips_the_tick(tick, tmp_path, monkeypatch):
    # another process holds the lock, so the tick returns 0 early
    def no_events(**_):
        raise AssertionError("fetch_events must not run")

    monkeypatch.setattr(tick, "fetch_events", no_events)
    with (tmp_path / "tick.lock").open("w") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(tick, tmp_path) == 0
    assert tick.odds_calls == []


def test_one_sport_failing_does_not_stop_the_other(tick, tmp_path, monkeypatch):
    # the NBA poll fails but the EPL is still polled and decided
    def odds(*, api_key, sport, **_):
        tick.odds_calls.append(sport)
        if sport == "basketball_nba":
            raise OddsApiError("boom")
        return payload(sport), 498

    monkeypatch.setattr(tick, "fetch_odds", odds)
    assert run(tick, tmp_path) == 0
    assert tick.odds_calls == SPORTS
    records = [e["payload"] for e in Ledger(tmp_path / "records" / "ledger.jsonl").entries()]
    assert [r["event_id"] for r in records if r["kind"] == "decision"] == ["soccer_epl-1"]
