"""Snapshot tests: archive rows in, one snapshot per event and capture out."""

from datetime import UTC, datetime, timedelta

import pandas as pd

from oddsengine.data.snapshots import snapshots, without_draws
from oddsengine.types import QuoteSnapshot

BASE = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def archive(n_captures: int = 3) -> pd.DataFrame:
    # two books quoting one h2h market, captured every four hours
    rows = []
    for i in range(n_captures):
        for book, home in (("bookA", 2.00), ("bookB", 1.95)):
            for selection, price in (("home", home), ("away", home + 0.05)):
                rows.append(
                    {
                        "captured_at_utc": BASE + timedelta(hours=4 * i),
                        "event_id": "e1",
                        "bookmaker_key": book,
                        "market_key": "h2h",
                        "outcome_name": selection,
                        "price_decimal": price,
                    }
                )
    return pd.DataFrame(rows)


def test_one_snapshot_per_capture_in_time_order() -> None:
    times = [s.observed_at_utc for s in snapshots(archive())]
    assert times == [BASE, BASE + timedelta(hours=4), BASE + timedelta(hours=8)]


def test_snapshots_of_several_games_come_out_in_capture_order() -> None:
    # ordered by time first, so a later capture never comes before an earlier one
    frame = pd.concat([archive(2).assign(event_id=e) for e in ("zzz", "aaa")], ignore_index=True)
    order = [(s.observed_at_utc, s.event_id) for s in snapshots(frame)]
    later = BASE + timedelta(hours=4)
    assert order == [(BASE, "aaa"), (BASE, "zzz"), (later, "aaa"), (later, "zzz")]


def test_replay_is_deterministic() -> None:
    frame = archive()
    assert list(snapshots(frame)) == list(snapshots(frame))


def test_a_snapshot_carries_every_book() -> None:
    first = next(snapshots(archive(1)))
    assert first.market_id == "e1:h2h"
    assert first.prices == {
        "bookA": {"home": 2.00, "away": 2.05},
        "bookB": {"home": 1.95, "away": 2.00},
    }


def test_other_markets_are_ignored() -> None:
    frame = archive(1)
    frame["market_key"] = "totals"
    assert list(snapshots(frame)) == []


def test_replay_carries_each_books_quote_time() -> None:
    # observed_at is our capture time, quoted_at is the book's
    frame = archive(1)
    stale = BASE - timedelta(minutes=30)
    frame["bookmaker_last_update_utc"] = [
        stale if b == "bookB" else BASE for b in frame["bookmaker_key"]
    ]
    first = next(snapshots(frame))
    assert first.observed_at_utc == BASE
    assert first.quoted_at_utc == {"bookA": BASE, "bookB": stale}


def test_missing_quote_times_stay_missing() -> None:
    # left empty, not filled with the capture time
    assert next(snapshots(archive(1))).quoted_at_utc == {}


def test_quote_time_prefers_the_market_time() -> None:
    # the market's own time wins over the book-level one
    frame = archive(1)
    frame["bookmaker_last_update_utc"] = BASE - timedelta(hours=1)
    frame["market_last_update_utc"] = BASE - timedelta(minutes=5)
    assert next(snapshots(frame)).quoted_at_utc["bookA"] == BASE - timedelta(minutes=5)


def test_old_archive_without_the_market_column_uses_the_book_time() -> None:
    frame = archive(1)
    frame["bookmaker_last_update_utc"] = BASE - timedelta(hours=1)
    assert next(snapshots(frame)).quoted_at_utc["bookA"] == BASE - timedelta(hours=1)


def test_null_market_time_falls_back_to_the_book_time() -> None:
    # bookA has a market time, bookB does not
    frame = archive(1)
    frame["bookmaker_last_update_utc"] = BASE - timedelta(hours=1)
    frame["market_last_update_utc"] = [
        BASE if b == "bookA" else None for b in frame["bookmaker_key"]
    ]
    quoted = next(snapshots(frame)).quoted_at_utc
    assert quoted == {"bookA": BASE, "bookB": BASE - timedelta(hours=1)}


def lines(market: str, quotes: list[tuple[str, float, float]]) -> pd.DataFrame:
    # one book quoting several lines, as (name, point, price)
    return pd.DataFrame(
        {
            "captured_at_utc": BASE,
            "event_id": "e1",
            "home_team": "Home",
            "bookmaker_key": "bookA",
            "market_key": market,
            "outcome_name": name,
            "outcome_point": point,
            "price_decimal": price,
        }
        for name, point, price in quotes
    )


def test_each_totals_line_is_its_own_market() -> None:
    # Over 220.5 and Over 221.5 are different bets
    frame = lines(
        "totals",
        [("Over", 220.5, 1.9), ("Under", 220.5, 1.9), ("Over", 221.5, 2.0), ("Under", 221.5, 1.8)],
    )
    out = {s.market_id: s.prices["bookA"] for s in snapshots(frame, market_key="totals")}
    assert out == {
        "e1:totals:220.5": {"Over": 1.9, "Under": 1.9},
        "e1:totals:221.5": {"Over": 2.0, "Under": 1.8},
    }


def test_a_spreads_line_pairs_the_home_handicap_with_the_opposite_away_one() -> None:
    # Home -3.5 and Away +3.5 are one market
    frame = lines(
        "spreads",
        [("Home", -3.5, 1.9), ("Away", 3.5, 1.9), ("Home", -4.5, 2.1), ("Away", 4.5, 1.7)],
    )
    out = {s.market_id: s.prices["bookA"] for s in snapshots(frame, market_key="spreads")}
    assert out == {
        "e1:spreads:-3.5": {"Home": 1.9, "Away": 1.9},
        "e1:spreads:-4.5": {"Home": 2.1, "Away": 1.7},
    }


def test_without_draws_keeps_only_the_two_way_books() -> None:
    # pinnacle quotes 2-way including overtime, unibet 3-way for regulation
    at = datetime(2026, 11, 8, 23, 0, tzinfo=UTC)
    s = QuoteSnapshot(
        market_id="e1:h2h",
        event_id="e1",
        observed_at_utc=at,
        prices={"pinnacle": {"A": 1.9, "B": 2.0}, "unibet_uk": {"A": 2.3, "B": 2.5, "Draw": 4.2}},
        quoted_at_utc={"pinnacle": at, "unibet_uk": at},
    )
    out = without_draws(s)
    assert set(out.prices) == set(out.quoted_at_utc) == {"pinnacle"}
    assert out.prices["pinnacle"] == {"A": 1.9, "B": 2.0}
