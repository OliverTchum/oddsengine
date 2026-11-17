"""Archiver tests, with no network, using a trimmed real response from The Odds API."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from oddsengine.data.archive import last_capture, partition_path, write_snapshot
from oddsengine.data.odds_api import (
    QUOTE_COLUMNS,
    OddsApiError,
    credits_per_cycle,
    fetch_events,
    fetch_odds,
    normalise_odds_payload,
    results_from_scores,
)

# an invented Odds API response with the same shape as a real one
FIXTURE = Path(__file__).parent / "fixtures" / "odds_api_totals.json"
CAPTURED_AT = datetime(2026, 8, 28, 15, 0, tzinfo=UTC)


def payload() -> list[dict[str, object]]:
    return json.loads(FIXTURE.read_text())  # type: ignore[no-any-return]


@pytest.fixture
def rows() -> list[dict[str, object]]:
    # the fixture response, normalised
    return normalise_odds_payload(payload(), captured_at=CAPTURED_AT)


def test_normalise_flattens_every_outcome_to_its_own_row(rows: list[dict[str, object]]) -> None:
    expected = sum(
        len(m["outcomes"])
        for event in payload()
        for bk in event["bookmakers"]  # type: ignore[attr-defined]
        for m in bk["markets"]
    )
    assert len(rows) == expected
    assert expected > 0


def test_normalised_rows_have_the_canonical_columns(rows: list[dict[str, object]]) -> None:
    assert set(rows[0]) == set(QUOTE_COLUMNS)


def test_capture_time_is_when_we_saw_the_quote_not_the_bookmakers(
    rows: list[dict[str, object]],
) -> None:
    # captured_at is when we saw the quote
    assert all(r["captured_at_utc"] == CAPTURED_AT for r in rows)
    assert any(r["bookmaker_last_update_utc"] != CAPTURED_AT for r in rows)


def test_market_and_bookmaker_update_times_are_kept_apart() -> None:
    # the market's own last_update has its own column
    book = {"key": "b", "last_update": "2026-08-28T14:00:00Z"}
    with_time = {
        "key": "h2h",
        "last_update": "2026-08-28T14:30:00Z",
        "outcomes": [{"name": "A", "price": 2.0}],
    }
    without = {"key": "h2h", "outcomes": [{"name": "A", "price": 2.0}]}
    event = {
        "id": "e1",
        "sport_key": "s",
        "commence_time": "2026-08-29T00:00:00Z",
        "home_team": "A",
        "away_team": "B",
    }
    events = [
        dict(event, bookmakers=[dict(book, markets=[with_time]), dict(book, markets=[without])])
    ]
    first, second = normalise_odds_payload(events, captured_at=CAPTURED_AT)
    assert first["market_last_update_utc"] == datetime(2026, 8, 28, 14, 30, tzinfo=UTC)
    assert first["bookmaker_last_update_utc"] == datetime(2026, 8, 28, 14, 0, tzinfo=UTC)
    assert second["market_last_update_utc"] is None


def test_prices_and_lines_survive_normalisation(rows: list[dict[str, object]]) -> None:
    assert all(float(r["price_decimal"]) > 1.0 for r in rows)  # type: ignore[arg-type]
    totals = [r for r in rows if r["market_key"] == "totals"]
    assert totals, "fixture should contain totals"
    assert all(r["outcome_point"] is not None for r in totals)
    assert {r["outcome_name"] for r in totals} == {"Over", "Under"}


def test_timestamps_are_timezone_aware_utc(rows: list[dict[str, object]]) -> None:
    for field in ("commence_time_utc", "bookmaker_last_update_utc"):
        assert all(r[field].tzinfo is not None for r in rows)  # type: ignore[union-attr]


@pytest.mark.parametrize("bookmakers", [None, []], ids=["empty payload", "no bookmakers"])
def test_nothing_to_normalise_yields_no_rows(bookmakers: list[Any] | None) -> None:
    events = [] if bookmakers is None else [dict(e, bookmakers=bookmakers) for e in payload()]
    assert normalise_odds_payload(events, captured_at=CAPTURED_AT) == []


def test_credits_per_cycle_is_the_product_of_the_three_dimensions() -> None:
    # one credit per region per market per sport
    assert credits_per_cycle(sports=["a", "b"], regions=["uk"], markets=["totals"]) == 2
    assert credits_per_cycle(sports=["a"], regions=["uk", "us"], markets=["totals", "h2h"]) == 4


def test_naming_books_costs_one_credit_per_ten_books_instead_of_per_region() -> None:
    # verifying the 3 bet books costs half a full uk and eu poll
    full = credits_per_cycle(sports=["nba"], regions=["uk", "eu"], markets=["h2h"])
    verify = credits_per_cycle(
        sports=["nba"], regions=["uk", "eu"], markets=["h2h"], bookmakers=["a", "b", "c"]
    )
    assert (full, verify) == (2, 1)
    eleven = [f"book{i}" for i in range(11)]
    assert credits_per_cycle(sports=["nba"], regions=[], markets=["h2h"], bookmakers=eleven) == 2


def test_a_verification_request_names_books_and_games_instead_of_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # capture what would be sent, without touching the network
    sent: dict[str, Any] = {}

    def fake_get(url: str, *, params: dict[str, str], timeout: float) -> httpx.Response:
        sent.update(params)
        return httpx.Response(200, json=[], headers={"x-requests-remaining": "441"})

    monkeypatch.setattr(httpx, "get", fake_get)
    body, remaining = fetch_odds(
        api_key="k",
        sport="basketball_nba",
        regions=["uk", "eu"],
        markets=["h2h"],
        bookmakers=["skybet", "paddypower"],
        event_ids=["e1", "e2"],
    )
    assert (body, remaining) == ([], 441)
    assert sent["bookmakers"] == "skybet,paddypower"
    assert sent["eventIds"] == "e1,e2"
    assert "regions" not in sent


def test_a_200_that_is_not_json_is_an_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # a reply that is not JSON must raise OddsApiError
    def fake_get(url: str, *, params: dict[str, str], timeout: float) -> httpx.Response:
        return httpx.Response(200, text="<html>down for maintenance</html>")

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(OddsApiError, match="not JSON"):
        fetch_events(api_key="k", sport="basketball_nba")


def test_partition_path_is_sport_and_date_partitioned() -> None:
    path = partition_path(Path("/tmp/arch"), sport="soccer_epl", captured_at=CAPTURED_AT)
    assert path.parent == Path("/tmp/arch/sport=soccer_epl/date=2026-08-28")
    assert path.suffix == ".parquet"
    assert "20260828T150000Z" in path.name


def test_write_snapshot_round_trips(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    written = write_snapshot(tmp_path, sport="soccer_epl", rows=rows, captured_at=CAPTURED_AT)
    back = pd.read_parquet(written)
    assert len(back) == len(rows)
    assert set(back.columns) == set(QUOTE_COLUMNS)
    assert (back["price_decimal"] > 1.0).all()


def test_write_snapshot_never_overwrites_an_earlier_capture(
    tmp_path: Path, rows: list[dict[str, object]]
) -> None:
    # a second capture is a new file, the first is untouched
    later = datetime(2026, 8, 28, 16, 30, tzinfo=UTC)
    first = write_snapshot(tmp_path, sport="soccer_epl", rows=rows, captured_at=CAPTURED_AT)
    second = write_snapshot(tmp_path, sport="soccer_epl", rows=rows, captured_at=later)
    assert first != second
    assert first.exists() and second.exists()
    assert len(list(first.parent.glob("*.parquet"))) == 2


def test_a_second_capture_at_the_same_second_is_refused(
    tmp_path: Path, rows: list[dict[str, object]]
) -> None:
    # the first capture must survive a re-run
    first = write_snapshot(tmp_path, sport="soccer_epl", rows=rows, captured_at=CAPTURED_AT)
    assert first is not None
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        write_snapshot(tmp_path, sport="soccer_epl", rows=rows[:1], captured_at=CAPTURED_AT)
    assert first.read_bytes() == before


def test_write_snapshot_leaves_no_temporary_file(
    tmp_path: Path, rows: list[dict[str, object]]
) -> None:
    # the file is written under a temporary name and renamed into place
    written = write_snapshot(tmp_path, sport="soccer_epl", rows=rows, captured_at=CAPTURED_AT)
    assert written is not None
    assert [p.name for p in written.parent.iterdir()] == [written.name]


def test_write_snapshot_with_no_rows_writes_nothing(tmp_path: Path) -> None:
    assert write_snapshot(tmp_path, sport="soccer_epl", rows=[], captured_at=CAPTURED_AT) is None
    assert list(tmp_path.rglob("*.parquet")) == []


def test_tagged_captures_are_told_apart_by_their_file_names(
    tmp_path: Path, rows: list[dict[str, object]]
) -> None:
    # a full poll at 15:00, a verification poll at 15:05: each kind's last run is known
    later = CAPTURED_AT + timedelta(minutes=5)
    write_snapshot(tmp_path, sport="soccer_epl", rows=rows, captured_at=CAPTURED_AT)
    tagged = write_snapshot(
        tmp_path, sport="soccer_epl", rows=rows, captured_at=later, tag="verify"
    )
    assert tagged is not None and tagged.name == "20260828T150500Z-verify.parquet"
    assert last_capture(tmp_path, sport="soccer_epl") == CAPTURED_AT
    assert last_capture(tmp_path, sport="soccer_epl", tag="verify") == later
    assert last_capture(tmp_path, sport="soccer_epl", tag="close") is None
    assert last_capture(tmp_path, sport="basketball_nba") is None


def test_events_become_the_schedule_free_of_charge(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, params: dict[str, str], timeout: float) -> httpx.Response:
        assert url.endswith("/sports/basketball_nba/events")
        body = [{"id": "e1", "commence_time": "2026-11-09T00:10:00Z", "home_team": "A"}]
        return httpx.Response(200, json=body, headers={"x-requests-remaining": "443"})

    monkeypatch.setattr(httpx, "get", fake_get)
    events, remaining = fetch_events(api_key="k", sport="basketball_nba")
    assert remaining == 443
    assert [(e.event_id, e.sport, e.commence_utc) for e in events] == [
        ("e1", "basketball_nba", datetime(2026, 11, 9, 0, 10, tzinfo=UTC))
    ]


def test_results_name_the_winner_or_a_draw_and_skip_unfinished_games() -> None:
    scores = [
        {
            "id": "e1",
            "completed": True,
            "scores": [{"name": "A", "score": "101"}, {"name": "B", "score": "99"}],
        },
        {
            "id": "e2",
            "completed": True,
            "scores": [{"name": "C", "score": "1"}, {"name": "D", "score": "1"}],
        },
        {
            "id": "e3",
            "completed": False,
            "scores": [{"name": "E", "score": "50"}, {"name": "F", "score": "40"}],
        },
        {"id": "e4", "completed": False, "scores": None},
    ]
    results = results_from_scores(scores)
    assert {k: r.winner for k, r in results.items()} == {"e1": "A", "e2": "Draw"}
