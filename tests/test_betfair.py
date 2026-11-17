"""Betfair Exchange tests, with no network: every HTTP call is faked with hand-made responses."""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from oddsengine.config import BetfairMarket
from oddsengine.data import betfair
from oddsengine.data.archive import captures_since, write_snapshot
from oddsengine.data.betfair import (
    BOOK_COLUMNS,
    BetfairError,
    LoginRefused,
    Market,
    Session,
    SessionExpired,
    book_rows,
    call,
    connect,
    exchange_probs,
    list_books,
    list_markets,
    map_markets,
    same_team,
)
from oddsengine.types import EventInfo

NOW = datetime(2026, 11, 9, 22, 0, tzinfo=UTC)
HOUR = timedelta(hours=1)
SESSION = Session(app_key="app", token="tok")


class FakePost:
    # fake for httpx.post that records calls and answers from a queue
    def __init__(self, *answers: httpx.Response) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        return self.answers.pop(0)


def fake(monkeypatch: pytest.MonkeyPatch, *answers: httpx.Response) -> FakePost:
    post = FakePost(*answers)
    monkeypatch.setattr(httpx, "post", post)
    return post


def ok_login(token: str = "fresh") -> httpx.Response:
    body = {"token": token, "product": "app", "status": "SUCCESS", "error": ""}
    return httpx.Response(200, json=body)


def game(event_id: str, home: str, away: str, start: datetime = NOW + 2 * HOUR) -> EventInfo:
    return EventInfo(
        event_id=event_id,
        sport="basketball_nba",
        commence_utc=start,
        home_team=home,
        away_team=away,
    )


def entry(market_id: str, start: datetime, *names: str) -> dict[str, Any]:
    # one listMarketCatalogue item, runners numbered from 101
    runners = [
        {"selectionId": 101 + i, "runnerName": name, "handicap": 0.0, "sortPriority": i + 1}
        for i, name in enumerate(names)
    ]
    return {
        "marketId": market_id,
        "marketName": "Match Odds",
        "marketStartTime": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "runners": runners,
        "event": {"id": "3001", "name": " @ ".join(names[:2]), "countryCode": "US"},
    }


def book(market_id: str, offers: dict[int, tuple[list[float], list[float]]]) -> dict[str, Any]:
    # one listMarketBook item, as selection id to (back prices, lay prices), size 100 each
    runners = []
    for selection, (backs, lays) in offers.items():
        ex = {
            "availableToBack": [{"price": p, "size": 100.0} for p in backs],
            "availableToLay": [{"price": p, "size": 100.0} for p in lays],
            "tradedVolume": [],
        }
        runners.append({"selectionId": selection, "handicap": 0.0, "status": "ACTIVE", "ex": ex})
    return {
        "marketId": market_id,
        "isMarketDataDelayed": True,
        "status": "OPEN",
        "inplay": False,
        "runners": runners,
    }


def connect_now(
    cache: Path, *, at: datetime = NOW, write: bool = True, fresh: bool = False
) -> Session:
    return connect(
        app_key="app",
        username="u",
        password="p",
        cache=cache,
        now=at,
        keep_alive_every=HOUR,
        write=write,
        fresh=fresh,
    )


def test_login_posts_the_form_with_the_app_key_and_returns_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = fake(monkeypatch, ok_login("t1"))
    assert betfair.login(app_key="app", username="u", password="p") == "t1"
    sent = post.calls[0]
    assert sent["url"] == "https://identitysso.betfair.com/api/login"
    assert sent["data"] == {"username": "u", "password": "p"}
    assert sent["headers"] == {"X-Application": "app", "Accept": "application/json"}


def test_a_refused_login_names_the_error_but_never_the_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"token": "", "status": "FAIL", "error": "INVALID_USERNAME_OR_PASSWORD"}
    fake(monkeypatch, httpx.Response(200, json=body))
    with pytest.raises(LoginRefused) as caught:
        betfair.login(app_key="app", username="oliver", password="hunter2")
    assert "INVALID_USERNAME_OR_PASSWORD" in str(caught.value)
    assert "oliver" not in str(caught.value) and "hunter2" not in str(caught.value)


def test_a_session_never_shows_its_token_or_app_key() -> None:
    assert repr(SESSION) == "Session()"


def test_the_first_connect_logs_in_and_caches_the_token_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "session.json"
    fake(monkeypatch, ok_login("t1"))
    assert connect_now(cache).token == "t1"
    assert json.loads(cache.read_text()) == {"token": "t1", "at": NOW.isoformat()}
    assert cache.stat().st_mode & 0o777 == 0o600


def test_a_recent_cached_token_is_reused_without_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "session.json"
    cache.write_text(json.dumps({"token": "t0", "at": (NOW - timedelta(minutes=30)).isoformat()}))
    post = fake(monkeypatch)
    assert connect_now(cache).token == "t0"
    assert post.calls == []


def test_an_older_token_is_kept_alive_and_a_dead_one_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "session.json"
    old = {"token": "t0", "at": (NOW - 2 * HOUR).isoformat()}

    # still alive, so it is kept
    cache.write_text(json.dumps(old))
    alive = httpx.Response(200, json={"token": "t0", "status": "SUCCESS", "error": ""})
    post = fake(monkeypatch, alive)
    assert connect_now(cache).token == "t0"
    assert post.calls[0]["url"].endswith("/api/keepAlive")
    assert post.calls[0]["headers"]["X-Authentication"] == "t0"
    assert json.loads(cache.read_text())["at"] == NOW.isoformat()

    # expired: a new login
    cache.write_text(json.dumps(old))
    dead = httpx.Response(200, json={"token": "", "status": "FAIL", "error": "NO_SESSION"})
    fake(monkeypatch, dead, ok_login("t1"))
    assert connect_now(cache).token == "t1"


def test_a_dry_run_never_logs_in_or_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "session.json"
    post = fake(monkeypatch)
    with pytest.raises(BetfairError, match="dry run"):
        connect_now(cache, write=False)
    assert post.calls == [] and not cache.exists()


def test_a_refused_login_is_never_retried_by_the_loop_until_a_fresh_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "session.json"
    refused = httpx.Response(200, json={"status": "FAIL", "error": "INVALID_USERNAME_OR_PASSWORD"})
    fake(monkeypatch, refused)
    with pytest.raises(LoginRefused):
        connect_now(cache)

    # the next tick does not try again
    post = fake(monkeypatch)
    with pytest.raises(BetfairError, match="betfair_check"):
        connect_now(cache, at=NOW + HOUR)
    assert post.calls == []

    # a fresh login clears it
    fake(monkeypatch, ok_login("t2"))
    assert connect_now(cache, fresh=True).token == "t2"
    assert connect_now(cache).token == "t2"


def test_forget_drops_the_token(tmp_path: Path) -> None:
    cache = tmp_path / "session.json"
    cache.write_text(json.dumps({"token": "t0", "at": NOW.isoformat()}))
    betfair.forget(cache)
    assert json.loads(cache.read_text()) == {}


def test_only_read_only_operations_can_be_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    post = fake(monkeypatch)
    for operation in ("placeOrders", "cancelOrders", "replaceOrders", "updateOrders"):
        with pytest.raises(ValueError, match="read-only"):
            call(SESSION, operation, {})
    assert post.calls == []


def test_a_rejected_session_is_told_apart_from_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def error(code: str) -> httpx.Response:
        detail = {"APINGException": {"errorCode": code, "errorDetails": ""}}
        return httpx.Response(400, json={"faultcode": "Client", "detail": detail})

    fake(monkeypatch, error("INVALID_SESSION_INFORMATION"), error("TOO_MUCH_DATA"))
    with pytest.raises(SessionExpired):
        call(SESSION, "listMarketBook", {})
    with pytest.raises(BetfairError, match="TOO_MUCH_DATA") as caught:
        call(SESSION, "listMarketBook", {})
    assert not isinstance(caught.value, SessionExpired)


def test_the_catalogue_asks_for_one_sports_market_type_in_a_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = fake(monkeypatch, httpx.Response(200, json=[]), httpx.Response(200, json=[]))
    nba = BetfairMarket(event_type="7522", market_type="MATCH_ODDS")
    assert list_markets(SESSION, nba, start_from=NOW, start_to=NOW + 6 * HOUR) == []
    sent = post.calls[0]
    assert sent["url"] == "https://api.betfair.com/exchange/betting/rest/v1.0/listMarketCatalogue/"
    assert sent["headers"]["X-Application"] == "app"
    assert sent["headers"]["X-Authentication"] == "tok"
    assert sent["json"]["filter"] == {
        "eventTypeIds": ["7522"],
        "marketTypeCodes": ["MATCH_ODDS"],
        "marketStartTime": {"from": "2026-11-09T22:00:00Z", "to": "2026-11-10T04:00:00Z"},
    }
    assert set(sent["json"]["marketProjection"]) == {
        "RUNNER_DESCRIPTION",
        "EVENT",
        "MARKET_START_TIME",
    }

    # a football market is narrowed by country
    epl = BetfairMarket(event_type="1", market_type="MATCH_ODDS", countries=["GB"])
    list_markets(SESSION, epl, start_from=NOW, start_to=NOW + HOUR)
    assert post.calls[1]["json"]["filter"]["marketCountries"] == ["GB"]


def test_books_are_fetched_in_batches_within_the_weight_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [f"1.{i}" for i in range(85)]
    answers = [httpx.Response(200, json=[{"marketId": "x"}]) for _ in range(3)]
    post = fake(monkeypatch, *answers)
    assert len(list_books(SESSION, ids)) == 3
    assert [len(c["json"]["marketIds"]) for c in post.calls] == [40, 40, 5]
    assert all(
        c["json"]["priceProjection"] == {"priceData": ["EX_BEST_OFFERS"]} for c in post.calls
    )


@pytest.mark.parametrize(
    ("betfair_name", "odds_api_name"),
    [
        ("Los Angeles Lakers", "Los Angeles Lakers"),
        ("LA Lakers", "Los Angeles Lakers"),
        ("Okc Thunder", "Oklahoma City Thunder"),
        ("NY Rangers", "New York Rangers"),
        ("Montreal Canadiens", "Montréal Canadiens"),
        ("St Louis Blues", "St. Louis Blues"),
        ("Man Utd", "Manchester United"),
        ("Man City", "Manchester City"),
        ("Nottm Forest", "Nottingham Forest"),
        ("QPR", "Queens Park Rangers"),
        ("Sheff Wed", "Sheffield Wednesday"),
        ("West Brom", "West Bromwich Albion"),
        ("Brighton", "Brighton and Hove Albion"),
        ("Wrexham", "Wrexham AFC"),
    ],
)
def test_the_same_team_under_two_spellings(betfair_name: str, odds_api_name: str) -> None:
    assert same_team(betfair_name, odds_api_name)


@pytest.mark.parametrize(
    ("betfair_name", "odds_api_name"),
    [
        ("LA Clippers", "Los Angeles Lakers"),
        ("Man City", "Manchester United"),
        ("Man Utd", "Manchester City"),
        ("West Ham", "West Bromwich Albion"),
        ("Bristol City", "Birmingham City"),
        ("NY Rangers", "New York Islanders"),
        ("Boston Celtics", ""),
    ],
)
def test_different_teams_are_told_apart(betfair_name: str, odds_api_name: str) -> None:
    assert not same_team(betfair_name, odds_api_name)


@given(st.lists(st.from_regex(r"[A-Za-z]{1,10}", fullmatch=True), min_size=1, max_size=4))
def test_every_name_is_its_own_team(words: list[str]) -> None:
    name = " ".join(words)
    assert same_team(name, name) or all(w.lower() in betfair.FILLER for w in words)


def test_a_market_maps_to_its_game_whichever_way_round_betfair_lists_it() -> None:
    games = [
        game("e1", "Detroit Pistons", "Los Angeles Lakers"),
        game("e2", "Utah Jazz", "Miami Heat"),
    ]
    catalogue = [
        entry("1.1", NOW + 2 * HOUR + timedelta(minutes=10), "LA Lakers", "Detroit Pistons"),
        entry("1.2", NOW + 2 * HOUR, "Utah Jazz", "Miami Heat"),
    ]
    matched, unmatched = map_markets(
        catalogue, games, two_way=True, tolerance=timedelta(minutes=15), aliases={}
    )
    assert unmatched == []
    assert sorted(matched, key=lambda m: m.market_id) == [
        Market(
            market_id="1.1",
            event_id="e1",
            runners={101: "Los Angeles Lakers", 102: "Detroit Pistons"},
        ),
        Market(market_id="1.2", event_id="e2", runners={101: "Utah Jazz", 102: "Miami Heat"}),
    ]


def test_markets_off_the_schedule_or_too_far_from_its_start_stay_unmatched() -> None:
    games = [game("e1", "Detroit Pistons", "Los Angeles Lakers")]
    catalogue = [
        entry("1.1", NOW + 3 * HOUR, "LA Lakers", "Detroit Pistons"),  # an hour off
        entry("1.2", NOW + 2 * HOUR, "Real Madrid", "Barcelona"),  # another league
    ]
    matched, unmatched = map_markets(
        catalogue, games, two_way=True, tolerance=timedelta(minutes=15), aliases={}
    )
    assert matched == [] and [e["marketId"] for e in unmatched] == ["1.1", "1.2"]


def test_a_3_way_market_never_maps_where_settlement_is_2_way() -> None:
    # the NHL regulation market quotes a draw but results include overtime
    games = [game("e1", "Winnipeg Jets", "Tampa Bay Lightning")]
    three_way = [entry("1.1", NOW + 2 * HOUR, "Winnipeg Jets", "Tampa Bay Lightning", "The Draw")]
    matched, _ = map_markets(three_way, games, two_way=True, tolerance=HOUR, aliases={})
    assert matched == []

    # in football the draw is a selection
    matched, _ = map_markets(three_way, games, two_way=False, tolerance=HOUR, aliases={})
    assert matched[0].runners == {101: "Winnipeg Jets", 102: "Tampa Bay Lightning", 103: "Draw"}


def test_an_ambiguous_match_is_left_unmatched_never_guessed() -> None:
    # two markets match the same game, so neither is used
    games = [game("e1", "Detroit Pistons", "Los Angeles Lakers")]
    catalogue = [
        entry("1.1", NOW + 2 * HOUR, "LA Lakers", "Detroit Pistons"),
        entry("1.2", NOW + 2 * HOUR, "Los Angeles Lakers", "Detroit"),
    ]
    matched, unmatched = map_markets(catalogue, games, two_way=True, tolerance=HOUR, aliases={})
    assert matched == [] and len(unmatched) == 2


def test_an_alias_maps_a_name_the_rules_miss() -> None:
    games = [game("e1", "Wolverhampton Wanderers", "Wrexham AFC")]
    catalogue = [entry("1.1", NOW + 2 * HOUR, "Spurs Of The Midlands", "Wrexham")]
    aliases = {"Spurs Of The Midlands": "Wolverhampton Wanderers"}
    assert map_markets(catalogue, games, two_way=True, tolerance=HOUR, aliases={})[0] == []
    matched, _ = map_markets(catalogue, games, two_way=True, tolerance=HOUR, aliases=aliases)
    assert matched[0].runners == {101: "Wolverhampton Wanderers", 102: "Wrexham AFC"}


MARKET = Market(market_id="1.1", event_id="e1", runners={101: "Home", 102: "Away"})


def test_book_rows_keep_three_levels_each_side_and_skip_unmapped_markets() -> None:
    books = [
        book("1.1", {101: ([1.90, 1.89, 1.88], [1.92, 1.93]), 102: ([2.08], [2.12, 2.14, 2.16])}),
        book("9.9", {101: ([1.5], [1.6])}),  # not mapped
    ]
    rows = book_rows(books, [MARKET], captured_at=NOW)
    assert [set(r) for r in rows] == [set(BOOK_COLUMNS)] * 2
    home, away = rows
    assert (home["runner"], home["event_id"], home["market_id"]) == ("Home", "e1", "1.1")
    assert (home["back_price_1"], home["back_price_3"], home["lay_price_2"]) == (1.90, 1.88, 1.93)
    assert math.isnan(home["lay_price_3"]) and math.isnan(away["back_size_2"])


def test_book_rows_round_trip_through_the_archive(tmp_path: Path) -> None:
    # same layout as the Odds API archive, under its own root
    rows = book_rows([book("1.1", {101: ([1.9], [1.92])})], [MARKET], captured_at=NOW)
    path = write_snapshot(
        tmp_path, sport="basketball_nba", rows=rows, captured_at=NOW, columns=BOOK_COLUMNS
    )
    assert (
        path == tmp_path / "sport=basketball_nba" / "date=2026-11-09" / "20261109T220000Z.parquet"
    )
    back = pd.read_parquet(path)
    assert list(back.columns) == list(BOOK_COLUMNS)
    assert exchange_probs(back, max_spread=0.02) == exchange_probs(
        pd.DataFrame(rows), max_spread=0.02
    )


def test_the_exchange_view_is_the_mid_of_best_back_and_lay() -> None:
    rows = book_rows(
        [book("1.1", {101: ([1.90], [1.92]), 102: ([2.08], [2.12])})], [MARKET], captured_at=NOW
    )
    probs = exchange_probs(pd.DataFrame(rows), max_spread=0.02)
    assert probs == {"e1": {"Home": pytest.approx(2 / 3.82), "Away": pytest.approx(2 / 4.20)}}


def test_runners_without_both_sides_or_with_a_wide_spread_are_ignored() -> None:
    offers = {101: ([1.90], []), 102: ([2.0], [2.5])}  # no lay; 0.5 - 0.4 = 0.1 apart
    rows = book_rows([book("1.1", offers)], [MARKET], captured_at=NOW)
    assert exchange_probs(pd.DataFrame(rows), max_spread=0.02) == {}


@pytest.mark.parametrize("change", [{"status": "SUSPENDED"}, {"inplay": True}])
def test_suspended_and_in_play_markets_are_ignored(change: dict[str, Any]) -> None:
    item = dict(book("1.1", {101: ([1.9], [1.92])}), **change)
    rows = book_rows([item], [MARKET], captured_at=NOW)
    assert exchange_probs(pd.DataFrame(rows), max_spread=0.02) == {}


@given(st.floats(1.01, 50.0), st.floats(0.0, 0.05))
def test_the_mid_probability_lies_between_the_back_and_lay(back: float, gap: float) -> None:
    lay = back * (1 + gap)
    rows = book_rows([book("1.1", {101: ([back], [lay])})], [MARKET], captured_at=NOW)
    probs = exchange_probs(pd.DataFrame(rows), max_spread=1.0)
    assert 1 / lay - 1e-12 <= probs["e1"]["Home"] <= 1 / back + 1e-12


def test_captures_since_lists_each_capture_with_its_tag(tmp_path: Path) -> None:
    rows = book_rows([book("1.1", {101: ([1.9], [1.92])})], [MARKET], captured_at=NOW)
    for at, tag in ((NOW - 2 * HOUR, ""), (NOW - HOUR, "trigger"), (NOW, "verify")):
        write_snapshot(
            tmp_path,
            sport="basketball_nba",
            rows=rows,
            captured_at=at,
            tag=tag,
            columns=BOOK_COLUMNS,
        )
    found = captures_since(tmp_path, sport="basketball_nba", since=NOW - HOUR)
    assert [(at, tag) for at, tag, _ in found] == [(NOW - HOUR, "trigger"), (NOW, "verify")]
    assert captures_since(tmp_path, sport="icehockey_nhl", since=NOW) == []
