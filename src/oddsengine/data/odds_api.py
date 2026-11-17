"""Fetches odds from the Odds API (the-odds-api.com) and flattens them into rows.

Billing is one credit per sport per region per market, per call.
"""

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import httpx

from oddsengine.types import EventInfo, Result

BASE_URL = "https://api.the-odds-api.com/v4"

# credits for a scores call; the events call is free
RESULTS_CREDITS = 2

# one row per (event, book, market, outcome)
QUOTE_COLUMNS = (
    "captured_at_utc",
    "sport_key",
    "event_id",
    "commence_time_utc",
    "home_team",
    "away_team",
    "bookmaker_key",
    "market_key",
    "outcome_name",
    "outcome_point",
    "price_decimal",
    "bookmaker_last_update_utc",
    "market_last_update_utc",
)


# raised when a call to the API fails
class OddsApiError(RuntimeError):
    pass


def parse_utc(raw: str) -> datetime:
    # turns a time string into a UTC datetime
    return datetime.fromisoformat(raw).astimezone(UTC)


def normalise_odds_payload(
    payload: Sequence[dict[str, Any]], *, captured_at: datetime
) -> list[dict[str, Any]]:
    # flattens events, books, markets and outcomes into one row per outcome
    rows: list[dict[str, Any]] = []
    for event in payload:
        commence = parse_utc(event["commence_time"])
        for book in event.get("bookmakers", []):
            updated = parse_utc(book["last_update"])
            for market in book.get("markets", []):
                # the market's own update time, if sent
                market_raw = market.get("last_update")
                market_updated = None if market_raw is None else parse_utc(market_raw)
                for outcome in market.get("outcomes", []):
                    point = outcome.get("point")
                    rows.append(
                        {
                            "captured_at_utc": captured_at,
                            "sport_key": event["sport_key"],
                            "event_id": event["id"],
                            "commence_time_utc": commence,
                            "home_team": event["home_team"],
                            "away_team": event["away_team"],
                            "bookmaker_key": book["key"],
                            "market_key": market["key"],
                            "outcome_name": outcome["name"],
                            "outcome_point": None if point is None else float(point),
                            "price_decimal": float(outcome["price"]),
                            "bookmaker_last_update_utc": updated,
                            "market_last_update_utc": market_updated,
                        }
                    )
    return rows


def credits_per_cycle(
    *,
    sports: Sequence[str],
    regions: Sequence[str],
    markets: Sequence[str],
    bookmakers: Sequence[str] = (),
) -> int:
    # naming bookmakers replaces regions, and each group of 10 books costs one region
    per_sport = math.ceil(len(bookmakers) / 10) if bookmakers else len(regions)
    return len(sports) * per_sport * len(markets)


def fetch_odds(
    *,
    api_key: str,
    sport: str,
    regions: Sequence[str],
    markets: Sequence[str],
    bookmakers: Sequence[str] = (),
    event_ids: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], int | None]:
    # fetches one sport's odds and the credits left
    params = {
        "apiKey": api_key,
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    else:
        params["regions"] = ",".join(regions)
    if event_ids:
        params["eventIds"] = ",".join(event_ids)
    return _get(f"/sports/{sport}/odds/", params)


def fetch_events(*, api_key: str, sport: str) -> tuple[list[EventInfo], int | None]:
    # fetches one sport's schedule and the credits left
    payload, remaining = _get(f"/sports/{sport}/events", {"apiKey": api_key, "dateFormat": "iso"})
    events = []
    for event in payload:
        events.append(
            EventInfo(
                event_id=event["id"],
                sport=sport,
                commence_utc=parse_utc(event["commence_time"]),
                home_team=event.get("home_team") or "",
                away_team=event.get("away_team") or "",
            )
        )
    return events, remaining


def fetch_scores(
    *, api_key: str, sport: str, days_from: int = 3
) -> tuple[list[dict[str, Any]], int | None]:
    # fetches live and recent games with their scores
    params = {"apiKey": api_key, "daysFrom": str(days_from), "dateFormat": "iso"}
    return _get(f"/sports/{sport}/scores/", params)


def results_from_scores(payload: Sequence[dict[str, Any]]) -> dict[str, Result]:
    # finds the winner of every completed game, or Draw if level
    results = {}
    for game in payload:
        if not game.get("completed") or not game.get("scores"):
            continue
        points = {}
        for side in game["scores"]:
            points[side["name"]] = float(side["score"])
        best = max(points.values())
        leaders = [name for name, score in points.items() if score == best]
        winner = leaders[0] if len(leaders) == 1 else "Draw"
        results[game["id"]] = Result(event_id=game["id"], winner=winner)
    return results


def _get(path: str, params: dict[str, str]) -> tuple[list[Any], int | None]:
    # makes one request and turns any failure into OddsApiError
    try:
        response = httpx.get(f"{BASE_URL}{path}", params=params, timeout=30.0)
    except httpx.HTTPError as exc:
        raise OddsApiError(f"request failed for {path}: {exc}") from exc
    if response.status_code != 200:
        raise OddsApiError(f"{path}: HTTP {response.status_code}: {response.text[:200]}")

    # treats a reply that is not JSON as a failure
    try:
        payload = response.json()
    except ValueError as exc:
        raise OddsApiError(f"{path}: response is not JSON: {response.text[:200]}") from exc
    if not isinstance(payload, list):
        raise OddsApiError(f"{path}: unexpected payload type {type(payload).__name__}")
    remaining = response.headers.get("x-requests-remaining")
    return payload, None if remaining is None else int(remaining)
