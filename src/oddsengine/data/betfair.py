"""Reads Betfair Exchange prices: logs in, finds each game's market and flattens its
best offers into rows.

Only read-only operations are ever sent, so nothing here can place or cancel a bet.
"""

import json
import math
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict

from oddsengine.config import BetfairMarket
from oddsengine.data.odds_api import parse_utc
from oddsengine.types import EventInfo

LOGIN_URL = "https://identitysso.betfair.com/api/login"
KEEP_ALIVE_URL = "https://identitysso.betfair.com/api/keepAlive"
BETTING_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"

# the only operations this module will send
READ_ONLY = frozenset({"listMarketCatalogue", "listMarketBook", "listMarketTypes"})

# markets per request, to stay within Betfair's request weight limit
BOOK_BATCH = 40

# error codes for a missing or expired session
EXPIRED = ("INVALID_SESSION_INFORMATION", "NO_SESSION")

# names Betfair gives the draw
DRAWS = frozenset({"the draw", "draw"})

# words ignored when matching team names
FILLER = frozenset({"fc", "afc", "and", "the"})

# one row per runner, with the best 3 offers each side
LEVELS = 3
BOOK_COLUMNS = (
    "captured_at_utc",
    "event_id",  # the Odds API's
    "market_id",  # Betfair's
    "market_status",
    "inplay",
    "selection_id",
    "runner",  # the Odds API's selection name, "Draw" for the draw
    "back_price_1",
    "back_size_1",
    "back_price_2",
    "back_size_2",
    "back_price_3",
    "back_size_3",
    "lay_price_1",
    "lay_size_1",
    "lay_price_2",
    "lay_size_2",
    "lay_price_3",
    "lay_size_3",
)


# raised when a call to Betfair fails
class BetfairError(RuntimeError):
    pass


# raised when the session token is rejected
class SessionExpired(BetfairError):
    pass


# raised when Betfair refuses the login
class LoginRefused(BetfairError):
    pass


# a logged-in session, with its values hidden from repr
@dataclass(frozen=True)
class Session:
    app_key: str = field(repr=False)
    token: str = field(repr=False)


# one Betfair market matched to one Odds API game
class Market(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    market_id: str
    event_id: str  # the Odds API's
    runners: dict[int, str]  # Betfair selection id to the Odds API's selection name


def login(*, app_key: str, username: str, password: str) -> str:
    # logs in and returns a new session token
    try:
        response = httpx.post(
            LOGIN_URL,
            data={"username": username, "password": password},
            headers={"X-Application": app_key, "Accept": "application/json"},
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise BetfairError(f"login request failed ({type(exc).__name__})") from None
    body = _identity(response, "login")
    if body.get("status") != "SUCCESS" or not body.get("token"):
        raise LoginRefused(f"login {body.get('status')}: {body.get('error')}")
    return str(body["token"])


def keep_alive(session: Session) -> bool:
    # extends the session, False if Betfair no longer knows it
    try:
        response = httpx.post(KEEP_ALIVE_URL, headers=_headers(session), timeout=30.0)
    except httpx.HTTPError as exc:
        raise BetfairError(f"keepAlive request failed ({type(exc).__name__})") from None
    return _identity(response, "keepAlive").get("status") == "SUCCESS"


def connect(
    *,
    app_key: str,
    username: str,
    password: str,
    cache: Path,
    now: datetime,
    keep_alive_every: timedelta,
    write: bool,
    fresh: bool = False,
) -> Session:
    # reuses the cached session if it is still good, else logs in
    saved = _read_cache(cache)
    if not fresh:
        if "refused" in saved:
            raise BetfairError(f"{saved['refused']} earlier; run scripts/betfair_check.py")
        if "token" in saved:
            session = Session(app_key=app_key, token=saved["token"])
            if now - datetime.fromisoformat(saved["at"]) < keep_alive_every:
                return session
            if keep_alive(session):
                if write:
                    _write_cache(cache, {"token": session.token, "at": now.isoformat()})
                return session
    if not write:
        raise BetfairError("no live session, and a dry run never logs in")

    # remembers a refused login so it is not retried
    try:
        token = login(app_key=app_key, username=username, password=password)
    except LoginRefused as exc:
        _write_cache(cache, {"refused": str(exc), "at": now.isoformat()})
        raise
    _write_cache(cache, {"token": token, "at": now.isoformat()})
    return Session(app_key=app_key, token=token)


def forget(cache: Path) -> None:
    # clears a rejected token
    _write_cache(cache, {})


def _read_cache(path: Path) -> dict[str, str]:
    # reads the saved session
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in saved.items()} if isinstance(saved, dict) else {}


def _write_cache(path: Path, saved: dict[str, str]) -> None:
    # saves the session to disk
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    # file is readable by the owner only
    handle = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w") as out:
        json.dump(saved, out)
    os.replace(temp, path)


def _headers(session: Session) -> dict[str, str]:
    # headers for authenticated requests
    return {
        "X-Application": session.app_key,
        "X-Authentication": session.token,
        "Accept": "application/json",
    }


def _identity(response: httpx.Response, what: str) -> dict[str, Any]:
    # decodes an answer from the identity service
    if response.status_code != 200:
        raise BetfairError(f"{what}: HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError:
        raise BetfairError(f"{what}: the answer was not JSON") from None
    if not isinstance(body, dict):
        raise BetfairError(f"{what}: unexpected answer")
    return body


def call(session: Session, operation: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    # makes one read-only call, refusing any other operation
    if operation not in READ_ONLY:
        raise ValueError(f"{operation} is not a read-only operation")
    try:
        response = httpx.post(
            f"{BETTING_URL}/{operation}/", json=body, headers=_headers(session), timeout=30.0
        )
    except httpx.HTTPError as exc:
        raise BetfairError(f"{operation} request failed ({type(exc).__name__})") from None

    # turns error replies into exceptions
    if response.status_code != 200:
        code = _error_code(response)
        if code in EXPIRED:
            raise SessionExpired(f"{operation}: {code}")
        raise BetfairError(f"{operation}: HTTP {response.status_code} {code}".rstrip())
    try:
        payload = response.json()
    except ValueError:
        raise BetfairError(f"{operation}: the answer was not JSON") from None
    if not isinstance(payload, list):
        raise BetfairError(f"{operation}: unexpected payload type {type(payload).__name__}")
    return payload


def _error_code(response: httpx.Response) -> str:
    # finds the error code in the reply
    try:
        return str(response.json()["detail"]["APINGException"]["errorCode"])
    except (ValueError, KeyError, TypeError):
        # falls back to searching the raw text
        return next((code for code in EXPIRED if code in response.text), "")


def list_markets(
    session: Session, market: BetfairMarket, *, start_from: datetime, start_to: datetime
) -> list[dict[str, Any]]:
    # lists one sport's markets starting in a window
    where: dict[str, Any] = {
        "eventTypeIds": [market.event_type],
        "marketTypeCodes": [market.market_type],
        "marketStartTime": {"from": _utc_string(start_from), "to": _utc_string(start_to)},
    }
    if market.countries:
        where["marketCountries"] = market.countries
    body = {
        "filter": where,
        "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
        "sort": "FIRST_TO_START",
        "maxResults": 1000,
    }
    return call(session, "listMarketCatalogue", body)


def list_books(session: Session, market_ids: Sequence[str]) -> list[dict[str, Any]]:
    # fetches the best offers on each market in batches
    books: list[dict[str, Any]] = []
    for start in range(0, len(market_ids), BOOK_BATCH):
        body = {
            "marketIds": list(market_ids[start : start + BOOK_BATCH]),
            "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
        }
        books += call(session, "listMarketBook", body)
    return books


def _utc_string(at: datetime) -> str:
    # formats the time as a UTC string
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def map_markets(
    catalogue: Sequence[dict[str, Any]],
    events: Sequence[EventInfo],
    *,
    two_way: bool,
    tolerance: timedelta,
    aliases: dict[str, str],
) -> tuple[list[Market], list[dict[str, Any]]]:
    # matches each game to a market with the same start and teams
    pairs = []
    for entry in catalogue:
        start = parse_utc(entry["marketStartTime"])
        for event in events:
            if abs(start - event.commence_utc) > tolerance:
                continue
            runners = _assign(entry.get("runners", []), event, two_way=two_way, aliases=aliases)
            if runners is not None:
                market_id = str(entry["marketId"])
                pairs.append(Market(market_id=market_id, event_id=event.event_id, runners=runners))

    # keeps only one-to-one matches, never guessing
    per_market = Counter(m.market_id for m in pairs)
    per_event = Counter(m.event_id for m in pairs)
    matched = [m for m in pairs if per_market[m.market_id] == 1 and per_event[m.event_id] == 1]
    ids = {m.market_id for m in matched}
    return matched, [e for e in catalogue if str(e["marketId"]) not in ids]


def _assign(
    runners: Sequence[dict[str, Any]],
    event: EventInfo,
    *,
    two_way: bool,
    aliases: dict[str, str],
) -> dict[int, str] | None:
    # maps selection ids to the game's sides, if the runners fit exactly one way round
    draws = [r for r in runners if str(r["runnerName"]).lower() in DRAWS]
    teams = [r for r in runners if str(r["runnerName"]).lower() not in DRAWS]
    if len(teams) != 2 or len(draws) != (0 if two_way else 1):
        return None
    # applies any name overrides from the config
    first, second = (aliases.get(r["runnerName"], r["runnerName"]) for r in teams)
    straight = same_team(first, event.home_team) and same_team(second, event.away_team)
    swapped = same_team(first, event.away_team) and same_team(second, event.home_team)
    if straight == swapped:
        return None
    home, away = teams if straight else teams[::-1]
    found = {int(home["selectionId"]): event.home_team, int(away["selectionId"]): event.away_team}
    found.update({int(d["selectionId"]): "Draw" for d in draws})
    return found


def same_team(one: str, other: str) -> bool:
    # checks if two names are one team, e.g. "Man Utd" and "Manchester United"
    a, b = _words(one), _words(other)
    if not a or not b:
        return False
    return _abbreviates(a, b) or _abbreviates(b, a)


def _words(name: str) -> list[str]:
    # splits a name into plain lower case words
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    plain = plain.replace("&", " and ").replace(".", "").replace("'", "")
    return [w for w in re.split(r"[^a-z0-9]+", plain) if w and w not in FILLER]


def _abbreviates(short: list[str], long: list[str]) -> bool:
    # checks each short word matches a later long word in turn
    j = 0
    for word in short:
        # skips long words this short word cannot stand for
        while j < len(long) and not _could_stand_for(word, long, j):
            j += 1
        if j == len(long):
            return False  # no long word left for this short word
        j += 1  # the next short word must match a later long word
    return True


def _could_stand_for(word: str, long: list[str], j: int) -> bool:
    # checks the first letter matches and the rest appear in order
    return long[j][0] == word[0] and _in_order(word, "".join(long[j:]))


def _in_order(letters: str, text: str) -> bool:
    # checks the letters appear in text in order
    start = 0
    for letter in letters:
        found = text.find(letter, start)
        if found == -1:
            return False
        start = found + 1
    return True


def book_rows(
    books: Sequence[dict[str, Any]], markets: Sequence[Market], *, captured_at: datetime
) -> list[dict[str, Any]]:
    # turns each mapped market's runners into rows, with NaN for a missing offer
    by_id = {m.market_id: m for m in markets}
    rows = []
    for book in books:
        market = by_id.get(str(book.get("marketId")))
        if market is None:
            continue
        for runner in book.get("runners", []):
            name = market.runners.get(int(runner["selectionId"]))
            if name is None:
                continue
            row: dict[str, Any] = {
                "captured_at_utc": captured_at,
                "event_id": market.event_id,
                "market_id": market.market_id,
                "market_status": str(book.get("status", "")),
                "inplay": bool(book.get("inplay", False)),
                "selection_id": int(runner["selectionId"]),
                "runner": name,
            }

            # fills the best offers on each side
            offers = runner.get("ex", {})
            for side, key in (("back", "availableToBack"), ("lay", "availableToLay")):
                levels = offers.get(key, [])
                for i in range(LEVELS):
                    level = levels[i] if i < len(levels) else {}
                    row[f"{side}_price_{i + 1}"] = float(level.get("price", math.nan))
                    row[f"{side}_size_{i + 1}"] = float(level.get("size", math.nan))
            rows.append(row)
    return rows


def exchange_probs(frame: pd.DataFrame, *, max_spread: float) -> dict[str, dict[str, float]]:
    # finds each runner's probability from the middle of its best back and lay prices
    probs: dict[str, dict[str, float]] = {}

    # goes through one runner at a time
    rows = zip(
        frame["event_id"],
        frame["runner"],
        frame["market_status"],
        frame["inplay"],
        frame["back_price_1"],
        frame["lay_price_1"],
        strict=True,
    )
    for event_id, runner, status, inplay, back, lay in rows:
        if status != "OPEN" or inplay or pd.isna(back) or pd.isna(lay) or back <= 1.0:
            continue
        if 1.0 / back - 1.0 / lay > max_spread:
            continue
        probs.setdefault(str(event_id), {})[str(runner)] = 2.0 / (back + lay)
    return probs
