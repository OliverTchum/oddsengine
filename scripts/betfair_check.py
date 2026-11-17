"""Smoke test for the Betfair Exchange feed, run by hand once the Betfair keys are in .env.

  uv run python scripts/betfair_check.py

Logs in, then prints each sport's market types, how its markets match the Odds API's
games, and one market's best offers.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from oddsengine.config import AppConfig, Secrets, load_config
from oddsengine.data import betfair
from oddsengine.data.odds_api import OddsApiError, fetch_events

ROOT = Path(__file__).resolve().parents[1]
SPORTS = ("basketball_nba", "icehockey_nhl")


def main() -> int:
    # logs in, checks each sport and returns the exit code
    config = load_config(ROOT / "configs" / "default.yaml")
    secrets = Secrets()
    login = secrets.betfair_login()
    if login is None:
        print("BETFAIR_APP_KEY, BETFAIR_USERNAME and BETFAIR_PASSWORD must all be in .env")
        return 1
    app_key, username, password = login
    now = datetime.now(UTC).replace(microsecond=0)

    # logs in fresh, ignoring the cache
    try:
        session = betfair.connect(
            app_key=app_key,
            username=username,
            password=password,
            cache=ROOT / config.betfair.session,
            now=now,
            keep_alive_every=config.betfair.keep_alive,
            write=True,
            fresh=True,
        )
    except betfair.BetfairError as exc:
        print(f"login FAILED: {exc}")
        return 1
    print(f"logged in at {now:%Y-%m-%dT%H:%M:%SZ}, session cached in {config.betfair.session}")

    # checks each sport, showing the book of the first matched market
    shown = False
    for sport in SPORTS:
        try:
            shown |= check_sport(
                session, sport, config, secrets.odds_api_key.get_secret_value(), now, show=not shown
            )
        except (betfair.BetfairError, OddsApiError) as exc:
            print(f"{sport}: FAILED {exc}", file=sys.stderr)
            return 1
    return 0


def check_sport(
    session: betfair.Session,
    sport: str,
    config: AppConfig,
    odds_api_key: str,
    now: datetime,
    *,
    show: bool,
) -> bool:
    # prints one sport's market types and matches, and returns whether a book was shown
    market = config.betfair.markets[sport]
    tolerance = config.betfair.start_tolerance
    until = now + timedelta(days=1)
    print(f"\n{sport}: event type {market.event_type}, configured market {market.market_type}")

    # lists every market type for this sport
    found = betfair.call(
        session, "listMarketTypes", {"filter": {"eventTypeIds": [market.event_type]}}
    )
    print("  market types: " + ", ".join(f"{t['marketType']} {t['marketCount']}" for t in found))

    # matches the configured markets to the Odds API's games
    games = [
        g
        for g in fetch_events(api_key=odds_api_key, sport=sport)[0]
        if now < g.commence_utc <= until
    ]
    catalogue = betfair.list_markets(
        session, market, start_from=now - tolerance, start_to=until + tolerance
    )
    matched, unmatched = betfair.map_markets(
        catalogue,
        games,
        two_way=sport in config.live.no_draws,
        tolerance=tolerance,
        aliases=config.betfair.aliases,
    )
    print(
        f"  {len(catalogue)} markets, {len(matched)} matched, {len(unmatched)} unmatched;"
        f" {len(games)} Odds API games in the next 24h"
    )

    # prints each match and anything left over
    by_id = {str(e["marketId"]): e for e in catalogue}
    for m in matched:
        entry = by_id[m.market_id]
        sides = ", ".join(
            f"{r['runnerName']} -> {m.runners.get(int(r['selectionId']), '?')}"
            for r in entry["runners"]
        )
        print(f"  matched    {m.market_id} {entry['marketStartTime']} {sides}  [{m.event_id}]")
    for item in unmatched:
        runners = " / ".join(str(r["runnerName"]) for r in item.get("runners", []))
        name = item.get("event", {}).get("name", "")
        print(f"  unmatched  {item['marketId']} {item['marketStartTime']} {name}: {runners}")
    mapped = {m.event_id for m in matched}
    for g in games:
        if g.event_id not in mapped:
            when = f"{g.commence_utc:%Y-%m-%dT%H:%MZ}"
            print(f"  no market  {when} {g.away_team} @ {g.home_team}  [{g.event_id}]")
    if not show or not matched:
        return False

    # prints one market's best offers and the probabilities taken from them
    books = betfair.list_books(session, [matched[0].market_id])
    if books:
        head = books[0]
        print(
            f"\n  book {head.get('marketId')}: status {head.get('status')},"
            f" in play {head.get('inplay')}, delayed {head.get('isMarketDataDelayed')}"
        )
    frame = pd.DataFrame(
        betfair.book_rows(books, matched[:1], captured_at=now), columns=list(betfair.BOOK_COLUMNS)
    )
    for row in frame.to_dict("records"):
        levels = []
        for side in ("back", "lay"):
            offers = [
                f"{row[f'{side}_price_{i}']} ({row[f'{side}_size_{i}']})"
                for i in range(1, betfair.LEVELS + 1)
            ]
            levels.append(f"{side} " + " ".join(offers))
        print(f"  {row['runner']}: " + " | ".join(levels))
    probs = betfair.exchange_probs(frame, max_spread=config.betfair.max_spread)
    print(f"  mid probabilities: {probs}")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
