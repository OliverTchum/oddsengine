"""One tick of the live paper trader, run every 2.5 minutes by launchd.

  uv run python scripts/tick.py                    # poll what is due, trade, record
  uv run python scripts/tick.py --dry-run          # what would be polled and its cost
  uv run python scripts/tick.py --dry-run --now 2026-11-08T23:00:00Z

Each tick:
1. state   rebuild the trader from the ledger, and read each sport's schedule
2. decide  poll each sport that is due, then trade; check Betfair and poll games that moved
3. verify  re-check pending orders that are due
4. close   close polls and close records for filled orders
5. settle  results for filled orders whose games should be over

--dry-run spends no credits and writes nothing; --now sets the clock for a dry run.
"""

import argparse
import fcntl
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO

import pandas as pd

from oddsengine.config import AppConfig, Secrets, load_config
from oddsengine.data import betfair, schedule
from oddsengine.data.archive import STAMP, captures_since, last_capture, write_snapshot
from oddsengine.data.odds_api import (
    RESULTS_CREDITS,
    OddsApiError,
    credits_per_cycle,
    fetch_events,
    fetch_odds,
    fetch_scores,
    normalise_odds_payload,
    parse_utc,
    results_from_scores,
)
from oddsengine.data.snapshots import snapshots, without_draws
from oddsengine.execution.ledger import Ledger
from oddsengine.execution.trader import (
    Closed,
    Decided,
    Decisions,
    Filled,
    Missed,
    Position,
    Settled,
    TraderState,
    close_payload,
    decide,
    fold,
    settle_payload,
    summarise,
    verify,
)
from oddsengine.models.registry import fair_prob_source
from oddsengine.types import EventInfo, QuoteSnapshot, Result

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data_local" / "tick.lock"
log = logging.getLogger("tick")


# everything one tick needs
@dataclass(frozen=True)
class Run:
    config: AppConfig
    archive: Path
    ledger: Ledger
    api_key: str = field(repr=False)
    now: datetime
    dry_run: bool


def take_lock() -> IO[str] | None:
    # takes the lock so only one tick runs, or returns None if another holds it
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def main(argv: Sequence[str] | None = None) -> int:
    # keeps httpx quiet because it would log the API key in each URL
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="spend no credits, write nothing")
    parser.add_argument("--now", help="pretend it is this UTC time (ISO), for dry runs")
    args = parser.parse_args(argv)

    # runs one tick at a time, except dry runs
    lock = None if args.dry_run else take_lock()
    if lock is None and not args.dry_run:
        log.info("another tick is running: skipped")
        return 0

    # uses one clock for the whole tick
    now = datetime.now(UTC) if args.now is None else parse_utc(args.now)
    config = load_config(args.config)
    # stops early if the fair probability source is unknown
    fair_prob_source(config)
    run = Run(
        config=config,
        archive=ROOT / config.archiver.root,
        ledger=Ledger(ROOT / config.live.ledger),
        api_key=Secrets().odds_api_key.get_secret_value(),
        now=now.replace(microsecond=0),
        dry_run=args.dry_run,
    )
    log.info(f"tick {run.now:%Y-%m-%dT%H:%M:%SZ}" + (" (dry run)" if run.dry_run else ""))

    # 1. state
    state = refold(run)
    events, remaining = schedule_of(run)
    log.info(f"  {len(events)} scheduled games, {remaining} credits left")

    # 2 to 5, each starting from the state the last left
    polled = poll_and_decide(run, events, remaining, state)
    exchange_step(run, events, remaining, polled)
    state = refold(run)
    state = verify_pending(run, events, polled, state)
    state = close_filled(run, events, polled, state)
    settle_finished(run, state)

    log.info("  " + json.dumps(summarise(refold(run)).model_dump(mode="json")))
    return 0


def refold(run: Run) -> TraderState:
    # rebuilds the trader's state from the ledger
    payloads = (entry["payload"] for entry in run.ledger.entries())
    return fold(payloads, bankroll=run.config.paper.bankroll)


def record(run: Run, records: Sequence[Decided | Filled | Missed | Closed | Settled]) -> None:
    # appends records to the ledger, or just logs them in a dry run
    for item in records:
        log.info(f"    {item.kind} {item.order_id}")
        if not run.dry_run:
            run.ledger.append(item.model_dump(mode="json"))


def schedule_of(run: Run) -> tuple[dict[str, EventInfo], int | None]:
    # fetches every live sport's upcoming and in-play games
    events: dict[str, EventInfo] = {}
    remaining = None
    for sport in run.config.live.sports:
        try:
            found, remaining = fetch_events(api_key=run.api_key, sport=sport)
        except OddsApiError as exc:
            log.warning(f"  {sport}: schedule FAILED {exc}")
            continue
        for event in found:
            events[event.event_id] = event
    return events, remaining


def poll(
    run: Run,
    sport: str,
    *,
    tag: str = "",
    books: list[str] | None = None,
    event_ids: list[str] | None = None,
) -> list[QuoteSnapshot]:
    # fetches, archives and returns one capture as market snapshots
    archiver = run.config.archiver
    named = books or []
    cost = credits_per_cycle(
        sports=[sport], regions=archiver.regions, markets=archiver.markets, bookmakers=named
    )
    label = tag or "full"
    if run.dry_run:
        log.info(f"  {sport}: {label} poll would cost {cost}")
        return []
    payload, remaining = fetch_odds(
        api_key=run.api_key,
        sport=sport,
        regions=archiver.regions,
        markets=archiver.markets,
        bookmakers=named,
        event_ids=event_ids or [],
    )
    rows = normalise_odds_payload(payload, captured_at=run.now)
    write_snapshot(run.archive, sport=sport, rows=rows, captured_at=run.now, tag=tag)
    log.info(
        f"  {sport}: {label} poll, {len(payload)} games, cost {cost}, {remaining} credits left"
    )
    return settleable(run, sport, list(snapshots(pd.DataFrame(rows))) if rows else [])


def settleable(run: Run, sport: str, snaps: list[QuoteSnapshot]) -> list[QuoteSnapshot]:
    # drops books with a draw for sports that settle without one
    if sport not in run.config.live.no_draws:
        return snaps
    return [without_draws(s) for s in snaps]


def poll_and_decide(
    run: Run, events: dict[str, EventInfo], remaining: int | None, state: TraderState
) -> dict[str, list[QuoteSnapshot]]:
    # step 2: polls and trades, returning the full poll snapshots by sport
    polled: dict[str, list[QuoteSnapshot]] = {}
    live = run.config.live
    if remaining is not None and remaining < live.credit_reserve:
        log.info(
            f"  {remaining} credits is below the reserve of {live.credit_reserve}: no full polls"
        )
        return polled

    for sport in live.sports:
        games = [e for e in events.values() if e.sport == sport]
        last = last_capture(run.archive, sport=sport)
        if not schedule.full_poll_due(games, last=last, now=run.now, config=run.config):
            continue
        # one sport failing never stops the others
        try:
            snaps = poll(run, sport)
            polled[sport] = snaps

            # records new orders and flags
            decisions = decide(snaps, events=events, state=state, config=run.config, as_of=run.now)
            log_flags(run, decisions)
            record(run, decisions.orders)
        except OddsApiError as exc:
            log.warning(f"  {sport}: full poll FAILED {exc}")
        except Exception:
            log.exception(f"  {sport}: FAILED")

        # refreshes the state so the next sport sees these orders
        state = refold(run)
    return polled


def log_flags(run: Run, decisions: Decisions) -> None:
    # logs flagged and disputed picks and appends them to the flags file
    found = [("flagged", decisions.flagged), ("disputed", decisions.disputed)]
    if not any(picks for _, picks in found):
        return
    path = ROOT / run.config.live.flags
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for kind, picks in found:
            for pick in picks:
                log.info(
                    f"    {kind} {pick.market_id} {pick.book} {pick.selection} edge {pick.edge:.3f}"
                )
                line = {"at": run.now.isoformat(), "kind": kind, **pick.model_dump(mode="json")}
                handle.write(json.dumps(line) + "\n")


def exchange_step(
    run: Run,
    events: dict[str, EventInfo],
    remaining: int | None,
    polled: dict[str, list[QuoteSnapshot]],
) -> None:
    # step 2: checks Betfair for moved prices and polls those games
    session = betfair_session(run)
    if session is None:
        return

    # credits left after this tick's full polls
    archiver = run.config.archiver
    spent = credits_per_cycle(
        sports=list(polled), regions=archiver.regions, markets=archiver.markets
    )
    credits = None if remaining is None else remaining - spent

    for sport in run.config.live.sports:
        try:
            since, moves = exchange_moves(run, session, sport, events)
        except betfair.SessionExpired as exc:
            log.warning(f"  betfair: session rejected ({exc}), logging in next tick")
            if not run.dry_run:
                betfair.forget(ROOT / run.config.betfair.session)
            return
        except Exception as exc:  # never stops the tick
            log.warning(f"  {sport}: betfair FAILED {type(exc).__name__}: {exc}")
            continue
        if not moves or since is None:
            continue

        # one sport failing never stops the others
        try:
            credits = trigger_poll(run, sport, moves, since, events, polled, credits)
        except Exception:
            log.exception(f"  {sport}: trigger FAILED")


def betfair_session(run: Run) -> betfair.Session | None:
    # connects to Betfair, or returns None without credentials or a login
    login = Secrets().betfair_login()
    if login is None:
        log.info("  betfair: no credentials, skipped")
        return None
    app_key, username, password = login
    cfg = run.config.betfair
    try:
        return betfair.connect(
            app_key=app_key,
            username=username,
            password=password,
            cache=ROOT / cfg.session,
            now=run.now,
            keep_alive_every=cfg.keep_alive,
            write=not run.dry_run,
        )
    except betfair.BetfairError as exc:
        log.info(f"  betfair: skipped, {exc}")
    except Exception as exc:  # never stops the tick
        log.warning(f"  betfair: skipped, {type(exc).__name__}")
    return None


def exchange_moves(
    run: Run, session: betfair.Session, sport: str, events: dict[str, EventInfo]
) -> tuple[datetime | None, dict[str, float]]:
    # archives one sport's best offers and finds games whose price moved
    cfg = run.config
    market = cfg.betfair.markets.get(sport)
    soonest, latest = schedule.capture_window(run.now, config=cfg)
    games = [e for e in events.values() if e.sport == sport and soonest < e.commence_utc <= latest]
    if market is None or not games:
        return None, {}

    # matches the sport's markets to its games and fetches their offers
    tolerance = cfg.betfair.start_tolerance
    catalogue = betfair.list_markets(
        session, market, start_from=soonest - tolerance, start_to=latest + tolerance
    )
    matched, unmatched = betfair.map_markets(
        catalogue,
        games,
        two_way=sport in cfg.live.no_draws,
        tolerance=tolerance,
        aliases=cfg.betfair.aliases,
    )
    rows = betfair.book_rows(
        betfair.list_books(session, [m.market_id for m in matched]), matched, captured_at=run.now
    )
    log.info(
        f"  {sport}: betfair {len(matched)} of {len(games)} games mapped,"
        f" {len(unmatched)} markets unmatched"
    )

    # compares with the previous capture before writing this one
    frame = pd.DataFrame(rows, columns=list(betfair.BOOK_COLUMNS))
    current = betfair.exchange_probs(frame, max_spread=cfg.betfair.max_spread)
    since, previous = previous_capture(run, sport)
    moves = schedule.moved(previous, current, config=cfg)
    if not run.dry_run:
        write_snapshot(
            ROOT / cfg.betfair.root,
            sport=sport,
            rows=rows,
            captured_at=run.now,
            columns=betfair.BOOK_COLUMNS,
        )
    return since, moves


def previous_capture(run: Run, sport: str) -> tuple[datetime | None, dict[str, dict[str, float]]]:
    # finds the newest Betfair capture that is old enough to compare with, but not too old
    cfg = run.config.betfair
    recent = captures_since(ROOT / cfg.root, sport=sport, since=run.now - cfg.compare_within)
    older = [(stamp, path) for stamp, _, path in recent if stamp <= run.now - cfg.compare_back]
    if not older:
        return None, {}
    stamp, path = older[-1]
    return stamp, betfair.exchange_probs(pd.read_parquet(path), max_spread=cfg.max_spread)


def trigger_poll(
    run: Run,
    sport: str,
    moves: dict[str, float],
    since: datetime,
    events: dict[str, EventInfo],
    polled: dict[str, list[QuoteSnapshot]],
    credits: int | None,
) -> int | None:
    # polls the moved games, then flags and trades as after a full poll
    cfg = run.config
    archiver = cfg.archiver
    cost = credits_per_cycle(sports=[sport], regions=archiver.regions, markets=archiver.markets)

    # finds when each game was last captured since the move began
    seen: dict[str, datetime] = {}
    for stamp, _, path in captures_since(run.archive, sport=sport, since=since):
        for event_id in pd.read_parquet(path, columns=["event_id"])["event_id"].unique():
            key = str(event_id)
            seen[key] = max(seen.get(key, stamp), stamp)

    # logs every move and keeps the games that can be polled
    due = []
    for event_id, move in sorted(moves.items()):
        event = events[event_id]
        last_seen = seen.get(event_id)
        ok = schedule.trigger_due(event, last_seen=last_seen, since=since, now=run.now, config=cfg)
        note = "" if ok else ", not polled: books captured since the move began, or near its start"
        log.info(f"    trigger {event.away_team} @ {event.home_team} moved {move:.3f}{note}")
        if ok:
            due.append(event_id)
    if not due:
        return credits

    # works out why a poll may be skipped
    recent = captures_since(run.archive, sport=sport, since=run.now - timedelta(days=1))
    used = sum(1 for _, tag, _ in recent if tag == "trigger")
    allowed = schedule.trigger_games(
        [events[event_id] for event_id in due], used=used, now=run.now, config=cfg
    )
    reason = ""
    if sport in polled:
        reason = "a full poll ran this tick"
    elif credits is not None and credits < cfg.live.credit_reserve:
        reason = f"{credits} credits is below the reserve of {cfg.live.credit_reserve}"
    elif used >= cfg.betfair.max_triggers:
        reason = f"{used} trigger polls in the last 24h"
    elif not allowed:
        hours = cfg.betfair.hot_zone.total_seconds() / 3600
        reason = f"{used} used in 24h, the rest kept for games within {hours:g}h of the start"
    if reason:
        log.info(f"  {sport}: trigger poll skipped, {reason}")
        return credits

    try:
        snaps = poll(run, sport, tag="trigger", event_ids=[event.event_id for event in allowed])
    except OddsApiError as exc:
        log.warning(f"  {sport}: trigger poll FAILED {exc}")
        return credits
    decisions = decide(snaps, events=events, state=refold(run), config=cfg, as_of=run.now)
    log_flags(run, decisions)
    record(run, decisions.orders)
    return None if credits is None else credits - cost


def verify_pending(
    run: Run,
    events: dict[str, EventInfo],
    polled: dict[str, list[QuoteSnapshot]],
    state: TraderState,
) -> TraderState:
    # step 3: verifies pending orders that are due
    due: dict[str, list[Position]] = {}
    for position in state.positions.values():
        if schedule.verify_due(position, now=run.now, config=run.config):
            event = events.get(position.decision.event_id)
            sport = event.sport if event else ""  # empty once the game has left the schedule
            due.setdefault(sport, []).append(position)

    for sport, positions in sorted(due.items()):
        # only orders before the start and not overdue need a poll
        snaps = polled.get(sport, [])
        paper = run.config.paper
        checkable = [
            p
            for p in positions
            if run.now < p.decision.commence_utc
            and run.now <= p.decision.decided_at + paper.verify_after + paper.verify_within
        ]
        if sport and sport not in polled and checkable:
            books = sorted({p.decision.book for p in checkable})
            ids = sorted({p.decision.event_id for p in checkable})
            try:
                snaps = poll(run, sport, tag="verify", books=books, event_ids=ids)
            except OddsApiError as exc:
                log.warning(f"  {sport}: verification poll FAILED {exc}")
                continue
        results = [verify(p, snaps, config=run.config, as_of=run.now) for p in positions]
        record(run, [r for r in results if r is not None])
    return refold(run)


def close_filled(
    run: Run,
    events: dict[str, EventInfo],
    polled: dict[str, list[QuoteSnapshot]],
    state: TraderState,
) -> TraderState:
    # step 4a: polls the panel books for games late in their closing window
    panel = sorted(b for b, v in run.config.venues.items() if v.role != "venue")
    due: dict[str, set[str]] = {}
    for event in events.values():
        if event.sport in polled:
            continue
        last = last_capture(run.archive, sport=event.sport, tag="close")
        if schedule.close_poll_due(event, last_close=last, now=run.now, config=run.config):
            due.setdefault(event.sport, set()).add(event.event_id)
    for sport, ids in sorted(due.items()):
        try:
            poll(run, sport, tag="close", books=panel, event_ids=sorted(ids))
        except OddsApiError as exc:
            log.warning(f"  {sport}: close poll FAILED {exc}")

    # step 4b: records the close for filled orders whose window has shut
    closes = []
    for position in state.positions.values():
        if schedule.close_ready(position, now=run.now, config=run.config):
            closes.append(
                close_payload(position, market_snapshots(run, position), config=run.config)
            )
    record(run, closes)
    return refold(run)


def market_snapshots(run: Run, position: Position) -> list[QuoteSnapshot]:
    # collects the archived snapshots of one order's market
    order = position.decision
    files: list[Path] = []
    for day in (order.commence_utc - timedelta(days=1), order.commence_utc):
        files += sorted(run.archive.glob(f"sport=*/date={day:%Y-%m-%d}/*.parquet"))

    # reads this game's rows and its sport from the folder name
    frames = []
    sport = ""
    for path in files:
        frame = pd.read_parquet(path)
        frame = frame[frame["event_id"] == order.event_id]
        if not frame.empty:
            frames.append(frame)
            sport = path.parent.parent.name.removeprefix("sport=")
    if not frames:
        return []
    found = snapshots(pd.concat(frames, ignore_index=True))
    return settleable(run, sport, [s for s in found if s.market_id == order.market_id])


def settle_finished(run: Run, state: TraderState) -> None:
    # step 5: settles filled orders, using saved results before a new call
    due: dict[str, list[Position]] = {}
    for position in state.positions.values():
        if schedule.settle_due(position, now=run.now, config=run.config):
            sport = sport_of(run, position)
            if sport is not None:
                due.setdefault(sport, []).append(position)

    for sport, positions in sorted(due.items()):
        known = results_for(run, sport, {p.decision.event_id for p in positions})
        settled = []
        for position in positions:
            result = known.get(position.decision.event_id)
            if result is not None:
                settled.append(settle_payload(position, result))
        record(run, settled)


def sport_of(run: Run, position: Position) -> str | None:
    # finds the sport whose archive holds this order's decision poll
    order = position.decision
    for sport in run.config.live.sports:
        for path in run.archive.glob(f"sport={sport}/date={order.decided_at:%Y-%m-%d}/*.parquet"):
            if order.event_id in set(pd.read_parquet(path, columns=["event_id"])["event_id"]):
                return sport
    return None


def results_for(run: Run, sport: str, needed: set[str]) -> dict[str, Result]:
    # loads saved results for a sport, calling for new ones only if some are missing
    folder = ROOT / run.config.live.results / f"sport={sport}"
    known: dict[str, Result] = {}
    last = None
    for path in sorted(folder.glob("*.json")):
        known.update(results_from_scores(json.loads(path.read_text())))
        last = datetime.strptime(path.stem, STAMP).replace(tzinfo=UTC)
    if needed.issubset(known) or not schedule.results_call_due(
        last=last, now=run.now, config=run.config
    ):
        return known

    if run.dry_run:
        log.info(f"  {sport}: results call would cost {RESULTS_CREDITS}")
        return known
    try:
        payload, remaining = fetch_scores(api_key=run.api_key, sport=sport)
    except OddsApiError as exc:
        log.warning(f"  {sport}: results call FAILED {exc}")
        return known
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run.now.strftime(STAMP)}.json").write_text(json.dumps(payload))
    log.info(f"  {sport}: results call, cost {RESULTS_CREDITS}, {remaining} credits left")
    known.update(results_from_scores(payload))
    return known


if __name__ == "__main__":
    raise SystemExit(main())
