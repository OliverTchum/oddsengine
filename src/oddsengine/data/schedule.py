"""When the live loop spends credits.

Pure functions of the clock, the schedule, one position and the time of the last
capture of each kind. The archive's file names say when each poll ran.

full poll     every book (uk + eu) for one sport: what decisions are made on.
              Due while a game can still be decided within `horizon`, at most
              once per `decide_every`.
verification  the bet books and games only, for orders verify_after old.
close poll    the panel books and games only, once per game, late in its
              closing window (its last `close_lead`). Every game gets one, bet
              on or not.
results       one call per sport, `settle_after` past a filled order's start,
              at most once per `settle_every`.
trigger       every book (uk + eu) for games whose Betfair Exchange price moved
              at least `betfair.move` since the last tick, before their closing
              window opens. At most `max_triggers` per sport in 24h; the last
              `hot_reserve` of them only for games within `hot_zone` of the start.
"""

from datetime import datetime

from oddsengine.config import AppConfig
from oddsengine.execution.trader import Position
from oddsengine.types import EventInfo


def full_poll_due(
    events: list[EventInfo], *, last: datetime | None, now: datetime, config: AppConfig
) -> bool:
    # skips if the last full poll was too recent
    if last is not None and now - last < config.live.decide_every:
        return False

    # needs a game that can still be decided and is within the horizon
    for event in events:
        if now + config.paper.verify_after < event.commence_utc <= now + config.live.horizon:
            return True
    return False


def capture_window(now: datetime, *, config: AppConfig) -> tuple[datetime, datetime]:
    # earliest and latest start times of games to capture Betfair prices for
    return now + config.paper.verify_after, now + config.betfair.horizon


def verify_due(position: Position, *, now: datetime, config: AppConfig) -> bool:
    # checks if a pending order is old enough to verify

    # skips orders already filled or missed
    if position.fill is not None or position.miss is not None:
        return False

    # skips orders decided too recently
    if now < position.decision.decided_at + config.paper.verify_after:
        return False
    return True


def close_poll_due(
    event: EventInfo, *, last_close: datetime | None, now: datetime, config: AppConfig
) -> bool:
    # checks if the game is in the last stretch of its closing window
    start = event.commence_utc
    shuts = start - config.paper.close_until
    opens = max(shuts - config.live.close_lead, start - config.paper.close_from)
    if not opens <= now <= shuts:
        return False

    # skips if a close poll already ran in this stretch
    if last_close is not None and last_close >= opens:
        return False
    return True


def close_ready(position: Position, *, now: datetime, config: AppConfig) -> bool:
    # checks if the closing window has shut for a filled order
    if position.fill is None or position.close is not None:
        return False

    # skips while the window is still open
    if now <= position.decision.commence_utc - config.paper.close_until:
        return False
    return True


def settle_due(position: Position, *, now: datetime, config: AppConfig) -> bool:
    # checks if a filled order is ready to settle
    if position.fill is None or position.settle is not None:
        return False

    # skips while the game may still be going
    if now < position.decision.commence_utc + config.live.settle_after:
        return False
    return True


def results_call_due(*, last: datetime | None, now: datetime, config: AppConfig) -> bool:
    # limits results calls, which cost 2 credits

    # always calls if never called for this sport
    if last is None:
        return True

    # skips if called too recently
    if now - last < config.live.settle_every:
        return False
    return True


def moved(
    previous: dict[str, dict[str, float]],
    current: dict[str, dict[str, float]],
    *,
    config: AppConfig,
) -> dict[str, float]:
    # finds games where a runner's probability moved enough, with the largest move
    moves = {}
    for event_id, probs in current.items():
        before = previous.get(event_id, {})
        shifts = [abs(p - before[runner]) for runner, p in probs.items() if runner in before]
        if shifts and max(shifts) >= config.betfair.move:
            moves[event_id] = max(shifts)
    return moves


def trigger_due(
    event: EventInfo,
    *,
    last_seen: datetime | None,
    since: datetime,
    now: datetime,
    config: AppConfig,
) -> bool:
    # skips games too close to the start
    lead = max(config.paper.verify_after, config.paper.close_from)
    if now + lead >= event.commence_utc:
        return False

    # skips games whose books were already captured after the move began
    if last_seen is not None and last_seen > since:
        return False
    return True


def trigger_games(
    due: list[EventInfo], *, used: int, now: datetime, config: AppConfig
) -> list[EventInfo]:
    # picks which moved games another trigger poll may cover
    cfg = config.betfair
    if used >= cfg.max_triggers:
        return []
    if used < cfg.max_triggers - cfg.hot_reserve:
        return due
    return [event for event in due if event.commence_utc - now <= cfg.hot_zone]
