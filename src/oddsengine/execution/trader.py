"""Paper trader: decide, verify the fill, take the closing line, settle.

Every step writes one record to the hash-chained ledger, and the state is
rebuilt from those records. Stakes and P&L are Decimals in units of the
starting bankroll.
"""

import hashlib
from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Any, Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from oddsengine.config import AppConfig
from oddsengine.execution.clv import clv_econ, clv_prob
from oddsengine.execution.ledger import canonical_json
from oddsengine.models.base import devig_books
from oddsengine.models.consensus import consensus
from oddsengine.models.registry import fair_prob_source
from oddsengine.portfolio.pipeline import MIN_STAKE, build_slate
from oddsengine.selection.select import Pick, fresh, net_price, select
from oddsengine.types import Candidate, EventInfo, QuoteSnapshot, Result

# used for rounding decimal values
STAKE_STEP = Decimal("0.000001")


# base class for ledger records: immutable, unknown fields rejected
class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# a pending order
class Decided(_Record):
    kind: Literal["decision"] = "decision"
    order_id: str  # deterministic and unique; includes decided_at, so a retry gets a new id
    event_id: str
    market_id: str
    selection: str
    book: str
    price: float  # gross price seen at the decision
    price_net: float  # after the book's commission
    p_consensus: float
    n_families: int
    edge: float
    latency_haircut: float
    stake: Decimal = Field(gt=0)  # fraction of the starting bankroll
    decided_at: datetime
    commence_utc: datetime
    strategy: str = "consensus"  # the model that priced it
    config_digest: str
    snapshot_digest: str  # of the snapshot the pick came from

    # checks the order is decided before the game starts
    @model_validator(mode="after")
    def _before_the_start(self) -> Self:
        if self.decided_at >= self.commence_utc:
            raise ValueError(f"order {self.order_id} decided at or after its start")
        return self


# an order whose price held at the verification snapshot
class Filled(_Record):
    kind: Literal["fill"] = "fill"
    order_id: str
    observed_at: datetime  # capture time of the verification snapshot
    price: float  # gross price then showing
    price_net: float  # after the book's commission


# an order whose price did not hold, or was never checked
class Missed(_Record):
    kind: Literal["miss"] = "miss"
    order_id: str
    observed_at: datetime  # verification snapshot's capture time; as_of for "started" and "late"
    # slipped is no longer produced, kept so older records load
    reason: Literal["slipped", "no_edge", "no_quote", "started", "late"]
    price: float | None = None  # the price then showing, for "slipped" and "no_edge"


# the closing line, and CLV against it
class Closed(_Record):
    kind: Literal["close"] = "close"
    order_id: str
    status: Literal["ok", "missing", "degenerate"]
    observed_at: datetime | None = None  # capture time of the closing snapshot
    p_close: float | None = None
    clv_econ: float | None = None  # only for "ok"
    clv_prob: float | None = None


# the result, and the P&L it gives
class Settled(_Record):
    kind: Literal["settle"] = "settle"
    order_id: str
    winner: str
    won: bool
    pnl: Decimal


# maps each ledger kind to its record class
RECORDS: dict[str, type[Decided] | type[Filled] | type[Missed] | type[Closed] | type[Settled]] = {
    "decision": Decided,
    "fill": Filled,
    "miss": Missed,
    "close": Closed,
    "settle": Settled,
}


# one order's life so far
class Position(_Record):
    decision: Decided
    fill: Filled | None = None
    miss: Missed | None = None
    close: Closed | None = None
    settle: Settled | None = None


# everything the ledger says, folded
class TraderState(_Record):
    bankroll: Decimal  # starting bankroll plus settled P&L
    positions: dict[str, Position] = Field(default_factory=dict)  # by order_id
    reserved: Decimal = Decimal(0)  # stakes of pending orders
    at_risk: Decimal = Decimal(0)  # stakes of filled orders not yet settled


# what summarise reports
class Summary(_Record):
    decisions: int
    pending: int
    filled: int
    missed: int
    fill_rate: float | None  # share of resolved orders that filled; None until one resolves
    shortfall: float | None  # mean change from decision price to fill price; None without a fill
    settled: int
    pnl: Decimal
    closes: dict[str, int]  # count per close status, every status present
    clv_econ: float | None  # mean over "ok" closes; None without one
    clv_prob: float | None


def digest(model: BaseModel) -> str:
    # SHA-256 of the model's canonical JSON
    return hashlib.sha256(canonical_json(model.model_dump(mode="json")).encode()).hexdigest()


def decidable(
    snapshot: QuoteSnapshot, events: dict[str, EventInfo], *, as_of: datetime, config: AppConfig
) -> bool:
    # ensures that the snapshot's game is scheduled, and that any orders on it
    # can be verified before the game starts, and that it starts within this "game day"
    event = events.get(snapshot.event_id)
    if event is None:
        return False
    lead = event.commence_utc - as_of
    return config.paper.verify_after < lead <= config.decision.max_lead


# what decide hands back: the new orders, and the picks too implausible to bet
class Decisions(NamedTuple):
    orders: list[Decided]
    flagged: tuple[Pick, ...] = ()
    disputed: tuple[Pick, ...] = ()


def open_book(state: TraderState) -> tuple[set[str], dict[str, Decimal]]:
    # keeps track of open orders, and each books exposure
    taken = set()
    exposure: dict[str, Decimal] = {}
    for position in state.positions.values():
        if position.miss is None:
            taken.add(position.decision.event_id)
            if position.settle is None:
                book = position.decision.book
                exposure[book] = exposure.get(book, Decimal(0)) + position.decision.stake
    return taken, exposure


def candidates_for(
    picks: Sequence[Pick], events: dict[str, EventInfo], config: AppConfig
) -> list[Candidate]:
    # turns each pick into a bet for the portfolio stage, grouped by sport and start date
    candidates = []
    for pick in picks:
        event = events[pick.event_id]
        candidates.append(
            Candidate(
                market_id=pick.market_id,
                event_id=pick.event_id,
                group=f"{event.sport}:{event.commence_utc.date()}",
                book=pick.book,
                selection=pick.selection,
                fair_prob=pick.p_consensus,
                quoted_price=pick.price_net * (1 - config.consensus.latency_haircut),
            )
        )
    return candidates


def orders_for(
    picks: Sequence[Pick],
    snaps: dict[str, QuoteSnapshot],
    *,
    events: dict[str, EventInfo],
    state: TraderState,
    exposure: dict[str, Decimal],
    config: AppConfig,
    as_of: datetime,
) -> list[Decided]:
    # stakes the picks in one slate and turn them into pending orders

    # risk budget
    budget = config.risk.max_total_stake - float(state.reserved + state.at_risk)
    if budget <= MIN_STAKE or not picks:
        return []

    # a copy of the config with the smaller budget, for the optimiser only
    budget_risk = config.risk.model_copy(update={"max_total_stake": budget})
    budget_config = config.model_copy(update={"risk": budget_risk})

    # how much more each book may take before its cap
    room = {}
    for pick in picks:
        cap = config.venues[pick.book].max_stake
        room[pick.book] = cap - float(exposure.get(pick.book, Decimal(0)))

    slate = build_slate(
        candidates_for(picks, events, config), config=budget_config, as_of=as_of, room=room
    )

    # creates digest for config that was in place for this slate
    config_digest = digest(config)
    # joins the optimiser's output back to the full pick
    pick_by_market = {pick.market_id: pick for pick in picks}
    orders = []
    for prop in slate.proposals:
        # rounds down stakes to a millionth of bankroll so no cap is broken
        stake = Decimal(prop.stake).quantize(STAKE_STEP, rounding=ROUND_DOWN)
        if stake <= 0:
            continue

        pick = pick_by_market[prop.market_id]
        orders.append(
            Decided(
                order_id=f"{pick.market_id}|{pick.book}|{pick.selection}|{as_of.isoformat()}",
                event_id=pick.event_id,
                market_id=pick.market_id,
                selection=pick.selection,
                book=pick.book,
                price=pick.price,
                price_net=pick.price_net,
                p_consensus=pick.p_consensus,
                n_families=pick.n_families,
                edge=pick.edge,
                latency_haircut=config.consensus.latency_haircut,
                stake=stake,
                decided_at=as_of,
                commence_utc=events[pick.event_id].commence_utc,
                strategy=config.decision.fair_prob,
                config_digest=config_digest,
                snapshot_digest=digest(snaps[pick.market_id]),
            )
        )
    return orders


def decide(
    snapshots: Sequence[QuoteSnapshot],
    *,
    events: dict[str, EventInfo],
    state: TraderState,
    config: AppConfig,
    as_of: datetime,
) -> Decisions:
    # searches the tick's snapshots for stale prices then returns them as
    # orders alongside the flagged and disputed picks
    taken, exposure = open_book(state)

    # dict for mapping picks to their snapshot via market_id
    snaps = {}
    for snapshot in snapshots:
        if snapshot.event_id not in taken and decidable(
            snapshot, events, as_of=as_of, config=config
        ):
            snaps[snapshot.market_id] = snapshot

    # the picks, priced by the configured fair probability source
    selection = select(
        list(snaps.values()),
        events=events,
        venues=config.venues,
        cfg=config.consensus,
        ev_min=config.decision.ev_min,
        as_of=as_of,
        source=fair_prob_source(config),
    )
    orders = orders_for(
        selection.picks,
        snaps,
        events=events,
        state=state,
        exposure=exposure,
        config=config,
        as_of=as_of,
    )
    return Decisions(orders=orders, flagged=selection.flagged, disputed=selection.disputed)


def verify(
    position: Position, snapshots: Sequence[QuoteSnapshot], *, config: AppConfig, as_of: datetime
) -> Filled | Missed | None:
    # fill or miss a pending order, or None while it waits for its verification snapshot
    order = position.decision
    if position.fill is not None or position.miss is not None:
        raise ValueError(f"order {order.order_id} is not pending")

    # the earliest snapshot of this market that is in the window
    # and taken before the start
    opens = order.decided_at + config.paper.verify_after
    shuts = opens + config.paper.verify_within
    check: QuoteSnapshot | None = None
    for snapshot in snapshots:
        seen = snapshot.observed_at_utc
        if snapshot.market_id != order.market_id:
            continue
        if seen < opens or seen > shuts or seen > as_of or seen >= order.commence_utc:
            continue
        if check is None or seen < check.observed_at_utc:
            check = snapshot

    # checks if the game has started or the verify window has passed
    if check is None:
        if as_of >= order.commence_utc:
            return Missed(order_id=order.order_id, observed_at=as_of, reason="started")
        if as_of > shuts:
            return Missed(order_id=order.order_id, observed_at=as_of, reason="late")
        return None

    # checks that the matching order actually exists in the snapshot
    quotes = check.prices.get(order.book, {})
    if order.selection not in quotes:
        return Missed(order_id=order.order_id, observed_at=check.observed_at_utc, reason="no_quote")
    price = quotes[order.selection]

    # checks if the order still has an edge on the market
    price_net = net_price(price, config.venues[order.book].commission)
    if order.p_consensus * price_net - 1 < config.decision.ev_min:
        return Missed(
            order_id=order.order_id,
            observed_at=check.observed_at_utc,
            reason="no_edge",
            price=price,
        )
    # counts a fill if edge persists
    return Filled(
        order_id=order.order_id,
        observed_at=check.observed_at_utc,
        price=price,
        price_net=price_net,
    )


def close_payload(
    position: Position, snapshots: Sequence[QuoteSnapshot], *, config: AppConfig
) -> Closed:
    # a filled order's closing line, and CLV against it when the close is valid
    order, fill = position.decision, position.fill
    if fill is None:
        raise ValueError(f"order {order.order_id} never filled, so it has no close")

    # this market's snapshots in the closing window, latest first
    opens = order.commence_utc - config.paper.close_from
    shuts = order.commence_utc - config.paper.close_until
    window = [
        s
        for s in snapshots
        if s.market_id == order.market_id and opens <= s.observed_at_utc <= shuts
    ]
    window.sort(key=lambda s: s.observed_at_utc, reverse=True)
    if not window:
        return Closed(order_id=order.order_id, status="missing")

    # don't count closes too near to the decision
    latest = window[0].observed_at_utc
    if latest - order.decided_at < config.paper.min_close_gap:
        return Closed(order_id=order.order_id, status="degenerate", observed_at=latest)

    # walks back from the latest snapshot until one gives a panel consensus,
    # leaving out the bet book, the same way decide built it
    cfg = config.consensus
    panel_members = []
    for book in config.venues:
        if config.venues[book].role != "venue":
            panel_members.append(book)
    for close in window:
        seen = close.observed_at_utc
        if seen - order.decided_at < config.paper.min_close_gap:
            break
        cleaned = fresh(close, as_of=seen, max_age=cfg.max_age, max_skew=cfg.max_skew)
        fair = devig_books(cleaned, books=panel_members, method=cfg.devig_method)
        pooled = consensus(
            fair, venues=config.venues, min_families=cfg.min_reference_families, exclude=order.book
        )
        if pooled is None or order.selection not in pooled.probs:
            continue

        # CLV at the net price taken, and against the decision's own consensus
        p_close = pooled.probs[order.selection]
        return Closed(
            order_id=order.order_id,
            status="ok",
            observed_at=seen,
            p_close=p_close,
            clv_econ=clv_econ(fill.price_net, p_close),
            clv_prob=clv_prob(order.p_consensus, p_close),
        )
    return Closed(order_id=order.order_id, status="missing", observed_at=latest)


def settle_payload(position: Position, result: Result) -> Settled:
    # settles a filled order on its event's result
    order, fill = position.decision, position.fill
    if fill is None:
        raise ValueError(f"order {order.order_id} never filled, so there is nothing to settle")
    if result.event_id != order.event_id:
        raise ValueError(f"a result for {result.event_id} cannot settle order {order.order_id}")

    # only the picked selection wins, a draw loses
    won = order.selection == result.winner
    winnings = (order.stake * (Decimal(str(fill.price_net)) - 1)).quantize(STAKE_STEP)
    pnl = winnings if won else -order.stake
    return Settled(order_id=order.order_id, winner=result.winner, won=won, pnl=pnl)


def fold(payloads: Iterable[dict[str, Any]], *, bankroll: float | Decimal) -> TraderState:
    # replays ledger payloads in order, raising on any record that does not fit
    positions: dict[str, Position] = {}
    pnl = Decimal(0)
    for payload in payloads:
        record = RECORDS[payload["kind"]].model_validate(payload)

        # a decision opens a position; every other record must find one
        if isinstance(record, Decided):
            if record.order_id in positions:
                raise ValueError(f"order {record.order_id} decided twice")
            positions[record.order_id] = Position(decision=record)
            continue
        position = positions.get(record.order_id)
        if position is None:
            raise ValueError(f"{record.kind} for unknown order {record.order_id}")

        # a fill or a miss resolves a pending order, once
        if isinstance(record, Filled | Missed):
            if position.fill is not None or position.miss is not None:
                raise ValueError(f"order {record.order_id} resolved twice")
            field = "fill" if isinstance(record, Filled) else "miss"

        # a close or a settle needs a fill, and each comes at most once
        else:
            field = "close" if isinstance(record, Closed) else "settle"
            earlier = position.close if isinstance(record, Closed) else position.settle
            if position.fill is None or earlier is not None:
                raise ValueError(
                    f"{record.kind} on order {record.order_id}, which is not open for it"
                )
            if isinstance(record, Settled):
                pnl += record.pnl
        positions[record.order_id] = position.model_copy(update={field: record})

    # pending stakes are reserved; filled ones not yet settled are at risk
    reserved = Decimal(0)
    at_risk = Decimal(0)
    for position in positions.values():
        if position.fill is None and position.miss is None:
            reserved += position.decision.stake
        elif position.fill is not None and position.settle is None:
            at_risk += position.decision.stake
    start = Decimal(str(bankroll))
    return TraderState(
        bankroll=start + pnl, positions=positions, reserved=reserved, at_risk=at_risk
    )


def summarise(state: TraderState) -> Summary:
    # counts per stage, P&L so far, and the mean CLV
    pending = filled = missed = settled = 0
    pnl = Decimal(0)
    closes = {"ok": 0, "missing": 0, "degenerate": 0}
    econ: list[float] = []
    prob: list[float] = []
    shortfalls: list[float] = []
    for position in state.positions.values():
        if position.fill is not None:
            filled += 1
            shortfalls.append(position.fill.price / position.decision.price - 1)
        elif position.miss is not None:
            missed += 1
        else:
            pending += 1
        if position.settle is not None:
            settled += 1
            pnl += position.settle.pnl
        close = position.close
        if close is not None:
            closes[close.status] += 1
            if close.clv_econ is not None and close.clv_prob is not None:
                econ.append(close.clv_econ)
                prob.append(close.clv_prob)

    # rates only once there is something to divide by
    resolved = filled + missed
    return Summary(
        decisions=len(state.positions),
        pending=pending,
        filled=filled,
        missed=missed,
        fill_rate=filled / resolved if resolved else None,
        shortfall=sum(shortfalls) / len(shortfalls) if shortfalls else None,
        settled=settled,
        pnl=pnl,
        closes=closes,
        clv_econ=sum(econ) / len(econ) if econ else None,
        clv_prob=sum(prob) / len(prob) if prob else None,
    )
