"""Finds stale prices: venue prices a model's fair probability says are too generous.

Books absent from venues are ignored. Ties break on (market_id, book, selection), so
the output does not depend on input order.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict

from oddsengine.config import ConsensusConfig, VenueConfig
from oddsengine.models.base import FairProbSource, Pooled, devig_books
from oddsengine.models.consensus import PanelConsensus
from oddsengine.types import EventInfo, QuoteSnapshot


# one bet worth considering
class Pick(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    market_id: str
    event_id: str
    selection: str
    book: str
    price: float  # gross decimal price at the venue
    price_net: float  # after commission, before the latency haircut
    p_consensus: float  # the consensus this book was judged against
    n_families: int  # independent reference opinions behind p_consensus
    edge: float


# select's output: picks to stake, and picks flagged or disputed
class Selection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    picks: tuple[Pick, ...] = ()
    flagged: tuple[Pick, ...] = ()  # edge above max_edge
    disputed: tuple[Pick, ...] = ()  # the bet book's own view too far from the consensus


def net_price(price: float, commission: float) -> float:
    # price after commission on net winnings
    return 1 + (price - 1) * (1 - commission)


def fresh(
    snapshot: QuoteSnapshot, *, as_of: datetime, max_age: timedelta, max_skew: timedelta
) -> QuoteSnapshot:
    # a copy of the snapshot holding only the books we can trust right now
    clean_prices = {}
    clean_timestamps = {}

    # every selection any book quotes
    selections = {s for sides in snapshot.prices.values() for s in sides}

    # keeps books with a recent, complete quote; books with no quote time are dropped
    for book, timestamp in snapshot.quoted_at_utc.items():
        latency = as_of - timestamp
        if -max_skew <= latency <= max_age and snapshot.prices[book].keys() >= selections:
            clean_timestamps[book] = timestamp
            clean_prices[book] = snapshot.prices[book]
    return snapshot.model_copy(update={"prices": clean_prices, "quoted_at_utc": clean_timestamps})


def select(
    snapshots: Sequence[QuoteSnapshot],
    *,
    events: dict[str, EventInfo],
    venues: dict[str, VenueConfig],
    cfg: ConsensusConfig,
    ev_min: float,
    as_of: datetime,
    source: FairProbSource | None = None,
) -> Selection:
    # at most one pick per event, with flagged and disputed options kept apart
    # fair probabilities come from source, the panel consensus by default; a snapshot
    # whose event is unknown is skipped
    if source is None:
        source = PanelConsensus(venues, cfg)
    candidates = []
    flagged = []
    disputed = []

    # the books that can be bet into: venue or both
    market = [book for book in venues if venues[book].role != "reference"]

    for snapshot in snapshots:
        event = events.get(snapshot.event_id)
        if event is None:
            continue

        # usable books only, then the source's fair probabilities
        cleaned_snap = fresh(snapshot, as_of=as_of, max_age=cfg.max_age, max_skew=cfg.max_skew)
        consensus_prob = source.fair_probs(cleaned_snap, event)
        if not consensus_prob:
            continue
        best_pick = None

        # every quoting venue, in name order
        for book in sorted(market):
            if book not in cleaned_snap.prices:
                continue
            spread = cleaned_snap.prices[book]

            # a book in both roles is judged against a consensus without its own price
            prob: Pooled | None = consensus_prob
            if venues[book].role == "both":
                prob = source.fair_probs(cleaned_snap, event, exclude=book)
            if not prob:
                continue

            # the bet book's own devigged view
            book_view = devig_books(cleaned_snap, books=[book], method=cfg.devig_method)[book]

            for selection, price in sorted(spread.items()):
                # skip selections below min_prob
                p = prob.probs.get(selection, 0.0)
                if p < cfg.min_prob:
                    continue

                # edge after commission and the haircut
                net_p = net_price(price, venues[book].commission)
                edge = p * net_p * (1 - cfg.latency_haircut) - 1

                # flagged and disputed options are set aside, the best of the rest is kept
                if ev_min <= edge:
                    pick = Pick(
                        market_id=cleaned_snap.market_id,
                        event_id=cleaned_snap.event_id,
                        selection=selection,
                        book=book,
                        price=price,
                        price_net=net_p,
                        p_consensus=p,
                        n_families=prob.n_families,
                        edge=edge,
                    )

                    if cfg.max_edge < edge:
                        flagged.append(pick)
                        continue

                    # skip a book whose own view is too far from the consensus
                    if cfg.max_disagreement < abs(p - book_view[selection]):
                        disputed.append(pick)
                        continue

                    if not best_pick:
                        best_pick = pick
                    if edge > best_pick.edge:
                        best_pick = pick
        if best_pick:
            candidates.append(best_pick)

    # best edge first (ties by market), keep the top max_candidates
    return Selection(
        picks=tuple(
            sorted(candidates, key=lambda pick: (-pick.edge, pick.market_id))[: cfg.max_candidates]
        ),
        flagged=tuple(
            sorted(flagged, key=lambda pick: (pick.market_id, pick.book, pick.selection))
        ),
        disputed=tuple(
            sorted(disputed, key=lambda pick: (pick.market_id, pick.book, pick.selection))
        ),
    )
