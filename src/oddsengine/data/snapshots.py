"""Turns rows of odds, from a live poll or the Parquet archive, into market snapshots.

Each row is one book's price for one outcome at one capture. A snapshot groups every
book's prices for one market of one event from one capture.
"""

from collections.abc import Iterator
from datetime import datetime

import pandas as pd

from oddsengine.types import QuoteSnapshot


def _line(rows: pd.DataFrame, market_key: str) -> pd.Series:
    # the line each row is quoted at, from the home side's view for spreads
    if "outcome_point" not in rows:
        return pd.Series(float("nan"), index=rows.index)
    point = rows["outcome_point"].astype(float)
    if market_key == "spreads":
        return point.where(rows["outcome_name"] == rows["home_team"], -point)
    return point


def _quote_times(group: pd.DataFrame) -> pd.Series:
    # the market's own update time, else the book's
    empty = pd.Series(pd.NaT, index=group.index, dtype=object)
    market = group.get("market_last_update_utc", empty)
    book = group.get("bookmaker_last_update_utc", empty)
    return market.where(market.notna(), book)


def snapshots(frame: pd.DataFrame, *, market_key: str = "h2h") -> Iterator[QuoteSnapshot]:
    # turns the rows into snapshots, one per line for totals and spreads
    rows = frame[frame["market_key"] == market_key]
    rows = rows.assign(line=_line(rows, market_key))

    # groups by capture time first so snapshots come out in time order
    groups = rows.groupby(["captured_at_utc", "event_id", "line"], sort=True, dropna=False)
    for (_, event_id, _), group in groups:
        # prices by book then selection, skipping prices of 1.0 or less
        prices: dict[str, dict[str, float]] = {}
        for book, selection, price in zip(
            group["bookmaker_key"], group["outcome_name"], group["price_decimal"], strict=True
        ):
            if price > 1.0:
                prices.setdefault(str(book), {})[str(selection)] = float(price)
        if not prices:
            continue

        # when each book's price was live, left out if unknown so staleness shows
        quoted: dict[str, datetime] = {}
        for book, stamp in zip(group["bookmaker_key"], _quote_times(group), strict=True):
            if str(book) in prices and pd.notna(stamp):
                quoted[str(book)] = pd.Timestamp(stamp).to_pydatetime()

        # names totals and spreads markets by their line
        line = group["line"].iloc[0]
        market_id = f"{event_id}:{market_key}" + ("" if pd.isna(line) else f":{line + 0.0:g}")

        # builds the snapshot
        yield QuoteSnapshot(
            market_id=market_id,
            event_id=str(event_id),
            observed_at_utc=group["captured_at_utc"].max().to_pydatetime(),
            prices=prices,
            quoted_at_utc=quoted,
        )


def without_draws(snapshot: QuoteSnapshot) -> QuoteSnapshot:
    # keeps only books with no draw, as results include overtime
    prices = {book: sides for book, sides in snapshot.prices.items() if "Draw" not in sides}
    quoted = {book: at for book, at in snapshot.quoted_at_utc.items() if book in prices}
    return snapshot.model_copy(update={"prices": prices, "quoted_at_utc": quoted})
