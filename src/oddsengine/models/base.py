"""What every fair-probability model shares: its output, its interface, and devigging
a snapshot's books.
"""

from collections.abc import Iterable
from typing import NamedTuple, Protocol

from oddsengine.models.devig import DEVIG_METHODS
from oddsengine.types import EventInfo, QuoteSnapshot


# fair probability per selection, and how many independent opinions back it
class Pooled(NamedTuple):
    probs: dict[str, float]
    n_families: int


# the interface every model fits; any object with this method can be used
class FairProbSource(Protocol):
    # fair probability per selection for one market, leaving out the excluded book's own
    # price; None means no view, and the market is skipped
    def fair_probs(
        self, snapshot: QuoteSnapshot, event: EventInfo, *, exclude: str | None = None
    ) -> Pooled | None: ...


def devig_books(
    snapshot: QuoteSnapshot, *, books: Iterable[str], method: str
) -> dict[str, dict[str, float]]:
    # fair probability per selection for each named book in the snapshot
    if method not in DEVIG_METHODS:
        raise ValueError(
            f"unknown devig method {method!r}, expected one of {sorted(DEVIG_METHODS)}"
        )
    fair = {}
    for book in books:
        odds = []
        if book in snapshot.prices:
            # this book's prices in a fixed selection order
            spread = snapshot.prices[book]
            selections = sorted(spread)
            for selection in selections:
                odds.append(spread[selection])
            probs = DEVIG_METHODS[method](odds)

            # matches the probabilities back to their selections
            fair[book] = {
                selection: float(p) for selection, p in zip(selections, probs, strict=True)
            }
    return fair
