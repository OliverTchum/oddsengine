"""Pinnacle alone: its own devigged prices as the fair probability."""

from oddsengine.models.base import Pooled, devig_books
from oddsengine.types import EventInfo, QuoteSnapshot


class Pinnacle:
    book = "pinnacle"

    def __init__(self, *, method: str) -> None:
        self.method = method

    def fair_probs(
        self, snapshot: QuoteSnapshot, event: EventInfo, *, exclude: str | None = None
    ) -> Pooled | None:
        # None when pinnacle is not quoting, or is the book being bet into
        if self.book == exclude or self.book not in snapshot.prices:
            return None
        fair = devig_books(snapshot, books=[self.book], method=self.method)
        return Pooled(probs=fair[self.book], n_families=1)
