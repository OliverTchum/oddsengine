"""The default model: devig every reference book, then pool them with one vote per family.

The closing line is always graded with this model, whichever one made the decision.
"""

from collections import Counter
from collections.abc import Iterable

from oddsengine.config import ConsensusConfig, VenueConfig
from oddsengine.models.base import Pooled, devig_books
from oddsengine.types import EventInfo, QuoteSnapshot


def family_weights(books: Iterable[str], venues: dict[str, VenueConfig]) -> dict[str, float]:
    # splits each family's vote evenly amongst its books
    weights = {}

    books = [b for b in books if b in venues]

    family_sizes = Counter(venues[b].family or b for b in books)

    # each book's share of its family's vote
    for book in books:
        member = venues[book]
        weights[book] = member.weight / family_sizes[member.family or book]

    # scales the weights to sum to 1
    tot = sum(weights.values())
    for book, weight in weights.items():
        weights[book] = weight / tot
    return weights


def consensus(
    fair: dict[str, dict[str, float]],
    venues: dict[str, VenueConfig],
    *,
    min_families: int,
    exclude: str | None = None,
) -> Pooled | None:
    # pools the reference books' fair probabilities, or None if too few families

    # the books allowed to vote, in name order
    eligible = []
    for book in sorted(fair):
        if book in venues and venues[book].role != "venue" and book != exclude:
            eligible.append(book)

    # too few independent families means no consensus
    families = {venues[b].family or b for b in eligible}
    if len(families) < min_families:
        return None

    # weighted average of the voting books' fair probabilities, per selection
    weights = family_weights(eligible, venues)
    panel: dict[str, float] = {}
    for book in eligible:
        spread = fair[book]
        for selection, p in spread.items():
            panel[selection] = panel.get(selection, 0.0) + p * weights[book]

    return Pooled(probs=panel, n_families=len(families))


# the panel consensus as a model; it reads only the prices, not the event
class PanelConsensus:
    def __init__(self, venues: dict[str, VenueConfig], cfg: ConsensusConfig) -> None:
        self.venues = venues
        self.cfg = cfg
        # the books allowed to vote: reference or both
        self.panel = [book for book in venues if venues[book].role != "venue"]

    def fair_probs(
        self, snapshot: QuoteSnapshot, event: EventInfo, *, exclude: str | None = None
    ) -> Pooled | None:
        # devig each panel book on its own, then the weighted pool
        fair = devig_books(snapshot, books=self.panel, method=self.cfg.devig_method)
        return consensus(
            fair, self.venues, min_families=self.cfg.min_reference_families, exclude=exclude
        )
