"""Consensus tests: worked examples for the consensus model and select, then hypothesis properties
of the pool.
"""

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from oddsengine.config import ConsensusConfig, VenueConfig
from oddsengine.models.base import devig_books
from oddsengine.models.consensus import consensus, family_weights
from oddsengine.selection.select import Pick, fresh, net_price, select
from oddsengine.types import EventInfo, QuoteSnapshot

AS_OF = datetime(2026, 11, 6, 19, 0, tzinfo=UTC)
SELS = ("home", "away", "draw")

REF = VenueConfig(role="reference")
VENUE = VenueConfig(role="venue")

# hand-computable settings: multiplicative devig, one family is enough
CFG = ConsensusConfig(devig_method="multiplicative", min_reference_families=1)


def fair(*probs: float) -> dict[str, float]:
    # margin-free prices that devig back to these probabilities
    return {name: 1.0 / p for name, p in zip(SELS, probs, strict=False)}


def snap(
    prices: dict[str, dict[str, float]],
    *,
    event: str = "e1",
    quoted: dict[str, datetime] | None = None,
) -> QuoteSnapshot:
    # one h2h market, every book quoted at AS_OF unless told otherwise
    return QuoteSnapshot(
        market_id=f"{event}:h2h",
        event_id=event,
        observed_at_utc=AS_OF,
        prices=prices,
        quoted_at_utc={b: AS_OF for b in prices} if quoted is None else quoted,
    )


def events(snapshots: list[QuoteSnapshot]) -> dict[str, EventInfo]:
    # every snapshot's game, starting in two hours
    start = AS_OF + timedelta(hours=2)
    return {
        s.event_id: EventInfo(event_id=s.event_id, sport="basketball_nba", commence_utc=start)
        for s in snapshots
    }


def run(
    snapshots: list[QuoteSnapshot],
    venues: dict[str, VenueConfig],
    *,
    ev_min: float = 0.0,
    **cfg: object,
) -> list[Pick]:
    # runs select with CFG plus overrides and returns the picks
    config = CFG.model_copy(update=cfg)
    found = select(
        snapshots, events=events(snapshots), venues=venues, cfg=config, ev_min=ev_min, as_of=AS_OF
    )
    return list(found.picks)


def test_net_price_charges_commission_on_winnings_only() -> None:
    assert net_price(3.0, 0.02) == pytest.approx(2.96)
    assert net_price(2.10, 0.05) == pytest.approx(2.045)
    assert net_price(1.80, 0.0) == 1.80


@settings(max_examples=200, deadline=None)
@given(
    price=st.floats(1.01, 100.0),
    bump=st.floats(0.01, 10.0),
    commission=st.floats(0.0, 0.2),
)
def test_net_price_stays_above_one_below_gross_and_rises_with_price(
    price: float, bump: float, commission: float
) -> None:
    net = net_price(price, commission)
    assert 1.0 < net <= price
    assert net_price(price + bump, commission) > net


FULL = fair(0.5, 0.5)
MAX_AGE = timedelta(minutes=10)
SKEW = timedelta(seconds=60)


@pytest.mark.parametrize(
    ("prices", "quoted", "kept"),
    [
        # exactly 10 minutes old stays, one second more goes
        (
            {"kept": FULL, "old": FULL},
            {"kept": AS_OF - MAX_AGE, "old": AS_OF - MAX_AGE - timedelta(seconds=1)},
            {"kept"},
        ),
        # a quote stamped in the future is kept up to the allowed skew
        (
            {"ok": FULL, "bad": FULL},
            {"ok": AS_OF + SKEW, "bad": AS_OF + SKEW + timedelta(seconds=1)},
            {"ok"},
        ),
        # no quote time, so it cannot be fresh
        ({"timed": FULL, "untimed": FULL}, {"timed": AS_OF}, {"timed"}),
        # a 1X2 market without the draw cannot be devigged
        (
            {"full": fair(0.5, 0.25, 0.25), "partial": {"home": 2.0, "away": 4.0}},
            None,
            {"full"},
        ),
    ],
    ids=["age limit", "future skew", "no quote time", "missing selection"],
)
def test_fresh_keeps_only_usable_books(
    prices: dict[str, dict[str, float]], quoted: dict[str, datetime] | None, kept: set[str]
) -> None:
    out = fresh(snap(prices, quoted=quoted), as_of=AS_OF, max_age=MAX_AGE, max_skew=SKEW)
    assert set(out.prices) == kept
    assert set(out.quoted_at_utc) == kept


def test_each_book_is_devigged_on_its_own() -> None:
    s = snap({"fairbook": {"home": 1.25, "away": 5.0}, "vigbook": {"home": 1.80, "away": 2.10}})
    out = devig_books(s, books=["fairbook", "vigbook"], method="multiplicative")
    assert out["fairbook"] == pytest.approx({"home": 0.8, "away": 0.2})
    # a 3.17% margin removed proportionally
    assert out["vigbook"] == pytest.approx({"home": 0.53846, "away": 0.46154}, abs=1e-5)


def test_only_the_named_books_are_devigged() -> None:
    s = snap({"a": fair(0.5, 0.5), "b": fair(0.6, 0.4)})
    assert set(devig_books(s, books=["a", "missing"], method="power")) == {"a"}


def test_an_unknown_devig_method_raises() -> None:
    with pytest.raises(ValueError):
        devig_books(snap({"a": fair(0.5, 0.5)}), books=["a"], method="nonsense")


def test_weighted_and_equal_panels_pool_to_the_hand_computed_value() -> None:
    # weights 2/2/1 normalise to 0.4/0.4/0.2
    fairs = {
        "pinnacle": {"home": 0.540, "away": 0.460},
        "betfair": {"home": 0.550, "away": 0.450},
        "wh": {"home": 0.520, "away": 0.480},
    }
    weighted = {
        "pinnacle": VenueConfig(role="reference", weight=2.0),
        "betfair": VenueConfig(role="reference", weight=2.0),
        "wh": VenueConfig(role="reference", weight=1.0),
    }
    equal = {book: REF for book in fairs}
    pooled = consensus(fairs, weighted, min_families=1)
    assert pooled is not None and pooled.probs["home"] == pytest.approx(0.540)
    pooled = consensus(fairs, equal, min_families=1)
    assert pooled is not None and pooled.probs["home"] == pytest.approx(0.53667, abs=1e-5)


def test_a_family_gets_one_vote_between_its_members() -> None:
    # Ladbrokes and Coral are one opinion, not two
    fairs = {
        "pinnacle": {"home": 0.54, "away": 0.46},
        "ladbrokes": {"home": 0.50, "away": 0.50},
        "coral": {"home": 0.50, "away": 0.50},
    }
    entain = VenueConfig(role="reference", family="entain")
    by_family = consensus(
        fairs, {"pinnacle": REF, "ladbrokes": entain, "coral": entain}, min_families=1
    )
    by_book = consensus(fairs, {b: REF for b in fairs}, min_families=1)
    assert by_family is not None and by_family.probs["home"] == pytest.approx(0.52)
    assert by_family.n_families == 2
    assert by_book is not None and by_book.probs["home"] == pytest.approx(0.51333, abs=1e-5)


def test_adding_an_identical_family_member_changes_nothing() -> None:
    entain = VenueConfig(role="reference", family="entain")
    base = {"pinnacle": {"home": 0.54, "away": 0.46}, "ladbrokes": {"home": 0.50, "away": 0.50}}
    plus = {**base, "coral": {"home": 0.50, "away": 0.50}}
    venues = {"pinnacle": REF, "ladbrokes": entain, "coral": entain}
    a = consensus(base, venues, min_families=1)
    b = consensus(plus, venues, min_families=1)
    assert a is not None and b is not None
    assert a.probs == pytest.approx(b.probs)


def test_family_weights_split_a_vote_evenly_and_sum_to_one() -> None:
    entain = VenueConfig(role="reference", family="entain", weight=3.0)
    venues = {"pinnacle": VenueConfig(role="reference", weight=1.0), "lad": entain, "cor": entain}
    w = family_weights(["pinnacle", "lad", "cor", "unconfigured"], venues)
    assert w == pytest.approx({"pinnacle": 0.25, "lad": 0.375, "cor": 0.375})


def test_venue_only_and_unconfigured_books_never_enter_the_consensus() -> None:
    fairs = {"ref": {"home": 0.6, "away": 0.4}}
    noisy = {**fairs, "soft": {"home": 0.1, "away": 0.9}, "stranger": {"home": 0.9, "away": 0.1}}
    a = consensus(fairs, {"ref": REF, "soft": VENUE}, min_families=1)
    b = consensus(noisy, {"ref": REF, "soft": VENUE}, min_families=1)
    assert a is not None and b is not None and a.probs == pytest.approx(b.probs)


def test_leave_one_out_ignores_the_judged_book_entirely() -> None:
    both = VenueConfig(role="both")
    venues = {"pin": REF, "bfx": both}
    a = {"pin": {"home": 0.6, "away": 0.4}, "bfx": {"home": 0.3, "away": 0.7}}
    b = {"pin": {"home": 0.6, "away": 0.4}, "bfx": {"home": 0.9, "away": 0.1}}
    pa = consensus(a, venues, min_families=1, exclude="bfx")
    pb = consensus(b, venues, min_families=1, exclude="bfx")
    assert pa is not None and pb is not None
    assert pa.probs == pytest.approx(pb.probs)
    assert pa.probs == pytest.approx({"home": 0.6, "away": 0.4})


def test_too_few_independent_families_gives_no_consensus() -> None:
    entain = VenueConfig(role="reference", family="entain")
    fairs = {"lad": {"home": 0.5, "away": 0.5}, "cor": {"home": 0.5, "away": 0.5}}
    assert consensus(fairs, {"lad": entain, "cor": entain}, min_families=2) is None
    fairs["pin"] = {"home": 0.54, "away": 0.46}
    pooled = consensus(fairs, {"lad": entain, "cor": entain, "pin": REF}, min_families=2)
    assert pooled is not None and pooled.n_families == 2


def test_a_pick_carries_the_hand_computed_edge() -> None:
    # p 0.60, price 1.80, haircut 0.01: 0.60 * 1.80 * 0.99 - 1 = 0.0692
    s = snap({"pin": fair(0.6, 0.4), "soft": {"home": 1.80, "away": 2.0}})
    [pick] = run([s], {"pin": REF, "soft": VENUE})
    assert (pick.book, pick.selection) == ("soft", "home")
    assert pick.edge == pytest.approx(0.0692)
    assert pick.p_consensus == pytest.approx(0.6) and pick.n_families == 1


def test_the_best_offer_is_judged_net_of_commission() -> None:
    # an exchange at 1.84 and 5% commission nets 1.798, below a soft book's 1.80
    s = snap(
        {
            "pin": fair(0.6, 0.4),
            "soft": {"home": 1.80, "away": 2.0},
            "exch": {"home": 1.84, "away": 2.0},
        }
    )
    venues = {"pin": REF, "soft": VENUE, "exch": VenueConfig(role="venue", commission=0.05)}
    [pick] = run([s], venues)
    assert pick.book == "soft"


def test_commission_changes_the_price_taken_never_the_opinion() -> None:
    s = snap({"bfx": fair(0.6, 0.4), "pin": fair(0.6, 0.4), "soft": {"home": 1.80, "away": 2.0}})
    plain = run([s], {"bfx": REF, "pin": REF, "soft": VENUE})
    taxed = run(
        [s], {"bfx": VenueConfig(role="reference", commission=0.05), "pin": REF, "soft": VENUE}
    )
    assert plain[0].p_consensus == pytest.approx(taxed[0].p_consensus)


def test_a_book_in_both_roles_is_judged_without_its_own_price() -> None:
    # left in, bfx would drag p from 0.60 to about 0.569 and cut its edge from 6.9% to 1.4%
    s = snap({"pin": fair(0.6, 0.4), "bfx": {"home": 1.80, "away": 2.10}})
    [pick] = run([s], {"pin": REF, "bfx": VenueConfig(role="both")})
    assert pick.book == "bfx"
    assert pick.p_consensus == pytest.approx(0.6)
    assert pick.edge == pytest.approx(0.0692)


def test_unconfigured_books_are_never_bet_into() -> None:
    s = snap({"pin": fair(0.6, 0.4), "stranger": {"home": 1.85, "away": 2.0}})
    assert run([s], {"pin": REF}) == []


def test_picks_are_ranked_by_edge() -> None:
    snaps = [
        snap({"pin": fair(0.6, 0.4), "soft": {"home": price, "away": 2.0}}, event=event)
        for event, price in (("e1", 1.734), ("e2", 1.768), ("e3", 1.751))
    ]
    picks = run(snaps, {"pin": REF, "soft": VENUE})
    assert [p.event_id for p in picks] == ["e2", "e3", "e1"]
    assert picks[0].edge > picks[1].edge > picks[2].edge


def test_the_probability_floor_drops_a_longshot_even_with_the_largest_edge() -> None:
    # away: 0.08 * 13.5 * 0.99 - 1 = 6.9%; home: 0.92 * 1.12 * 0.99 - 1 = 2.0%
    s = snap({"pin": fair(0.92, 0.08), "soft": {"home": 1.12, "away": 13.5}})
    venues = {"pin": REF, "soft": VENUE}
    [floored] = run([s], venues, min_prob=0.10)
    assert floored.selection == "home" and floored.edge == pytest.approx(0.0201, abs=1e-4)
    [unfloored] = run([s], venues, min_prob=0.0)
    assert unfloored.selection == "away"


def test_an_implausibly_large_edge_is_flagged_not_picked() -> None:
    # 0.6 * 2.0 * 0.99 - 1 = 18.8%, above max_edge 0.10: a bug until proven otherwise
    s = snap({"pin": fair(0.6, 0.4), "soft": {"home": 2.0, "away": 2.0}})
    out = select(
        [s],
        events=events([s]),
        venues={"pin": REF, "soft": VENUE},
        cfg=CFG,
        ev_min=0.0,
        as_of=AS_OF,
    )
    assert out.picks == ()
    assert [(p.book, p.selection) for p in out.flagged] == [("soft", "home")]


def test_a_book_far_from_the_consensus_is_disputed_not_picked() -> None:
    # soft's own view of home is 0.6998 against a consensus of 0.80, a gap of 0.1002; its edge is
    # 0.8 * 1.36 * 0.99 - 1 = 7.7%
    s = snap({"pin": fair(0.8, 0.2), "soft": {"home": 1.36, "away": 3.17}})
    venues = {"pin": REF, "soft": VENUE}
    out = select([s], events=events([s]), venues=venues, cfg=CFG, ev_min=0.0, as_of=AS_OF)
    assert out.picks == () and out.flagged == ()
    assert [(p.book, p.selection) for p in out.disputed] == [("soft", "home")]
    assert len(run([s], venues, max_disagreement=0.5)) == 1


def test_a_disputed_option_leaves_the_event_its_next_best_pick() -> None:
    # a has 7.7% edge but gap 0.1002, b has 3.0% edge and gap 0.0653, so b is picked
    s = snap(
        {
            "pin": fair(0.8, 0.2),
            "a": {"home": 1.36, "away": 3.17},
            "b": {"home": 1.30, "away": 3.6},
        }
    )
    [pick] = run([s], {"pin": REF, "a": VENUE, "b": VENUE})
    assert (pick.book, pick.selection) == ("b", "home")
    assert pick.edge == pytest.approx(0.0296, abs=1e-4)


def test_the_disagreement_check_devigs_with_the_configured_method() -> None:
    # soft's view of home is 0.4487 multiplicative or 0.4459 power, so against 0.50 the gaps are
    # 0.0513 and 0.0541
    s = snap({"pin": fair(0.5, 0.5), "soft": {"home": 2.15, "away": 1.75}})
    venues = {"pin": REF, "soft": VENUE}
    assert len(run([s], venues, max_disagreement=0.0525)) == 1
    assert run([s], venues, max_disagreement=0.0525, devig_method="power") == []


def test_an_edge_exactly_at_ev_min_is_kept() -> None:
    # p 0.5 at price 2.5 with no haircut is edge 0.25 exactly; max_edge and max_disagreement are
    # loosened
    s = snap({"pin": fair(0.5, 0.5), "soft": {"home": 2.5, "away": 1.5}})
    venues = {"pin": REF, "soft": VENUE}
    loose: dict[str, object] = {"latency_haircut": 0.0, "max_edge": 0.5, "max_disagreement": 0.5}
    assert len(run([s], venues, ev_min=0.25, **loose)) == 1
    assert run([s], venues, ev_min=0.2500001, **loose) == []


def test_max_candidates_keeps_the_top_by_edge() -> None:
    snaps = [
        snap({"pin": fair(0.6, 0.4), "soft": {"home": price, "away": 2.0}}, event=event)
        for event, price in (("e1", 1.734), ("e2", 1.768), ("e3", 1.751), ("e4", 1.74))
    ]
    picks = run(snaps, {"pin": REF, "soft": VENUE}, max_candidates=2)
    assert [p.event_id for p in picks] == ["e2", "e3"]


def test_one_selection_per_event() -> None:
    # home at A: 0.5 * 2.12 * 0.99 - 1 = 4.9%; away at B: 0.5 * 2.10 * 0.99 - 1 = 4.0%
    s = snap(
        {
            "pin": fair(0.5, 0.5),
            "a": {"home": 2.12, "away": 1.80},
            "b": {"home": 1.80, "away": 2.10},
        }
    )
    [pick] = run([s], {"pin": REF, "a": VENUE, "b": VENUE})
    assert (pick.book, pick.selection) == ("a", "home")


def test_no_consensus_means_no_pick() -> None:
    s = snap({"pin": fair(0.6, 0.4), "soft": {"home": 1.80, "away": 2.0}})
    assert run([s], {"pin": REF, "soft": VENUE}, min_reference_families=2) == []


def test_stale_reference_books_cannot_form_a_consensus() -> None:
    quoted = {"pin": AS_OF - timedelta(hours=1), "soft": AS_OF}
    s = snap({"pin": fair(0.6, 0.4), "soft": {"home": 1.80, "away": 2.0}}, quoted=quoted)
    assert run([s], {"pin": REF, "soft": VENUE}) == []


def test_select_ignores_input_order() -> None:
    venues = {"pin": REF, "a": VENUE, "b": VENUE}
    s1 = snap({"pin": fair(0.6, 0.4), "a": {"home": 1.76, "away": 2.0}}, event="e1")
    s2 = snap({"b": {"home": 1.75, "away": 2.0}, "pin": fair(0.6, 0.4)}, event="e2")
    assert run([s1, s2], venues) == run([s2, s1], venues)


def test_tied_edges_break_on_market_not_input_order() -> None:
    # identical prices give identical edges, so e1 survives the cut either way
    venues = {"pin": REF, "soft": VENUE}
    s1, s2 = (
        snap({"pin": fair(0.6, 0.4), "soft": {"home": 1.76, "away": 2.0}}, event=e)
        for e in ("e1", "e2")
    )
    for order in ([s1, s2], [s2, s1]):
        [pick] = run(order, venues, max_candidates=1)
        assert pick.event_id == "e1"


def test_tied_venues_break_on_book_name_not_config_order() -> None:
    # a tie on price within an event goes to the first name
    s = snap(
        {
            "pin": fair(0.6, 0.4),
            "a": {"home": 1.76, "away": 2.0},
            "b": {"home": 1.76, "away": 2.0},
        }
    )
    for venues in ({"pin": REF, "a": VENUE, "b": VENUE}, {"pin": REF, "b": VENUE, "a": VENUE}):
        [pick] = run([s], venues)
        assert pick.book == "a"


def simplex(k: int) -> st.SearchStrategy[list[float]]:
    return st.lists(st.floats(0.05, 1.0), min_size=k, max_size=k).map(
        lambda xs: [x / sum(xs) for x in xs]
    )


@st.composite
def panels(draw: st.DrawFn) -> tuple[dict[str, dict[str, float]], dict[str, VenueConfig]]:
    # random reference books with random families and one weight per family
    k = draw(st.sampled_from([2, 3]))
    family_weight = {f: draw(st.floats(0.1, 5.0)) for f in ("fa", "fb", "fc")}
    fairs: dict[str, dict[str, float]] = {}
    venues: dict[str, VenueConfig] = {}
    for i in range(draw(st.integers(1, 6))):
        family = draw(st.sampled_from(sorted(family_weight)))
        fairs[f"b{i}"] = dict(zip(SELS, draw(simplex(k)), strict=False))
        venues[f"b{i}"] = VenueConfig(role="reference", family=family, weight=family_weight[family])
    return fairs, venues


@settings(max_examples=200, deadline=None)
@given(panel=panels())
def test_weights_are_positive_and_sum_to_one(
    panel: tuple[dict[str, dict[str, float]], dict[str, VenueConfig]],
) -> None:
    fairs, venues = panel
    w = family_weights(fairs, venues)
    assert set(w) == set(fairs)
    assert all(x > 0 for x in w.values())
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)


@settings(max_examples=200, deadline=None)
@given(panel=panels())
def test_consensus_is_a_distribution_and_ignores_book_order(
    panel: tuple[dict[str, dict[str, float]], dict[str, VenueConfig]],
) -> None:
    fairs, venues = panel
    pooled = consensus(fairs, venues, min_families=1)
    flipped = consensus(dict(reversed(list(fairs.items()))), venues, min_families=1)
    assert pooled is not None and flipped is not None
    assert sum(pooled.probs.values()) == pytest.approx(1.0, abs=1e-9)
    assert pooled.probs == pytest.approx(flipped.probs, abs=1e-12)


@settings(max_examples=200, deadline=None)
@given(panel=panels())
def test_consensus_never_leaves_the_range_of_its_inputs(
    panel: tuple[dict[str, dict[str, float]], dict[str, VenueConfig]],
) -> None:
    fairs, venues = panel
    pooled = consensus(fairs, venues, min_families=1)
    assert pooled is not None
    for sel, p in pooled.probs.items():
        opinions = [book[sel] for book in fairs.values()]
        assert min(opinions) - 1e-12 <= p <= max(opinions) + 1e-12


@settings(max_examples=200, deadline=None)
@given(
    probs=simplex(3), copies=st.integers(1, 6), weights=st.lists(st.floats(0.1, 5.0), min_size=6)
)
def test_identical_books_reproduce_that_book(
    probs: list[float], copies: int, weights: list[float]
) -> None:
    one = dict(zip(SELS, probs, strict=True))
    fairs = {f"b{i}": one for i in range(copies)}
    venues = {f"b{i}": VenueConfig(role="reference", weight=weights[i]) for i in range(copies)}
    pooled = consensus(fairs, venues, min_families=1)
    assert pooled is not None and pooled.probs == pytest.approx(one, abs=1e-12)
