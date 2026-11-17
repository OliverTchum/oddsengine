"""Portfolio stage tests: candidates in, a staked and risk-checked slate out."""

from datetime import UTC, datetime

import pytest

from oddsengine.config import AppConfig, DecisionConfig, RiskConfig, SimConfig
from oddsengine.portfolio.pipeline import MIN_STAKE, build_slate
from oddsengine.types import Candidate, SlateDecision

AS_OF = datetime(2026, 11, 6, 18, 0, tzinfo=UTC)
RISK = RiskConfig(cvar_limit=0.15)


def config(risk: RiskConfig = RISK, **decision: float) -> AppConfig:
    return AppConfig(
        sim=SimConfig(n_sims=3000, seed=11), risk=risk, decision=DecisionConfig(**decision)
    )


def candidate(
    *,
    prob: float = 0.60,
    price: float = 1.9,
    event: str = "e1",
    group: str = "",
) -> Candidate:
    return Candidate(
        market_id=f"{event}:h2h",
        event_id=event,
        group=group,
        selection="home",
        fair_prob=prob,
        quoted_price=price,
    )


def strong(n: int = 4) -> list[Candidate]:
    return [candidate(event=f"e{i}") for i in range(n)]


def total_stake(decision: SlateDecision) -> float:
    return sum(p.stake for p in decision.proposals)


def test_a_strong_slate_gets_staked() -> None:
    decision = build_slate(strong(), config=config(), as_of=AS_OF)
    assert decision.as_of == AS_OF
    assert decision.proposals
    assert all(p.stake > MIN_STAKE for p in decision.proposals)


def test_no_candidates_gives_an_empty_slate_not_an_error() -> None:
    assert build_slate([], config=config(), as_of=AS_OF).proposals == ()


def test_a_proposal_carries_what_it_was_built_from() -> None:
    (proposal,) = build_slate([candidate()], config=config(), as_of=AS_OF).proposals
    assert proposal.market_id == "e1:h2h"
    assert proposal.quoted_price == 1.9
    assert proposal.fair_prob == 0.60
    assert proposal.edge == pytest.approx(0.60 * 1.9 - 1.0)


def test_no_edge_is_not_bet() -> None:
    assert build_slate([candidate(prob=0.52)], config=config(), as_of=AS_OF).proposals == ()


def test_raising_ev_min_bets_less() -> None:
    generous = build_slate(strong(6), config=config(ev_min=0.01), as_of=AS_OF)
    strict = build_slate(strong(6), config=config(ev_min=0.13), as_of=AS_OF)
    assert len(strict.proposals) <= len(generous.proposals)


def test_stakes_respect_the_total_budget() -> None:
    capped = config(RiskConfig(cvar_limit=0.15, max_total_stake=0.05))
    assert total_stake(build_slate(strong(6), config=capped, as_of=AS_OF)) <= 0.05 + 1e-6


def test_a_tighter_drawdown_promise_stakes_less() -> None:
    loose = config(RiskConfig(cvar_limit=0.15, drawdown_prob=0.50))
    tight = config(RiskConfig(cvar_limit=0.15, drawdown_prob=0.01))
    loose_total = total_stake(build_slate(strong(5), config=loose, as_of=AS_OF))
    assert total_stake(build_slate(strong(5), config=tight, as_of=AS_OF)) < loose_total


def test_legs_in_different_groups_diversify_and_get_more_stake() -> None:
    # at rho 0.8 two legs in one group are nearly one bet, in different groups they are independent
    cfg = config().model_copy(update={"sim": SimConfig(n_sims=3000, seed=11, rho=0.8)})
    same = [candidate(event="e1", group="nba"), candidate(event="e2", group="nba")]
    split = [candidate(event="e1", group="nba"), candidate(event="e2", group="epl")]
    split_total = total_stake(build_slate(split, config=cfg, as_of=AS_OF))
    assert split_total > total_stake(build_slate(same, config=cfg, as_of=AS_OF))


def test_the_slate_is_reproducible_from_the_same_config_and_seed() -> None:
    first = build_slate(strong(), config=config(), as_of=AS_OF)
    second = build_slate(strong(), config=config(), as_of=AS_OF)
    assert [p.stake for p in first.proposals] == pytest.approx([p.stake for p in second.proposals])
