"""Every model decide can use, by the name `decision.fair_prob` gives it.

A new model is one new file in this folder plus one line here, then its name in the
YAML; decide does not change.
"""

from collections.abc import Callable

from oddsengine.config import AppConfig
from oddsengine.models.base import FairProbSource
from oddsengine.models.consensus import PanelConsensus
from oddsengine.models.pinnacle import Pinnacle

# builds each model from the config
FAIR_PROB_SOURCES: dict[str, Callable[[AppConfig], FairProbSource]] = {
    "consensus": lambda config: PanelConsensus(config.venues, config.consensus),
    "pinnacle": lambda config: Pinnacle(method=config.consensus.devig_method),
}


def fair_prob_source(config: AppConfig) -> FairProbSource:
    # the model named by decision.fair_prob; an unknown name raises ValueError
    name = config.decision.fair_prob
    if name not in FAIR_PROB_SOURCES:
        raise ValueError(f"unknown fair_prob {name!r}, expected one of {sorted(FAIR_PROB_SOURCES)}")
    return FAIR_PROB_SOURCES[name](config)
