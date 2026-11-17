"""Config tests: the default config parses, and typos fail loudly at startup."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from oddsengine.config import Secrets, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "default.yaml"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT)
    assert (cfg.sim.n_sims, cfg.sim.seed, cfg.sim.rho) == (10_000, 7, 0.05)
    assert (cfg.risk.cvar_limit, cfg.risk.variance_limit) == (0.08, 0.02)
    assert cfg.archiver.markets == ["h2h"]


def test_default_config_declares_the_reference_panel_and_uk_venues() -> None:
    cfg = load_config(DEFAULT)
    families = {v.family or k for k, v in cfg.venues.items() if v.role in ("reference", "both")}
    assert families == {"pinnacle", "betfair_ex", "evoke", "kambi"}
    assert cfg.consensus.min_reference_families <= len(families)
    # Pinnacle leads (best or tied in football and the NBA); the rest share equally
    weights = {v.family or k: v.weight for k, v in cfg.venues.items() if v.role != "venue"}
    assert weights == {"pinnacle": 2.0, "betfair_ex": 1.25, "evoke": 1.25, "kambi": 1.25}
    # eu-only books may inform the consensus but must never be bettable
    assert cfg.venues["unibet_nl"].role == "reference"
    assert cfg.venues["smarkets"].commission > 0


@pytest.mark.parametrize(
    "text",
    [
        "simulation:\n  n_sims: 10\n",  # misspelt section
        "venues:\n  coral: { role: referee }\n",  # unknown venue role
        "consensus:\n  devig_method: average\n",  # unknown devig method
    ],
    ids=["unknown key", "unknown venue role", "unknown devig method"],
)
def test_a_typo_in_the_config_is_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValidationError):
        load_config(write(tmp_path, text))


def test_family_members_with_different_weights_are_rejected(tmp_path: Path) -> None:
    text = (
        "venues:\n"
        "  coral: { role: reference, family: entain, weight: 1.0 }\n"
        "  ladbrokes_uk: { role: reference, family: entain, weight: 2.0 }\n"
    )
    with pytest.raises(ValidationError, match="different weights"):
        load_config(write(tmp_path, text))


def test_the_api_key_is_readable_but_hidden_from_repr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # chdir so no .env file in the repo is picked up
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ODDS_API_KEY", "secret-123")
    assert "secret-123" not in repr(Secrets())
    assert Secrets().odds_api_key.get_secret_value() == "secret-123"
