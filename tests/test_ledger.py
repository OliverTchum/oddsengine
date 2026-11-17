"""Ledger tests: the record is append-only and tamper-evident."""

import json
from pathlib import Path

import pytest

from oddsengine.execution.ledger import GENESIS_HASH, Ledger, LedgerError, canonical_json


def pick(n: int) -> dict[str, object]:
    return {"market_id": f"m{n}", "selection": "home", "price": 1.9 + n / 100, "stake": 0.01}


def filled(tmp_path: Path, n: int = 5) -> Ledger:
    ledger = Ledger(tmp_path / "picks.jsonl")
    for i in range(n):
        ledger.append(pick(i))
    return ledger


def test_an_empty_ledger_verifies_and_heads_at_genesis(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "picks.jsonl")
    assert ledger.verify() == 0
    assert ledger.head_hash() == GENESIS_HASH


def test_appending_grows_the_chain(tmp_path: Path) -> None:
    ledger = filled(tmp_path, 5)
    assert ledger.verify() == 5


def test_the_first_entry_follows_genesis(tmp_path: Path) -> None:
    ledger = filled(tmp_path, 1)
    assert next(ledger.entries())["prev_hash"] == GENESIS_HASH


def test_each_entry_points_at_the_one_before_it(tmp_path: Path) -> None:
    entries = list(filled(tmp_path, 4).entries())
    for previous, current in zip(entries[:-1], entries[1:], strict=True):
        assert current["prev_hash"] == previous["sha256"]


def test_head_hash_is_the_last_entrys_hash(tmp_path: Path) -> None:
    ledger = filled(tmp_path, 3)
    assert ledger.head_hash() == list(ledger.entries())[-1]["sha256"]


def test_identical_payloads_still_get_different_hashes(tmp_path: Path) -> None:
    # each hash covers its predecessor, so repeats differ
    ledger = Ledger(tmp_path / "picks.jsonl")
    first, second = ledger.append(pick(1)), ledger.append(pick(1))
    assert first != second


def test_serialisation_is_order_independent() -> None:
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def rewrite(ledger: Ledger, index: int, mutate) -> None:  # type: ignore[no-untyped-def]
    lines = ledger.path.read_text().splitlines()
    record = json.loads(lines[index])
    mutate(record)
    lines[index] = canonical_json(record)
    ledger.path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"price": 99.0}, "does not match its hash"),
        ({"selection": "hone"}, "does not match its hash"),  # one character
    ],
)
def test_editing_a_payload_is_detected(
    tmp_path: Path, changes: dict[str, object], match: str
) -> None:
    ledger = filled(tmp_path, 5)
    rewrite(ledger, 2, lambda r: r["payload"].update(changes))
    with pytest.raises(LedgerError, match=match):
        ledger.verify()


def test_deleting_an_entry_is_detected(tmp_path: Path) -> None:
    # caught by the prev_hash link
    ledger = filled(tmp_path, 5)
    lines = ledger.path.read_text().splitlines()
    ledger.path.write_text("\n".join(lines[:2] + lines[3:]) + "\n")
    with pytest.raises(LedgerError, match="deleted or reordered"):
        ledger.verify()


def test_reordering_entries_is_detected(tmp_path: Path) -> None:
    ledger = filled(tmp_path, 5)
    lines = ledger.path.read_text().splitlines()
    lines[1], lines[3] = lines[3], lines[1]
    ledger.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError):
        ledger.verify()


def test_appending_a_forged_entry_is_detected(tmp_path: Path) -> None:
    ledger = filled(tmp_path, 3)
    forged = {"prev_hash": GENESIS_HASH, "payload": pick(99), "sha256": "0" * 64}
    with ledger.path.open("a") as handle:
        handle.write(canonical_json(forged) + "\n")
    with pytest.raises(LedgerError):
        ledger.verify()


def test_an_untouched_ledger_still_verifies_after_reopening(tmp_path: Path) -> None:
    filled(tmp_path, 6)
    assert Ledger(tmp_path / "picks.jsonl").verify() == 6
