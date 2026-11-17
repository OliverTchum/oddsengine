"""Append-only, hash-chained record of every decision, fill and settlement.

Each line stores the hash of the line before it, so editing, deleting or
reordering a past record breaks every hash after it. The chain is
tamper-evident, not tamper-proof.
"""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# the prev_hash of the first entry
GENESIS_HASH = "0" * 64


# raised when the chain does not check out
class LedgerError(RuntimeError):
    pass


def canonical_json(payload: dict[str, Any]) -> str:
    # sorted keys and no spaces, so the same record always hashes the same
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def payload_digest(payload: dict[str, Any], *, prev_hash: str) -> str:
    # the SHA-256 hex of the previous hash plus this payload
    return hashlib.sha256((prev_hash + canonical_json(payload)).encode()).hexdigest()


# a JSON-lines file of {prev_hash, payload, sha256} records
class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        # the latest hash, read from the file once and then kept up to date
        self._head: str | None = None

    def entries(self) -> Iterator[dict[str, Any]]:
        # every record in the file, oldest first
        if not self.path.exists():
            return
        with self.path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def head_hash(self) -> str:
        # hash of the latest entry
        if self._head is None:
            self._head = GENESIS_HASH
            for entry in self.entries():
                self._head = entry["sha256"]
        return self._head

    def append(self, payload: dict[str, Any]) -> str:
        # adds one record and returns its hash
        prev_hash = self.head_hash()
        digest = payload_digest(payload, prev_hash=prev_hash)
        record = {"prev_hash": prev_hash, "payload": payload, "sha256": digest}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(canonical_json(record) + "\n")
        self._head = digest
        return digest

    def verify(self) -> int:
        # recomputes the chain and returns the number of entries, or raises
        expected_prev = GENESIS_HASH
        count = 0
        for index, entry in enumerate(self.entries()):
            if entry.get("prev_hash") != expected_prev:
                raise LedgerError(
                    f"entry {index}: prev_hash breaks the chain (deleted or reordered?)"
                )
            recomputed = payload_digest(entry["payload"], prev_hash=expected_prev)
            if recomputed != entry.get("sha256"):
                raise LedgerError(f"entry {index}: payload does not match its hash (edited?)")
            expected_prev = recomputed
            count += 1
        return count
