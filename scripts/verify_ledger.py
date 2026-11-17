"""Recomputes a ledger's hash chain from scratch.

Anyone can run this on a ledger to confirm no record was edited, deleted or reordered.

  uv run python scripts/verify_ledger.py records/paper_ledger.jsonl
"""

import sys
from pathlib import Path

from oddsengine.execution.ledger import Ledger, LedgerError


def main(path: str) -> int:
    # checks the ledger and returns the exit code
    ledger = Ledger(Path(path))
    try:
        count = ledger.verify()
    except LedgerError as exc:
        print(f"CHAIN BROKEN: {exc}", file=sys.stderr)
        return 1
    print(f"chain verified: {count} entries, head hash {ledger.head_hash()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: verify_ledger.py <ledger.jsonl>")
    raise SystemExit(main(sys.argv[1]))
