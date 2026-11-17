"""Append-only Parquet archive of every capture.

Each capture is written as a new file and nothing is changed or deleted. Paths look like:

    <root>/sport=<sport_key>/date=<YYYY-MM-DD>/<YYYYMMDDTHHMMSSZ>[-<tag>].parquet
"""

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from oddsengine.data.odds_api import QUOTE_COLUMNS

STAMP = "%Y%m%dT%H%M%SZ"
STAMP_LENGTH = len("20260101T000000Z")


def partition_path(root: Path, *, sport: str, captured_at: datetime, tag: str = "") -> Path:
    # builds the path: sport and date folders, then the time and tag as the file name
    name = captured_at.strftime(STAMP) + (f"-{tag}" if tag else "") + ".parquet"
    return root / f"sport={sport}" / f"date={captured_at:%Y-%m-%d}" / name


def _stamp(path: Path) -> datetime:
    # reads the capture time from the file name
    return datetime.strptime(path.name[:STAMP_LENGTH], STAMP).replace(tzinfo=UTC)


def _tag(path: Path) -> str:
    # reads the kind of poll from the file name
    return path.stem[STAMP_LENGTH:].removeprefix("-")


def last_capture(root: Path, *, sport: str, tag: str = "") -> datetime | None:
    # finds when this kind of capture last ran for a sport
    paths = (root / f"sport={sport}").glob("date=*/*.parquet")
    return max((_stamp(p) for p in paths if _tag(p) == tag), default=None)


def captures_since(root: Path, *, sport: str, since: datetime) -> list[tuple[datetime, str, Path]]:
    # lists the captures of a sport from a given time on, oldest first
    found = []
    for path in (root / f"sport={sport}").glob("date=*/*.parquet"):
        stamp = _stamp(path)
        if stamp >= since:
            found.append((stamp, _tag(path), path))
    return sorted(found)


def write_snapshot(
    root: Path,
    *,
    sport: str,
    rows: Sequence[dict[str, Any]],
    captured_at: datetime,
    tag: str = "",
    columns: Sequence[str] = QUOTE_COLUMNS,
) -> Path | None:
    # writes one capture and returns its path
    if not rows:
        return None
    path = partition_path(root, sport=sport, captured_at=captured_at, tag=tag)

    # never overwrites an existing capture
    if path.exists():
        raise FileExistsError(f"capture already archived: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    # writes to a temporary file then renames it, so a crash never leaves a half file
    temp = path.with_name(path.name + ".tmp")
    pd.DataFrame(list(rows), columns=list(columns)).to_parquet(
        temp, index=False, compression="zstd"
    )
    os.replace(temp, path)
    return path
