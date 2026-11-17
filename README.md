# oddsengine

A prototype paper trader for sports betting markets. It prices each game from the
market itself, looks for bookmaker prices that have not caught up, sizes a portfolio of
bets under risk limits, and records every step in a tamper-evident ledger. It covers the
NBA, the NHL, the Premier League and the Championship. No real money is placed.

The aim is a platform where people can test and run their own betting strategies with
the risk management already built in. A strategy only has to say what each outcome is
worth (one method, see [Adding a model](#adding-a-model)). The engine does the rest: it
finds the prices worth betting, stakes them as one portfolio under the risk limits,
paper trades them, and grades every bet against the closing line. Today there are two
strategies (the panel consensus and Pinnacle alone), and one runs at a time.

1. **Fair probability.** A model prices every outcome. The default devigs a small panel
   of sharp books and exchanges and pools them, one vote per family of similar books.
2. **Stale prices.** Every UK-licensed venue is searched for a price the fair
   probability says is too generous after commission, skipping anything implausible.
3. **Portfolio.** The slate's joint outcomes are simulated (one shared factor per sport
   and date), then every stake is chosen at once with risk-constrained Kelly under
   drawdown, CVaR, variance and per-book limits.
4. **Paper trading.** A few minutes later the price is checked again to see if it would
   have filled, and edge is measured as closing-line value (CLV).

## Setup

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run python scripts/tick.py --dry-run          # what one tick would poll, and its cost
uv run python scripts/verify_ledger.py records/paper_ledger.jsonl
```

A live tick needs an `ODDS_API_KEY` from [the-odds-api.com](https://the-odds-api.com) in
a local `.env` (never committed). Betfair Exchange prices are optional and skipped
without `BETFAIR_APP_KEY`, `BETFAIR_USERNAME` and `BETFAIR_PASSWORD`.
`scripts/launchd.sh install` runs a tick every 2.5 minutes on macOS.

## Layout

```
configs/default.yaml   every tunable number, including the reference panel
src/oddsengine/
  config.py, types.py  settings and the data passed between stages
  data/                The Odds API and Betfair clients, Parquet archive, snapshots, poll schedule
  models/              fair-probability models, one file each, and the interface they share
  selection/           the search for stale prices
  portfolio/           simulator, CVaR, convex optimiser, slate builder
  execution/           paper trader, hash-chained ledger, closing-line value
scripts/               the live tick, ledger check, Betfair check, launchd installer
records/               the paper ledger
tests/                 the pytest suite, mostly one file per module
DATA_SOURCES.md        what each data feed provides and costs
```

## Adding a model

A model is any class with this method (the interface is in `models/base.py`):

```python
def fair_probs(
    self, snapshot: QuoteSnapshot, event: EventInfo, *, exclude: str | None = None
) -> Pooled | None: ...
```

It gets one market's prices and the game's details, and returns a fair probability per
selection, or `None` to skip the market. Add the file to `models/`, add one line to
`models/registry.py`, and set `decision.fair_prob` in the config. Each order records
which model priced it. Closing lines are always graded with the panel consensus, so
every model is measured against the same close.

## Design rules

1. Everything tunable lives in config, not code. Unknown keys are rejected.
2. Two clocks per price: when we captured it (the only clock lookahead checks use)
   and when the book quoted it (used only to judge staleness).
3. A book missing from config is ignored, so it can never be bet into.
4. The ledger is append-only and the trader's state is rebuilt from it alone.
5. Every number regenerates from a committed config and seed. Money is a Decimal.

## Known limitations

- Paper fills ignore available size, so the fill rate is an upper bound.
- CLV is graded against the engine's own consensus, not an independent close.
