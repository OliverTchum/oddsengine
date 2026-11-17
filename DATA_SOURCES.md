# Data sources

The examples below are trimmed to the fields the code reads, and every value in them
is invented.

## The Odds API

[the-odds-api.com](https://the-odds-api.com), the main feed. It gives decimal h2h prices
from UK and EU bookmakers and exchanges, the schedule of upcoming games, and final
scores for settlement. Calls are paid in credits: an odds call costs one credit per
market per region (naming up to 10 bookmakers counts as one region), a scores call
costs 2, and the schedule call is free. Every odds capture is archived as Parquet in
`data_local/odds`. Needs an API key.

Request (`data/odds_api.py`, `fetch_odds`):

```
GET https://api.the-odds-api.com/v4/sports/basketball_nba/odds/
    ?apiKey=…&regions=uk,eu&markets=h2h&oddsFormat=decimal&dateFormat=iso
```

Response, one game and one book (the credits left come back in the
`x-requests-remaining` header):

```json
[
  {
    "id": "event0001",
    "sport_key": "basketball_nba",
    "commence_time": "2026-11-17T00:30:00Z",
    "home_team": "Home Town",
    "away_team": "Away United",
    "bookmakers": [
      {
        "key": "book_a",
        "last_update": "2026-11-16T22:14:05Z",
        "markets": [
          {
            "key": "h2h",
            "last_update": "2026-11-16T22:14:05Z",
            "outcomes": [
              { "name": "Home Town", "price": 1.80 },
              { "name": "Away United", "price": 2.05 }
            ]
          }
        ]
      }
    ]
  }
]
```

Each outcome becomes one archived row (`normalise_odds_payload`):

| captured_at_utc | event_id | bookmaker_key | market_key | outcome_name | outcome_point | price_decimal | bookmaker_last_update_utc | market_last_update_utc |
|---|---|---|---|---|---|---|---|---|
| 2026-11-16 22:15:00 | event0001 | book_a | h2h | Home Town | | 1.80 | 2026-11-16 22:14:05 | 2026-11-16 22:14:05 |

The row also carries `sport_key`, `commence_time_utc`, `home_team` and `away_team`.

## Betfair Exchange (delayed key)

Read-only market data from the [Betfair Exchange API](https://developer.betfair.com),
using the free delayed application key. Betfair says delayed prices arrive between 1
and 180 seconds late, and the key returns no traded volume. Each tick archives the best
three back and lay offers for every game matched to an Odds API game, in
`data_local/betfair`. When a game's mid-price probability moves by 2 points or more,
the engine polls the Odds API for that game early. The client only ever calls
`listMarketTypes`, `listMarketCatalogue` and `listMarketBook`, so it cannot place a bet.
Needs a Betfair account and app key, and is skipped without them.

Every call is a POST to `https://api.betfair.com/exchange/betting/rest/v1.0/<operation>/`
with the headers `X-Application: …` (the app key) and `X-Authentication: …` (the
session token).

`listMarketCatalogue` finds each game's market (`data/betfair.py`, `list_markets`):

```json
{
  "filter": {
    "eventTypeIds": ["7522"],
    "marketTypeCodes": ["MATCH_ODDS"],
    "marketStartTime": { "from": "2026-11-16T22:00:00Z", "to": "2026-11-17T04:00:00Z" }
  },
  "marketProjection": ["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
  "sort": "FIRST_TO_START",
  "maxResults": 1000
}
```

```json
[
  {
    "marketId": "1.200000001",
    "marketStartTime": "2026-11-17T00:30:00.000Z",
    "runners": [
      { "selectionId": 101, "runnerName": "Home Town" },
      { "selectionId": 102, "runnerName": "Away United" }
    ]
  }
]
```

A market is matched to an Odds API game when the start times are within 15 minutes and
the runner names match the two teams one way round only.

`listMarketBook` fetches the best offers on matched markets (`list_books`):

```json
{ "marketIds": ["1.200000001"], "priceProjection": { "priceData": ["EX_BEST_OFFERS"] } }
```

```json
[
  {
    "marketId": "1.200000001",
    "status": "OPEN",
    "inplay": false,
    "runners": [
      {
        "selectionId": 101,
        "ex": {
          "availableToBack": [{ "price": 1.79, "size": 412.0 }, { "price": 1.78, "size": 950.0 }],
          "availableToLay": [{ "price": 1.81, "size": 230.0 }, { "price": 1.82, "size": 610.0 }]
        }
      }
    ]
  }
]
```

Each runner becomes one archived row, named with the Odds API's selection name:

| captured_at_utc | event_id | market_id | market_status | inplay | selection_id | runner | back_price_1 | back_size_1 | lay_price_1 | lay_size_1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-16 22:15:00 | event0001 | 1.200000001 | OPEN | false | 101 | Home Town | 1.79 | 412.0 | 1.81 | 230.0 |

The row continues to the second and third levels on each side (`back_price_2` to
`lay_size_3`), with NaN where Betfair shows fewer offers.
