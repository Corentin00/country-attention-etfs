# country-attention-etfs

Does abnormal global English-language attention to a country, measured by
English Wikipedia page views, predict the relative returns of that country's
iShares MSCI ETF? Weekly cross-sectional test, 2015-2026.

Systematic-track assignment for the Hedge Fund Club St. Gallen.

The research specification was frozen on 2026-10-03, before any return was
computed: [docs/spec_frozen_2026-10-03.txt](docs/spec_frozen_2026-10-03.txt).

## Data (all free)

- Page views: Wikimedia REST API (`en.wikipedia`, `agent=user`, daily, from 2015-07-01)
- Prices: Yahoo Finance via `yfinance` (adjusted close, volume)
- Risk-free rate: Ken French Data Library
- US dollar: UUP ETF (Yahoo)

Raw downloads are committed under `data/raw/` so results are reproducible even
if the sources change.

## Run

```bash
uv sync
uv run python -m cae.universe
```
