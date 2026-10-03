"""Step 1: download ETF prices and apply the universe rules (spec section 3).

Writes:
  data/raw/prices_daily.parquet   long table: date, ticker, close, adj_close, volume
  data/processed/universe.csv     one row per candidate with the filter outcome
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import yfinance as yf

from cae.config import (
    CANDIDATES,
    INCEPTION_CUTOFF,
    LIQUIDITY_WINDOW,
    MIN_MEDIAN_DOLLAR_VOLUME,
    PRICE_END,
    PRICE_START,
    PROCESSED,
    RAW,
    USD_ETF,
)


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Daily close, adjusted close and volume for all tickers, as a long table.

    Yahoo batch requests occasionally drop a ticker that downloads fine on its own,
    so tickers missing from the batch are retried individually.
    """
    long = _download(tickers)
    missing = [t for t in tickers if t not in set(long["ticker"])]
    retried = [_download([t]) for t in missing]
    return (
        pd.concat([long, *retried], ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def _download(tickers: list[str]) -> pd.DataFrame:
    wide = yf.download(
        tickers,
        start=PRICE_START.isoformat(),
        # yfinance treats `end` as exclusive.
        end=(PRICE_END + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
        multi_level_index=True,
    )
    if wide.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close", "volume"])
    long = (
        wide[["Close", "Adj Close", "Volume"]]
        .stack(level=1, future_stack=True)
        .reset_index()
        .rename(
            columns={
                "Date": "date",
                "Ticker": "ticker",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        .dropna(subset=["adj_close"])
    )
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None)
    return long


def apply_rules(prices: pd.DataFrame) -> pd.DataFrame:
    """Rules 3.2 (inception) and 3.3 (pre-sample liquidity) for every candidate."""
    lo, hi = (pd.Timestamp(d) for d in LIQUIDITY_WINDOW)
    rows = []
    for ticker, (title, emerging) in CANDIDATES.items():
        p = prices[prices["ticker"] == ticker]
        first = p["date"].min() if not p.empty else pd.NaT
        last = p["date"].max() if not p.empty else pd.NaT
        window = p[(p["date"] >= lo) & (p["date"] <= hi)]
        median_dv = float((window["close"] * window["volume"]).median()) if len(window) else 0.0
        has_data = not p.empty
        incepted = has_data and first < pd.Timestamp(INCEPTION_CUTOFF)
        liquid = median_dv >= MIN_MEDIAN_DOLLAR_VOLUME
        if not has_data:
            reason = "no Yahoo data"
        elif not incepted:
            reason = "inception after cutoff"
        elif not liquid:
            reason = "below liquidity threshold"
        else:
            reason = ""
        rows.append(
            {
                "ticker": ticker,
                "wiki_title": title,
                "emerging": emerging,
                "first_price": first.date() if has_data else None,
                "last_price": last.date() if has_data else None,
                "median_dollar_volume_2014_15": round(median_dv),
                "included": has_data and incepted and liquid,
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    prices = download_prices([*CANDIDATES, USD_ETF])
    prices.to_parquet(RAW / "prices_daily.parquet", index=False)
    universe = apply_rules(prices)
    universe.to_csv(PROCESSED / "universe.csv", index=False)
    pd.set_option("display.width", 200)
    print(universe.sort_values(["included", "median_dollar_volume_2014_15"], ascending=False))
    print(f"\nIncluded: {int(universe['included'].sum())} / {len(universe)}")


if __name__ == "__main__":
    main()
