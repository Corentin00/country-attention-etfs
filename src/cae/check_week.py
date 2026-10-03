"""Recompute one panel row by hand from the raw daily files (independent of cae.panel).

Usage: uv run python -m cae.check_week EWZ 2022-10-30
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from cae.config import PROCESSED, RAW


def main(ticker: str, week_end: str) -> None:
    sunday = pd.Timestamp(week_end)
    universe = pd.read_csv(PROCESSED / "universe.csv").set_index("ticker")
    title = universe.loc[ticker, "wiki_title"]
    views = pd.read_parquet(RAW / "views_daily.parquet")
    v = views[views["title"] == title].set_index("date")["views"]

    print(f"== {ticker} ({title}), signal week ending Sunday {sunday.date()} ==\n")
    sums = []
    for k in range(8, -1, -1):
        end = sunday - pd.Timedelta(weeks=k)
        start = end - pd.Timedelta(days=6)
        days = v.loc[start:end]
        assert len(days) == 7, f"week {start.date()}-{end.date()} has {len(days)} days"
        sums.append(int(days.sum()))
        label = "this week (V)" if k == 0 else f"week -{k}"
        print(f"  {label:14s} {start.date()} to {end.date()}: {sums[-1]:>9,}")
    baseline = float(np.median(sums[:-1]))
    asv8 = np.log(sums[-1]) - np.log(baseline)
    print(f"\n  median of the 8 previous weeks = {baseline:,.1f}")
    print(f"  ASV = ln({sums[-1]:,}) - ln({baseline:,.1f}) = {asv8:.4f}   (~{np.exp(asv8) - 1:+.0%} vs normal)")

    prices = pd.read_parquet(RAW / "prices_daily.parquet")
    p = prices[prices["ticker"] == ticker].set_index("date")["adj_close"]
    entry = p.index[p.index > sunday][0]
    exit_ = p.index[p.index > sunday + pd.Timedelta(weeks=1)][0]
    ret = p[exit_] / p[entry] - 1
    print(f"\n  entry {entry.date()} ({entry:%A}) adj close {p[entry]:.4f}")
    print(f"  exit  {exit_.date()} ({exit_:%A}) adj close {p[exit_]:.4f}")
    print(f"  forward 1-week return = {ret:+.4%}")

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    row = panel[(panel["ticker"] == ticker) & (panel["week_end"] == sunday)].iloc[0]
    print("\n  panel says: "
          f"views={int(row['views']):,} asv8={row['asv8']:.4f} entry={row['entry'].date()} "
          f"exit={row['exit'].date()} ret_fwd={row['ret_fwd']:+.4%}")
    assert int(row["views"]) == sums[-1]
    assert np.isclose(row["asv8"], asv8)
    assert row["entry"] == entry and row["exit"] == exit_
    assert np.isclose(row["ret_fwd"], ret)

    week = panel[panel["week_end"] == sunday].sort_values("asv8", ascending=False).reset_index(drop=True)
    n = len(week) // 3
    week["tercile"] = ["High"] * n + ["Middle"] * (len(week) - 2 * n) + ["Low"] * n
    rank = int(week.index[week["ticker"] == ticker][0]) + 1
    print(f"\n  rank {rank} of {len(week)} -> {week.loc[rank - 1, 'tercile']}")
    print("\nMATCH: hand computation equals the panel row.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
