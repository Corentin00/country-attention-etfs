"""Step 3: build the weekly panel (spec sections 4, 5 and 8.1).

Weeks run Monday 00:00 to Sunday 24:00 UTC and are labelled by their Sunday (`week_end`).
For signal week w, the position is opened at the close of the first US trading day after
`week_end` (`entry`) and closed at the next week's entry (`exit`).

Writes:
  data/processed/panel.parquet    one row per (week_end, ticker)
  data/processed/weekly.parquet   one row per week_end: dates, rf, UUP return, fear index
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cae.config import FEAR_TITLES, PROCESSED, RAW, USD_ETF, economy_title

ASV_WINDOWS = (4, 8, 12)  # 8 is the main spec; 4 and 12 are R1.
FIRST_SIGNAL_WEEK = pd.Timestamp("2015-09-06")
LAST_SIGNAL_WEEK = pd.Timestamp("2026-09-20")


def weekly_views(views: pd.DataFrame) -> pd.DataFrame:
    """Weekly view sums (week_end x title); NaN unless all 7 days are present (rule 4.4)."""
    g = views.set_index("date").groupby("title")["views"].resample("W-SUN")
    total, days = g.sum().unstack(0), g.size().unstack(0)
    return total.where(days.eq(7))


def asv(weekly: pd.DataFrame, window: int) -> pd.DataFrame:
    """ln V(w) - ln median(V(w-1..w-window)); NaN if any of those weeks is missing."""
    baseline = weekly.shift(1).rolling(window, min_periods=window).median()
    return np.log(weekly) - np.log(baseline)


def trading_days(adj: pd.DataFrame) -> pd.DatetimeIndex:
    """US trading days: dates on which at least 90% of the ETFs have a price."""
    return adj.index[adj.notna().mean(axis=1) >= 0.9]


def schedule(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Entry and exit dates per signal week (main timing and R6 one-day delay)."""
    week_ends = pd.date_range(days[0] - pd.Timedelta(days=7), days[-1], freq="W-SUN")
    first = days.searchsorted(week_ends, side="right")
    ok = first + 1 < len(days)
    s = pd.DataFrame(index=week_ends[ok])
    s.index.name = "week_end"
    s["entry"] = days[first[ok]]
    s["entry_r6"] = days[first[ok] + 1]
    s["exit"] = s["entry"].shift(-1)
    s["exit_r6"] = s["entry_r6"].shift(-1)
    return s


def price_on(adj: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Adjusted prices at each week's given date (week_end x ticker)."""
    out = adj.reindex(dates.to_numpy())
    out.index = dates.index
    return out


def weekly_rf(rf_daily: pd.Series, entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Compounded daily RF over (entry, exit]."""
    growth = (1 + rf_daily).cumprod()
    start = growth.reindex(entry.to_numpy(), method="ffill").to_numpy()
    end = growth.reindex(exit_.to_numpy(), method="ffill").to_numpy()
    return pd.Series(end / start - 1, index=entry.index)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = pd.read_csv(PROCESSED / "universe.csv")
    universe = universe[universe["included"]]
    tickers = universe["ticker"].tolist()
    title_of = dict(zip(universe["ticker"], universe["wiki_title"], strict=True))

    prices = pd.read_parquet(RAW / "prices_daily.parquet")
    adj_all = prices.pivot(index="date", columns="ticker", values="adj_close")
    adj = adj_all[tickers]
    days = trading_days(adj)
    adj = adj.loc[days]
    sched = schedule(days)

    # Ken French RF ends before the price data; carry the last value forward.
    rf = pd.read_csv(RAW / "rf_daily.csv", parse_dates=["date"]).set_index("date")["rf"]
    rf_daily = rf.reindex(days).ffill()

    # Views -> ASV per window, for country, economy and fear pages.
    weekly = weekly_views(pd.read_parquet(RAW / "views_daily.parquet"))
    asvs = {k: asv(weekly, k) for k in ASV_WINDOWS}
    country_titles = [title_of[t] for t in tickers]
    econ_titles = [economy_title(title_of[t]) for t in tickers]

    def by_ticker(frame: pd.DataFrame, titles: list[str]) -> pd.DataFrame:
        out = frame[titles].copy()
        out.columns = tickers
        return out

    # Returns and controls, all as (week_end x ticker).
    p_entry = price_on(adj, sched["entry"])
    p_exit = price_on(adj, sched["exit"])
    p_entry_r6 = price_on(adj, sched["entry_r6"])
    p_exit_r6 = price_on(adj, sched["exit_r6"])
    p_lag = {k: p_entry.shift(k) for k in (1, 4, 52)}
    daily_log_ret = np.log(adj).diff()
    vol = daily_log_ret.rolling("182D", min_periods=100).std()

    fields: dict[str, pd.DataFrame] = {
        "views": by_ticker(weekly, country_titles),
        **{f"asv{k}": by_ticker(asvs[k], country_titles) for k in ASV_WINDOWS},
        "asv8_econ": by_ticker(asvs[8], econ_titles),
        "lvl": np.log(by_ticker(weekly.shift(1).rolling(8, min_periods=8).median(), country_titles)),
        "ret_fwd": p_exit / p_entry - 1,
        "ret_fwd_r6": p_exit_r6 / p_entry_r6 - 1,
        "r1": p_entry / p_lag[1] - 1,
        "r4": p_lag[1] / p_lag[4] - 1,
        "mom": p_lag[4] / p_lag[52] - 1,
        "vol": price_on(vol, sched["entry"]),
    }
    weeks = sched.index[(sched.index >= FIRST_SIGNAL_WEEK) & (sched.index <= LAST_SIGNAL_WEEK)]
    panel = pd.concat(
        {name: f.reindex(weeks).stack(future_stack=True) for name, f in fields.items()}, axis=1
    )
    panel.index.names = ["week_end", "ticker"]
    panel = panel.reset_index()
    panel["title"] = panel["ticker"].map(title_of)
    panel["emerging"] = panel["ticker"].map(dict(zip(universe["ticker"], universe["emerging"], strict=True)))
    panel = panel.merge(sched.reset_index(), on="week_end", how="left")

    uup = adj_all[USD_ETF].reindex(days)
    wk = sched.loc[weeks].copy()
    wk["rf"] = weekly_rf(rf_daily, wk["entry"], wk["exit"])
    wk["uup_ret"] = (price_on(uup.to_frame(), wk["exit"]) / price_on(uup.to_frame(), wk["entry"]) - 1)[USD_ETF]
    wk["fear_asv8"] = asvs[8][list(FEAR_TITLES)].mean(axis=1).reindex(weeks)
    wk["n_ranked"] = panel.groupby("week_end")["asv8"].count()
    return panel, wk.reset_index()


def main() -> None:
    panel, weekly = build()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED / "panel.parquet", index=False)
    weekly.to_parquet(PROCESSED / "weekly.parquet", index=False)
    pd.set_option("display.width", 200)
    print(f"panel: {len(panel):,} rows, {panel['week_end'].nunique()} weeks, {panel['ticker'].nunique()} ETFs")
    print(f"weeks {weekly['week_end'].min().date()} to {weekly['week_end'].max().date()}")
    print("\nMissing values per column:")
    print(panel.isna().sum().to_string())
    print("\nWeekly table:")
    print(weekly.isna().sum().to_string())
    print(weekly.head(3).to_string(index=False))
    print(weekly.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
