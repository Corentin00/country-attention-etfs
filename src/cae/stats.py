"""Portfolio construction and test statistics shared by all analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

WEEKS_PER_YEAR = 52
NW_LAGS = 4


def wide(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    """week_end x ticker matrix of one panel column."""
    return panel.pivot(index="week_end", columns="ticker", values=column)


def tercile_membership(signal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Boolean High / Middle / Low masks (spec 6.1): top and bottom floor(N/3) by signal."""
    n = signal.notna().sum(axis=1)
    k = (n // 3).to_numpy()[:, None]
    rank_desc = signal.rank(axis=1, ascending=False, method="first")
    rank_asc = signal.rank(axis=1, ascending=True, method="first")
    high = rank_desc.le(k)
    low = rank_asc.le(k)
    middle = signal.notna() & ~high & ~low
    return high, middle, low


def masked_mean(returns: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """Equal-weighted mean of returns over the members of mask, per week."""
    return returns.where(mask).mean(axis=1)


def nw_mean(series: pd.Series, lags: int = NW_LAGS) -> tuple[float, float]:
    """Mean and its Newey-West t-statistic."""
    y = series.dropna().to_numpy()
    fit = sm.OLS(y, np.ones_like(y)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.tvalues[0])


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summary(returns: pd.Series, lags: int = NW_LAGS) -> dict[str, float]:
    r = returns.dropna()
    mean, t = nw_mean(r, lags)
    return {
        "weeks": len(r),
        "mean_weekly_pct": mean * 100,
        "nw_t": t,
        "ann_mean_pct": mean * WEEKS_PER_YEAR * 100,
        "ann_vol_pct": r.std() * np.sqrt(WEEKS_PER_YEAR) * 100,
        "sharpe_ann": mean / r.std() * np.sqrt(WEEKS_PER_YEAR),
        "max_drawdown_pct": max_drawdown(r) * 100,
        "hit_rate_pct": (r > 0).mean() * 100,
    }


def event_time_ls(returns: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, k: int) -> pd.Series:
    """Return in week k after formation (k=1 is the next week) of the High-Low portfolio,
    indexed by formation week."""
    future = returns.shift(-(k - 1))
    return masked_mean(future, high) - masked_mean(future, low)


def overlapping_ls(returns: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, h: int) -> pd.Series:
    """Jegadeesh-Titman: weekly return of holding h cohorts, each formed one week apart,
    each with 1/h of capital. Indexed by the week in which the return is earned
    (labelled by that week's formation date)."""
    cohorts = [event_time_ls(returns, high, low, k).shift(k - 1) for k in range(1, h + 1)]
    return pd.concat(cohorts, axis=1).mean(axis=1, skipna=False)
