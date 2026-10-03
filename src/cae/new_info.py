"""Step 5: is attention new information? (spec section 8)

8.1 Fama-MacBeth regressions of next-week return on the ASV rank and controls.
8.2 Alpha of the High-Low portfolio against four factors built from the same universe.

Writes:
  report/results/fama_macbeth.csv, alpha_regression.csv, factors_weekly.csv
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from cae.config import PROCESSED, ROOT
from cae.stats import NW_LAGS, WEEKS_PER_YEAR, masked_mean, nw_mean, tercile_membership, wide

RESULTS = ROOT / "report" / "results"
CONTROLS = ["r1", "r4", "mom", "vol", "lvl"]


def zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1), axis=0)


def fama_macbeth(panel: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    """Weekly cross-sectional OLS of next-week return on the ASV rank and controls,
    then time-series mean of each coefficient with NW t."""
    ret = wide(panel, "ret_fwd")
    x = {"asv_rank": zscore_rows(wide(panel, "asv8").rank(axis=1))}
    x |= {name: zscore_rows(wide(panel, name)) for name in controls}
    names = list(x)
    coefs = []
    for week in ret.index:
        data = pd.concat({"y": ret.loc[week], **{n: x[n].loc[week] for n in names}}, axis=1).dropna()
        fit = sm.OLS(data["y"], sm.add_constant(data[names])).fit()
        coefs.append(fit.params.rename(week))
    coefs = pd.DataFrame(coefs)
    rows = {}
    for name in coefs.columns:
        mean, t = nw_mean(coefs[name])
        rows[name] = {"mean_coef_pct_per_sd": mean * 100, "nw_t": t}
    return pd.DataFrame(rows).T


def factors(panel: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    ret = wide(panel, "ret_fwd")
    rf = weekly.set_index("week_end")["rf"]
    mom_high, _, mom_low = tercile_membership(wide(panel, "mom"))
    r1_high, _, r1_low = tercile_membership(wide(panel, "r1"))
    return pd.DataFrame(
        {
            "MKT": ret.mean(axis=1) - rf,
            "MOM": masked_mean(ret, mom_high) - masked_mean(ret, mom_low),
            "REV": masked_mean(ret, r1_low) - masked_mean(ret, r1_high),
            "USD": weekly.set_index("week_end")["uup_ret"],
        }
    )


def alpha_regression(ls: pd.Series, f: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    data = pd.concat([ls.rename("LS"), f], axis=1).dropna()
    fit = sm.OLS(data["LS"], sm.add_constant(data[f.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS}
    )
    table = pd.DataFrame({"coef": fit.params, "nw_t": fit.tvalues})
    table.loc["const", "coef_annualized_pct"] = fit.params["const"] * WEEKS_PER_YEAR * 100
    table = table.rename(index={"const": "alpha"})
    return table, float(fit.rsquared)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    weekly = pd.read_parquet(PROCESSED / "weekly.parquet")

    # 8.1: baseline (ASV only) and the specified model (ASV + controls).
    fm_base = fama_macbeth(panel, [])
    fm_full = fama_macbeth(panel, CONTROLS)
    fm = pd.concat({"asv_only": fm_base, "with_controls": fm_full}, axis=1)
    fm.to_csv(RESULTS / "fama_macbeth.csv")

    # 8.2
    ls = pd.read_csv(RESULTS / "ls_weekly.csv", index_col=0, parse_dates=True)["LS"]
    f = factors(panel, weekly)
    f.to_csv(RESULTS / "factors_weekly.csv")
    alpha, r2 = alpha_regression(ls, f)
    alpha.to_csv(RESULTS / "alpha_regression.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("8.1 Fama-MacBeth (coef = % weekly return per 1 cross-sectional SD):")
    print(fm.to_string(), "\n")
    print("Factor premia (mean weekly %, NW t):")
    for name in f:
        m, t = nw_mean(f[name])
        print(f"  {name}: {m * 100:.3f}  t={t:.2f}")
    print("\nCorrelation of LS with factors:")
    print(pd.concat([ls.rename("LS"), f], axis=1).corr().loc["LS"].to_string())
    print(f"\n8.2 Alpha regression (R2 = {r2:.3f}):")
    print(alpha.to_string())
    print(f"\n(raw LS mean for comparison: {ls.mean() * 100:.3f}% weekly, "
          f"{ls.mean() * WEEKS_PER_YEAR * 100:.2f}% annualized)")


if __name__ == "__main__":
    main()
