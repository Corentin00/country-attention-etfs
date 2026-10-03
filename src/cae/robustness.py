"""Step 6: transaction costs (spec section 9), robustness R1-R9 (section 10) and the
decision rule (section 11).

Writes:
  report/results/costs.csv, robustness.csv, leave_one_out.csv, decision.csv
  report/figures/placebo.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cae.config import PROCESSED, ROOT
from cae.stats import WEEKS_PER_YEAR, masked_mean, nw_mean, summary, tercile_membership, wide

RESULTS = ROOT / "report" / "results"
FIGURES = ROOT / "report" / "figures"
COST_DEVELOPED = 0.0010
COST_EMERGING = 0.0020
ENGLISH_MAJORITY = ["EWU", "EWC", "EWA"]  # US, Ireland, New Zealand are not in the universe.
SUB_PERIODS = {
    "2015-09 to 2019-12": ("2015-09-01", "2019-12-31"),
    "2020-01 to 2021-12": ("2020-01-01", "2021-12-31"),
    "2022-01 to 2026-09": ("2022-01-01", "2026-09-30"),
}
N_PLACEBO = 1000
SEED = 20261003


def sort_ls(signal: pd.DataFrame, ret: pd.DataFrame, buckets: int = 3) -> pd.Series:
    """Equal-weighted top-minus-bottom portfolio; buckets=3 terciles, 5 quintiles."""
    n = signal.notna().sum(axis=1)
    k = (n // buckets).to_numpy()[:, None]
    high = signal.rank(axis=1, ascending=False, method="first").le(k)
    low = signal.rank(axis=1, ascending=True, method="first").le(k)
    return masked_mean(ret, high) - masked_mean(ret, low)


def rank_weighted_ls(signal: pd.DataFrame, ret: pd.DataFrame) -> pd.Series:
    """Weights proportional to demeaned rank, scaled to +1 long and -1 short."""
    r = signal.rank(axis=1)
    w = r.sub(r.mean(axis=1), axis=0)
    w = w.div(w.clip(lower=0).sum(axis=1), axis=0)
    return (w * ret).sum(axis=1, min_count=1)


def ls_weights(signal: pd.DataFrame) -> pd.DataFrame:
    high, _, low = tercile_membership(signal)
    return high.div(high.sum(axis=1), axis=0) - low.div(low.sum(axis=1), axis=0)


def costs(panel: pd.DataFrame, gross: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    """Section 9: cost = sum |weight change| x one-way cost (first week: build from zero)."""
    w = ls_weights(wide(panel, "asv8")).astype(float)
    dw = w.diff().abs()
    dw.iloc[0] = w.iloc[0].abs()
    emerging = panel.drop_duplicates("ticker").set_index("ticker")["emerging"]
    one_way = np.where(emerging.reindex(w.columns), COST_EMERGING, COST_DEVELOPED)
    cost = (dw * one_way).sum(axis=1)
    turnover = dw.sum(axis=1)
    net = gross - cost
    mean_gross = gross.mean()
    info = {
        "mean_turnover_per_week": turnover.mean(),
        "mean_cost_weekly_pct": cost.mean() * 100,
        "mean_cost_annual_pct": cost.mean() * WEEKS_PER_YEAR * 100,
        "break_even_one_way_cost_bps": (mean_gross / turnover.mean() * 1e4) if mean_gross > 0 else np.nan,
    }
    return net, info


def placebo(signal: pd.DataFrame, ret: pd.DataFrame, actual: float) -> tuple[float, np.ndarray]:
    """R7: shuffle the signal across countries within each week."""
    rng = np.random.default_rng(SEED)
    values = signal.to_numpy()
    means = np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        shuffled = rng.permuted(values, axis=1)
        means[i] = sort_ls(pd.DataFrame(shuffled, index=signal.index, columns=signal.columns), ret).mean()
    p_value = float((np.abs(means) >= abs(actual)).mean())
    return p_value, means


def row(label: str, ls: pd.Series) -> dict[str, object]:
    s = summary(ls)
    return {"test": label, **{k: s[k] for k in ("weeks", "mean_weekly_pct", "nw_t", "ann_mean_pct", "sharpe_ann")}}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    weekly = pd.read_parquet(PROCESSED / "weekly.parquet").set_index("week_end")
    ret = wide(panel, "ret_fwd")
    asv8 = wide(panel, "asv8")
    main_ls = sort_ls(asv8, ret)

    # Section 9.
    net, cost_info = costs(panel, main_ls)
    cost_table = pd.DataFrame([row("gross", main_ls), row("net of costs", net)]).set_index("test")
    cost_table.to_csv(RESULTS / "costs.csv")

    # Section 10.
    rows = [row("MAIN (asv8, terciles, 1 week)", main_ls)]
    rows += [row(f"R1 window {k} weeks", sort_ls(wide(panel, f"asv{k}"), ret)) for k in (4, 12)]
    rows += [row("R2 quintiles", sort_ls(asv8, ret, buckets=5)),
             row("R2 rank-weighted", rank_weighted_ls(asv8, ret))]
    rows += [row("R3 Economy-of pages", sort_ls(wide(panel, "asv8_econ"), ret))]
    rows += [row(f"R4 {name}", main_ls.loc[a:b]) for name, (a, b) in SUB_PERIODS.items()]
    rows += [row("R6 trade one day later", sort_ls(asv8, wide(panel, "ret_fwd_r6")))]
    fear = weekly["fear_asv8"].reindex(main_ls.index)
    rows += [row("R8 high global fear weeks", main_ls[fear > fear.median()]),
             row("R8 low global fear weeks", main_ls[fear <= fear.median()])]
    keep = [t for t in asv8.columns if t not in ENGLISH_MAJORITY]
    rows += [row("R9 no English-majority countries", sort_ls(asv8[keep], ret[keep]))]
    robustness = pd.DataFrame(rows).set_index("test")

    # R5.
    loo = pd.DataFrame(
        [row(f"without {t}", sort_ls(asv8.drop(columns=t), ret.drop(columns=t))) for t in asv8.columns]
    ).set_index("test").sort_values("nw_t")
    loo.to_csv(RESULTS / "leave_one_out.csv")

    # R7.
    p_value, placebo_means = placebo(asv8, ret, main_ls.mean())
    robustness.loc["R7 placebo p-value (two-sided)", "mean_weekly_pct"] = p_value
    robustness.to_csv(RESULTS / "robustness.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(placebo_means * 100, bins=40, color="lightgray", edgecolor="gray")
    ax.axvline(main_ls.mean() * 100, color="black", lw=2, label=f"actual ({main_ls.mean() * 100:.3f}%)")
    ax.set_xlabel("mean weekly LS return, % (shuffled signal)")
    ax.set_title(f"Placebo: {N_PLACEBO} random signals, p = {p_value:.3f}"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "placebo.png", dpi=150); plt.close(fig)

    # Section 11.
    net_mean, net_t = nw_mean(net)
    sub_signs = [np.sign(main_ls.loc[a:b].mean()) for a, b in SUB_PERIODS.values()]
    main_sign = np.sign(main_ls.mean())
    alpha = pd.read_csv(RESULTS / "alpha_regression.csv", index_col=0)
    be = cost_info["break_even_one_way_cost_bps"]
    decision = pd.DataFrame(
        {
            "condition": [
                "a) net-of-cost NW t > 2 (profitable as specified; |t| of a loss does not count)",
                "b) same sign in >= 2 of 3 sub-periods",
                "c) alpha against market, momentum, reversal and dollar factors significant (|t| > 2)",
                "d) break-even cost > 2 x assumed cost",
            ],
            "value": [
                f"t = {net_t:.2f}",
                f"{sum(s == main_sign for s in sub_signs)} of 3",
                f"t = {alpha.loc['alpha', 'nw_t']:.2f}",
                "n/a (gross mean negative)" if np.isnan(be) else f"{be:.1f} bps",
            ],
            "pass": [
                net_t > 2,
                sum(s == main_sign for s in sub_signs) >= 2,
                abs(alpha.loc["alpha", "nw_t"]) > 2,
                (not np.isnan(be)) and be > 2 * COST_EMERGING * 1e4,
            ],
        }
    )
    decision.to_csv(RESULTS / "decision.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("SECTION 9 - costs")
    print(cost_table.to_string())
    print({k: round(v, 4) for k, v in cost_info.items()}, "\n")
    print("SECTION 10 - robustness")
    print(robustness.to_string(), "\n")
    print("R5 leave one out (sorted by t):")
    print(loo.to_string(), "\n")
    print("SECTION 11 - decision")
    print(decision.to_string(index=False))
    print("\nTRADE IT:", "YES" if decision["pass"].all() else "NO")


if __name__ == "__main__":
    main()
