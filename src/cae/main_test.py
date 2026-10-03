"""Step 4: main test (spec sections 6 and 7), gross of costs.

Writes:
  report/results/main_summary.csv, main_terciles.csv, horizon_profile.csv, ls_weekly.csv
  report/figures/cumulative_ls.png, cumulative_terciles.png, horizon_profile.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from cae.config import PROCESSED, ROOT
from cae.stats import (
    event_time_ls,
    masked_mean,
    nw_mean,
    overlapping_ls,
    summary,
    tercile_membership,
    wide,
)

RESULTS = ROOT / "report" / "results"
FIGURES = ROOT / "report" / "figures"
MAX_H = 12


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    ret = wide(panel, "ret_fwd")
    high, middle, low = tercile_membership(wide(panel, "asv8"))

    # 7.1-7.3: one-week holding.
    terciles = pd.DataFrame(
        {"High": masked_mean(ret, high), "Middle": masked_mean(ret, middle), "Low": masked_mean(ret, low)}
    )
    ls = (terciles["High"] - terciles["Low"]).rename("LS")
    pd.concat([terciles, ls], axis=1).to_csv(RESULTS / "ls_weekly.csv")

    main_summary = pd.Series(summary(ls), name="LS (High - Low), 1-week hold")
    main_summary.to_csv(RESULTS / "main_summary.csv")
    tercile_table = pd.DataFrame({name: summary(terciles[name]) for name in terciles}).T
    tercile_table.to_csv(RESULTS / "main_terciles.csv")

    # 7.4: horizon profile. Event time k (return earned in week k after formation) and
    # Jegadeesh-Titman overlapping portfolios held h weeks. NW lags = max(4, h) for overlap.
    rows = []
    for k in range(1, MAX_H + 1):
        m_k, t_k = nw_mean(event_time_ls(ret, high, low, k))
        m_h, t_h = nw_mean(overlapping_ls(ret, high, low, k), lags=max(4, k))
        rows.append(
            {"k_or_h": k, "event_week_k_mean_pct": m_k * 100, "event_week_k_t": t_k,
             "overlap_h_mean_weekly_pct": m_h * 100, "overlap_h_t": t_h}
        )
    horizon = pd.DataFrame(rows).set_index("k_or_h")
    horizon["cumulative_event_pct"] = horizon["event_week_k_mean_pct"].cumsum()
    horizon.to_csv(RESULTS / "horizon_profile.csv")

    # Figures.
    fig, ax = plt.subplots(figsize=(10, 4))
    (1 + ls).cumprod().plot(ax=ax, color="black", lw=1)
    ax.axhline(1, color="gray", lw=0.5)
    ax.set_title("High-minus-Low attention, growth of $1 per side (gross, weekly rebalanced)")
    ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(FIGURES / "cumulative_ls.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    (1 + terciles).cumprod().plot(ax=ax, lw=1)
    ax.set_title("Tercile portfolios, growth of $1 (gross, equal-weighted)"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(FIGURES / "cumulative_terciles.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(horizon.index, horizon["event_week_k_mean_pct"], color="lightgray", label="return in week k")
    ax.plot(horizon.index, horizon["cumulative_event_pct"], "o-", color="black", label="cumulative")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("weeks after formation (k)"); ax.set_ylabel("High - Low, % ")
    ax.set_title("Horizon profile of the attention spread"); ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "horizon_profile.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(main_summary.to_string(), "\n")
    print(tercile_table.to_string(), "\n")
    print(horizon.to_string())


if __name__ == "__main__":
    main()
