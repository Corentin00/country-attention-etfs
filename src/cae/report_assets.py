"""Step 7: figures (PDF), LaTeX tables and number macros for the report.

Everything in report/ that LaTeX reads is generated here from report/results/ and
data/processed/, so no number in the PDF is typed by hand.

Writes:
  report/figures/*.pdf
  report/tables/*.tex
  report/numbers.tex
  report/spec_frozen.txt   (copy of the frozen spec for Appendix A)
"""

from __future__ import annotations

import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cae.config import PROCESSED, ROOT
from cae.new_info import alpha_regression
from cae.robustness import costs, placebo
from cae.stats import masked_mean, nw_mean, summary, tercile_membership, wide

REPORT = ROOT / "report"
RESULTS = REPORT / "results"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"
FULL_WIDTH = 6.7  # inches, A4 with 2 cm margins
# Muted palette, chosen to stay distinct from black: steel blue for the main series,
# terracotta for the contrast series.
BLUE = "#4f7cac"
TERRACOTTA = "#b5523b"

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf"); plt.close(fig)


def fmt(x: float, d: int = 2) -> str:
    """Number with a real minus sign for LaTeX."""
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


# --------------------------------------------------------------------------- figures

def fig_main(ls: pd.Series, net: pd.Series, horizon: pd.DataFrame) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.3), gridspec_kw={"width_ratios": [1.6, 1]})
    (1 + ls).cumprod().plot(ax=a, color="black", lw=0.9, label="gross")
    (1 + net).cumprod().plot(ax=a, color=TERRACOTTA, lw=0.9, label="net of costs")
    a.set_yscale("log"); a.axhline(1, color="gray", lw=0.4)
    a.set_yticks([0.1, 0.2, 0.5, 1.0]); a.set_yticklabels(["0.1", "0.2", "0.5", "1.0"])
    a.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a.set_xlabel(""); a.set_ylabel("growth of \\$1 per side (log)")
    a.set_title("(a) High minus Low, 1-week holding"); a.legend(frameon=False)

    k = horizon.index.to_numpy()
    m = horizon["event_week_k_mean_pct"].to_numpy()
    se = m / horizon["event_week_k_t"].to_numpy()
    b.bar(k, m, color="lightgray", yerr=1.96 * se, error_kw={"lw": 0.6, "capsize": 1.5}, label="week $k$ (95% CI)")
    b.plot(k, horizon["cumulative_event_pct"], "o-", color="black", ms=2.5, lw=0.9, label="cumulative")
    b.axhline(0, color="gray", lw=0.4)
    b.set_xticks(k[::2] if len(k) > 6 else k)
    b.set_xlabel("weeks after formation $k$"); b.set_ylabel("High $-$ Low, %")
    b.set_title("(b) Horizon profile"); b.legend(frameon=False, loc="lower left")
    fig.tight_layout(); save(fig, "fig_main")


def fig_robustness(rob: pd.DataFrame) -> None:
    r = rob.drop(index=[i for i in rob.index if "placebo" in i]).iloc[::-1]
    ann = r["ann_mean_pct"].to_numpy()
    se = ann / r["nw_t"].to_numpy()
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 1.9))
    colors = [TERRACOTTA if lbl.startswith("MAIN") else BLUE for lbl in r.index]
    ax.errorbar(ann, y, xerr=1.96 * se, fmt="none", ecolor="gray", lw=0.7, capsize=1.5)
    ax.scatter(ann, y, c=colors, s=10, zorder=3)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([lbl.replace("MAIN (asv8, terciles, 1 week)", "MAIN") for lbl in r.index])
    ax.set_xlabel("High $-$ Low, annualized % (gross), 95% CI")
    fig.tight_layout(); save(fig, "fig_robustness")


def fig_signal_example(panel: pd.DataFrame) -> None:
    uk = panel[panel["ticker"] == "EWU"].set_index("week_end").loc["2016-03-01":"2016-10-31"]
    baseline = uk["views"] / np.exp(uk["asv8"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.0))
    a.plot(uk.index, uk["views"] / 1e3, "o-", ms=2, lw=0.8, color="black", label="weekly views $V$")
    a.plot(uk.index, baseline / 1e3, lw=1.0, ls="--", color=BLUE, label="median of previous 8 weeks")
    a.set_ylabel("thousand views"); a.set_title("(a) United Kingdom page, 2016"); a.legend(frameon=False)
    b.bar(uk.index, uk["asv8"], width=5, color=BLUE)
    b.axhline(0, color="black", lw=0.4)
    b.set_title("(b) ASV = ln($V$) $-$ ln(median)"); b.set_ylabel("ASV")
    for ax in (a, b):
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); save(fig, "fig_signal_example")


def fig_terciles(terciles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.2))
    for name, color in [("High", TERRACOTTA), ("Middle", "gray"), ("Low", BLUE)]:
        (1 + terciles[name]).cumprod().plot(ax=ax, lw=0.9, color=color, label=f"{name} attention")
    ax.set_xlabel(""); ax.set_ylabel("growth of \\$1"); ax.legend(frameon=False)
    fig.tight_layout(); save(fig, "fig_terciles")


def fig_placebo(placebo_means: np.ndarray, actual: float, p_value: float) -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.6, 2.0))
    ax.hist(placebo_means * 100, bins=40, color="lightgray", edgecolor="gray", lw=0.4)
    ax.axvline(actual * 100, color="black", lw=1.2, label=f"actual ({actual * 100:.3f}%)")
    ax.set_xlabel("mean weekly High $-$ Low, % (shuffled signal)"); ax.set_ylabel("count")
    ax.set_title(f"1,000 placebo signals, two-sided $p$ = {p_value:.3f}"); ax.legend(frameon=False)
    fig.tight_layout(); save(fig, "fig_placebo")


def fig_liquidity(universe: pd.DataFrame) -> None:
    # ETFs without price data have unknown (not zero) volume and cannot sit on a log axis.
    u = universe[universe["exclusion_reason"] != "no Yahoo data"]
    u = u.sort_values("median_dollar_volume_2014_15")
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.75, 5.0))
    ax.barh(u["ticker"] + " " + u["wiki_title"].str.replace("_", " "),
            u["median_dollar_volume_2014_15"] / 1e6,
            color=np.where(u["included"], BLUE, "lightgray"))
    ax.axvline(5, color="black", ls="--", lw=0.8, label="\\$5M threshold")
    ax.set_xscale("log"); ax.set_xlabel("median daily dollar volume, Jul 2014 $-$ Jun 2015, \\$M (log)")
    ax.tick_params(axis="y", labelsize=6); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, "fig_liquidity")


# --------------------------------------------------------------------------- tables

def write(name: str, body: str) -> None:
    (TABLES / f"{name}.tex").write_text(body, encoding="utf-8")


def tab_main(terciles: pd.DataFrame, ls: pd.Series, net: pd.Series) -> None:
    series = {"High": terciles["High"], "Middle": terciles["Middle"], "Low": terciles["Low"],
              "High $-$ Low (gross)": ls, "High $-$ Low (net)": net}
    lines = []
    for name, s in series.items():
        x = summary(s)
        lines.append(f"{name} & {fmt(x['mean_weekly_pct'], 3)} & {fmt(x['ann_mean_pct'], 1)} & "
                     f"{fmt(x['nw_t'])} & {fmt(x['ann_vol_pct'], 1)} & {fmt(x['sharpe_ann'])} & "
                     f"{fmt(x['max_drawdown_pct'], 0)} & {fmt(x['hit_rate_pct'], 0)} \\\\")
        if name == "Low":
            lines.append("\\midrule")
    write("tab_main", "\n".join([
        "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        " & Mean/wk & Ann. & NW $t$ & Vol & Sharpe & Max DD & Hit \\\\",
        " & \\% & \\% & & \\% & & \\% & \\% \\\\", "\\midrule", *lines, "\\bottomrule", "\\end{tabular}",
    ]))


def tab_newinfo(fm: pd.DataFrame, alpha: pd.DataFrame, r2: float) -> None:
    labels = {"asv_rank": "ASV rank", "r1": "Return, last week", "r4": "Return, weeks $-4$ to $-2$",
              "mom": "Return, weeks $-52$ to $-5$", "vol": "Volatility (26 weeks)", "lvl": "Baseline attention level"}
    fm_lines = []
    for key, label in labels.items():
        a = fm.loc[key, ("asv_only", "mean_coef_pct_per_sd")] if key == "asv_rank" else np.nan
        at = fm.loc[key, ("asv_only", "nw_t")] if key == "asv_rank" else np.nan
        c, ct = fm.loc[key, ("with_controls", "mean_coef_pct_per_sd")], fm.loc[key, ("with_controls", "nw_t")]
        left = f"{fmt(a, 3)} & ({fmt(at)})" if not np.isnan(a) else " & "
        fm_lines.append(f"{label} & {left} & {fmt(c, 3)} & ({fmt(ct)}) \\\\")
    al = {"alpha": "Alpha (\\% p.a.)", "MKT": "Market", "MOM": "Momentum", "REV": "Short-term reversal", "USD": "US dollar"}
    a_lines = []
    for key, label in al.items():
        v = alpha.loc[key, "coef_annualized_pct"] if key == "alpha" else alpha.loc[key, "coef"]
        a_lines.append(f"{label} & {fmt(v, 2)} & ({fmt(alpha.loc[key, 'nw_t'])}) \\\\")
    write("tab_newinfo", "\n".join([
        "\\begin{tabular}{lrrrr}", "\\toprule",
        "\\multicolumn{5}{l}{\\textit{Panel A: Fama--MacBeth, next-week return (\\% per cross-sectional s.d.)}} \\\\",
        " & \\multicolumn{2}{c}{ASV only} & \\multicolumn{2}{c}{With controls} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", *fm_lines, "\\midrule",
        "\\multicolumn{5}{l}{\\textit{Panel B: High $-$ Low on four factors from the same universe}} \\\\",
        " & Coef. & ($t$) & & \\\\", *[l.replace(" \\\\", " & & \\\\") for l in a_lines],
        f"$R^2$ & {r2:.3f} & & & \\\\", "\\bottomrule", "\\end{tabular}",
    ]))


def tab_robustness(rob: pd.DataFrame) -> None:
    lines = []
    for name, x in rob.iterrows():
        if "placebo" in name:
            continue
        lines.append(f"{name.replace('MAIN (asv8, terciles, 1 week)', 'Main specification')} & {int(x['weeks'])} & "
                     f"{fmt(x['mean_weekly_pct'], 3)} & {fmt(x['ann_mean_pct'], 1)} & {fmt(x['nw_t'])} & "
                     f"{fmt(x['sharpe_ann'])} \\\\")
    write("tab_robustness", "\n".join([
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Test & Weeks & Mean/wk \\% & Ann. \\% & NW $t$ & Sharpe \\\\", "\\midrule",
        *lines, "\\bottomrule", "\\end{tabular}",
    ]))


def tab_loo(loo: pd.DataFrame, names: dict[str, str]) -> None:
    lines = [f"{t.replace('without ', '')} ({names[t.replace('without ', '')]}) & {fmt(x['ann_mean_pct'], 1)} & {fmt(x['nw_t'])} \\\\"
             for t, x in loo.iterrows()]
    half = (len(lines) + 1) // 2
    left, right = lines[:half], lines[half:] + [" & & \\\\"] * (2 * half - len(lines))
    rows = [f"{lft.removesuffix(' \\\\')} & {rgt}" for lft, rgt in zip(left, right, strict=True)]
    write("tab_loo", "\n".join([
        "\\begin{tabular}{lrr@{\\hspace{2em}}lrr}", "\\toprule",
        "Dropped ETF & Ann. \\% & NW $t$ & Dropped ETF & Ann. \\% & NW $t$ \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}",
    ]))


def volume_cell(r: object) -> str:
    """Dollar volume in $M, or n/a when there is no price data (unknown, not zero)."""
    if r.exclusion_reason == "no Yahoo data":
        return "n/a"
    return f"{r.median_dollar_volume_2014_15 / 1e6:.2f}"


def tab_universe(universe: pd.DataFrame) -> None:
    u = universe.sort_values("median_dollar_volume_2014_15", ascending=False)
    lines = [f"{r.ticker} & {r.wiki_title.replace('_', ' ')} & {'EM' if r.emerging else 'DM'} & "
             f"{volume_cell(r)} & {'yes' if r.included else r.exclusion_reason} \\\\"
             for r in u.itertuples()]
    write("tab_universe", "\n".join([
        "\\begin{tabular}{llcrl}", "\\toprule",
        "ETF & Wikipedia page & Market & \\$M/day & Included \\\\", "\\midrule",
        *lines, "\\bottomrule", "\\end{tabular}",
    ]))


def tab_decision(decision: pd.DataFrame) -> None:
    lines = [f"{r.condition.split(' (')[0].replace('>=', r'$\geq$').replace('> ', '$>$ ').replace('|t|', '$|t|$').replace(' t ', ' $t$ ')} & "
             f"{r.value.replace('t = -', 't = $-$')} & {'pass' if r._3 else 'fail'} \\\\"
             for r in decision.itertuples()]
    write("tab_decision", "\n".join([
        "\\begin{tabular}{p{0.55\\textwidth}lc}", "\\toprule", "Condition (frozen spec, section 11) & Value & Result \\\\",
        "\\midrule", *lines, "\\bottomrule", "\\end{tabular}",
    ]))


def tab_horizon(horizon: pd.DataFrame, week0: tuple[float, float]) -> None:
    """Horizon profile: event-time returns (Figure 1b) and the overlapping
    Jegadeesh-Titman portfolios of spec 7.4, plus the exploratory week 0."""
    m0, t0 = week0
    lines = [f"0 (signal week) & {fmt(m0 * 100, 3)} & ({fmt(t0)}) & & \\\\", "\\midrule"]
    for k, r in horizon.iterrows():
        lines.append(
            f"{k} & {fmt(r['event_week_k_mean_pct'], 3)} & ({fmt(r['event_week_k_t'])}) & "
            f"{fmt(r['overlap_h_mean_weekly_pct'], 3)} & ({fmt(r['overlap_h_t'])}) \\\\"
        )
    write("tab_horizon", "\n".join([
        "\\begin{tabular}{lrrrr}", "\\toprule",
        " & \\multicolumn{2}{c}{Event time: week $k$} & \\multicolumn{2}{c}{Overlapping, held $h$ weeks} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "$k$ or $h$ & Mean \\% & ($t$) & Mean \\%/wk & ($t$) \\\\", "\\midrule",
        *lines, "\\bottomrule", "\\end{tabular}",
    ]))


def week0_spread(panel: pd.DataFrame) -> tuple[float, float]:
    """Exploratory: High-minus-Low return during the signal week itself (not in the spec)."""
    high, _, low = tercile_membership(wide(panel, "asv8"))
    r0 = wide(panel, "r1")  # return from the previous rebalance close to this one
    return nw_mean(masked_mean(r0, high) - masked_mean(r0, low))


def spec_appendix(spec_text: str) -> str:
    """Appendix A: the frozen spec, one framed box per section, text unchanged.

    The spec separates sections with a title line followed by a line of dashes; the
    header is enclosed in lines of '='. Those decorative lines become box titles; every
    other line is reproduced verbatim, and no box breaks across pages.
    """
    lines = spec_text.rstrip("\n").split("\n")
    sections: list[tuple[str, list[str]]] = []
    title, body = "Header", []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if nxt.startswith("-----") and line.strip():
            sections.append((title, body))
            title, body = line.strip(), []
            i += 2
            continue
        if not line.startswith("====="):
            body.append(line)
        i += 1
    sections.append((title, body))

    out = []
    for title, body in sections:
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if not body:
            continue
        out.append(
            "\\begin{Verbatim}[fontsize=\\footnotesize, frame=single, framerule=0.3pt,"
            " rulecolor=\\color{black!35}, framesep=2.5mm, samepage=true,"
            f" label={{\\textcolor{{black}}{{\\textbf{{{title}}}}}}}]"
        )
        out.extend(body)
        out.append("\\end{Verbatim}")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- numbers

def numbers(values: dict[str, str]) -> None:
    """Number macros; negatives are wrapped in \\ensuremath so they work in text and math."""

    def safe(v: str) -> str:
        return "\\ensuremath{-" + v.removeprefix("$-$") + "}" if v.startswith("$-$") else v

    (REPORT / "numbers.tex").write_text(
        "% Generated by cae.report_assets - do not edit by hand.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{safe(v)}}}" for k, v in values.items()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for d in (FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    universe = pd.read_csv(PROCESSED / "universe.csv")
    weekly_ls = pd.read_csv(RESULTS / "ls_weekly.csv", index_col=0, parse_dates=True)
    terciles, ls = weekly_ls[["High", "Middle", "Low"]], weekly_ls["LS"]
    net, cost_info = costs(panel, ls)
    horizon = pd.read_csv(RESULTS / "horizon_profile.csv", index_col=0)
    rob = pd.read_csv(RESULTS / "robustness.csv", index_col=0)
    loo = pd.read_csv(RESULTS / "leave_one_out.csv", index_col=0)
    fm = pd.read_csv(RESULTS / "fama_macbeth.csv", header=[0, 1], index_col=0)
    alpha = pd.read_csv(RESULTS / "alpha_regression.csv", index_col=0)
    decision = pd.read_csv(RESULTS / "decision.csv")

    # R7 placebo distribution, recomputed with the same seed as cae.robustness.
    p_value, placebo_means = placebo(wide(panel, "asv8"), wide(panel, "ret_fwd"), ls.mean())

    fig_main(ls, net, horizon)
    fig_robustness(rob)
    fig_signal_example(panel)
    fig_terciles(terciles)
    fig_placebo(placebo_means, ls.mean(), p_value)
    fig_liquidity(universe)

    factors = pd.read_csv(RESULTS / "factors_weekly.csv", index_col=0, parse_dates=True)
    _, r2_alpha = alpha_regression(ls, factors)
    tab_main(terciles, ls, net)
    tab_newinfo(fm, alpha, r2_alpha)
    tab_robustness(rob)
    tab_loo(loo, dict(zip(universe["ticker"], universe["wiki_title"].str.replace("_", " "), strict=True)))
    tab_universe(universe)
    tab_decision(decision)
    week0 = week0_spread(panel)
    tab_horizon(horizon, week0)
    variations = rob.drop(index=[i for i in rob.index if 'placebo' in i or i.startswith('MAIN')])

    g, n = summary(ls), summary(net)
    yearly = ls.groupby(ls.index.year).apply(lambda r: (1 + r).prod() - 1)
    wealth = (1 + ls).cumprod()
    trough = (wealth / wealth.cummax() - 1).idxmin()
    turnover = cost_info["mean_turnover_per_week"]
    uk = panel[(panel["ticker"] == "EWU") & (panel["week_end"] == "2016-06-26")]
    brexit_asv = float(uk["asv8"].iloc[0])
    inc = universe[universe["included"]]
    numbers({
        "NWeeks": str(g["weeks"]), "NEtfs": str(len(inc)), "NCandidates": str(len(universe)),
        "NDeveloped": str(int((~inc["emerging"]).sum())), "NEmerging": str(int(inc["emerging"].sum())),
        "LSWeekly": fmt(g["mean_weekly_pct"], 3), "LSAnn": fmt(g["ann_mean_pct"], 1), "LSt": fmt(g["nw_t"]),
        "LSSharpe": fmt(g["sharpe_ann"]), "NetAnn": fmt(n["ann_mean_pct"], 1), "Nett": fmt(n["nw_t"]),
        "LSLoss": fmt(-g["ann_mean_pct"], 1), "NetLoss": fmt(-n["ann_mean_pct"], 1),
        # Each side holds $1, so replacing a whole side moves sum|dw| by 2: total max is 4.
        "TurnoverPerSide": f"{cost_info['mean_turnover_per_week'] / 4 * 100:.0f}",
        "HoldWeeks": f"{4 / cost_info['mean_turnover_per_week']:.1f}",
        "CostAnn": f"{cost_info['mean_cost_annual_pct']:.1f}",
        "HighAnn": fmt(summary(terciles['High'])["ann_mean_pct"], 1),
        "MiddleAnn": fmt(summary(terciles['Middle'])["ann_mean_pct"], 1),
        "LowAnn": fmt(summary(terciles['Low'])["ann_mean_pct"], 1),
        "FMasv": fmt(fm.loc["asv_rank", ("asv_only", "mean_coef_pct_per_sd")], 3),
        "FMasvt": fmt(fm.loc["asv_rank", ("asv_only", "nw_t")]),
        "FMasvc": fmt(fm.loc["asv_rank", ("with_controls", "mean_coef_pct_per_sd")], 3),
        "FMasvct": fmt(fm.loc["asv_rank", ("with_controls", "nw_t")]),
        "AlphaAnn": fmt(alpha.loc["alpha", "coef_annualized_pct"], 1), "Alphat": fmt(alpha.loc["alpha", "nw_t"]),
        "AlphaRsq": f"{r2_alpha * 100:.1f}",
        "PlaceboP": f"{p_value:.2f}", "PlaceboPct": f"{p_value * 100:.0f}",
        "RoneFourt": fmt(rob.loc["R1 window 4 weeks", "nw_t"]),
        **{f"Prem{name}": fmt(nw_mean(factors[name])[0] * 100, 3) for name in factors},
        **{f"Prem{name}t": fmt(nw_mean(factors[name])[1]) for name in factors},
        "LOOtMin": fmt(loo["nw_t"].min()), "LOOtMax": fmt(loo["nw_t"].max()),
        "RobNegative": str(int((variations["mean_weekly_pct"] < 0).sum())),
        "RobTotal": str(len(variations)),
        "WeekZero": fmt(week0[0] * 100, 3), "WeekZerot": fmt(week0[1]),
        "LossTwentyTwo": fmt(-yearly.loc[2022] * 100, 1), "LossTwentyThree": fmt(-yearly.loc[2023] * 100, 1),
        "TroughDate": f"{trough:%B %Y}",
        "TurnoverTotal": f"{turnover:.2f}", "CostMultiplier": f"{turnover * 52:.0f}",
        "BreakEvenBps": f"{abs(ls.mean()) / turnover * 1e4:.1f}",
        "BrexitASV": f"{brexit_asv:.1f}", "BrexitRatio": f"{np.exp(brexit_asv):.0f}",
    })
    shutil.copy(ROOT / "docs" / "spec_frozen_2026-10-03.txt", REPORT / "spec_frozen.txt")
    spec = (ROOT / "docs" / "spec_frozen_2026-10-03.txt").read_text(encoding="utf-8")
    write("spec_appendix", spec_appendix(spec))
    print("Wrote", sorted(p.name for p in FIGURES.glob("*.pdf")), sorted(p.name for p in TABLES.glob("*.tex")))
    print((REPORT / "numbers.tex").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
