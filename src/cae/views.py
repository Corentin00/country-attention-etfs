"""Step 2: download daily English Wikipedia page views (spec section 2.1).

Pages: the country page of every included ETF, its "Economy of" page (R3),
and the global fear pages (R8).

Writes:
  data/raw/views_daily.parquet   long table: date, title, group, views
"""

from __future__ import annotations

import time
from urllib.parse import quote

import pandas as pd
import requests

from cae.config import (
    FEAR_TITLES,
    PROCESSED,
    RAW,
    VIEWS_END,
    VIEWS_START,
    economy_title,
)

API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{title}/daily/{start}/{end}"
)
# Wikimedia requires a descriptive User-Agent.
HEADERS = {"User-Agent": "country-attention-etfs/0.1 (academic research script)"}
PAUSE_SECONDS = 1.0
MAX_ATTEMPTS = 6


def fetch(title: str, session: requests.Session) -> pd.DataFrame:
    url = API.format(
        title=quote(title, safe=""),
        start=VIEWS_START.strftime("%Y%m%d"),
        end=VIEWS_END.strftime("%Y%m%d"),
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = session.get(url, headers=HEADERS, timeout=60)
        if response.status_code == 429:
            time.sleep(5 * attempt)
            continue
        response.raise_for_status()
        items = response.json()["items"]
        return pd.DataFrame(
            {
                "date": pd.to_datetime([i["timestamp"][:8] for i in items], format="%Y%m%d"),
                "title": title,
                "views": [i["views"] for i in items],
            }
        )
    raise RuntimeError(f"{title}: still rate-limited after {MAX_ATTEMPTS} attempts")


def pages() -> dict[str, str]:
    """title -> group, for every page to download."""
    universe = pd.read_csv(PROCESSED / "universe.csv")
    countries = universe.loc[universe["included"], "wiki_title"].tolist()
    out = {t: "country" for t in countries}
    out |= {economy_title(t): "economy" for t in countries}
    out |= {t: "fear" for t in FEAR_TITLES}
    return out


def completeness(views: pd.DataFrame) -> pd.DataFrame:
    """Per page: number of days present vs. expected, and the missing dates."""
    expected = pd.date_range(VIEWS_START, VIEWS_END, freq="D")
    rows = []
    for title, g in views.groupby("title"):
        missing = expected.difference(pd.DatetimeIndex(g["date"]))
        rows.append(
            {
                "title": title,
                "days": len(g),
                "missing_days": len(missing),
                "first_missing": missing.min().date() if len(missing) else None,
                "median_daily_views": int(g["views"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_days", ascending=False)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    with requests.Session() as session:
        for title, group in pages().items():
            df = fetch(title, session)
            df["group"] = group
            frames.append(df)
            print(f"{group:8s} {title:40s} {len(df):5d} days")
            time.sleep(PAUSE_SECONDS)
    views = pd.concat(frames, ignore_index=True)[["date", "title", "group", "views"]]
    views.to_parquet(RAW / "views_daily.parquet", index=False)
    pd.set_option("display.width", 200)
    print(completeness(views).to_string(index=False))


if __name__ == "__main__":
    main()
