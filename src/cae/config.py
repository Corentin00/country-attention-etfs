"""Fixed parameters from docs/spec_frozen_2026-10-03.txt."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Section 3: candidate iShares MSCI single-country ETFs.
# ticker -> (English Wikipedia title, emerging market per MSCI as of 2015-07-01)
CANDIDATES: dict[str, tuple[str, bool]] = {
    "EWA": ("Australia", False),
    "EWO": ("Austria", False),
    "EWK": ("Belgium", False),
    "EWZ": ("Brazil", True),
    "EWC": ("Canada", False),
    "ECH": ("Chile", True),
    "MCHI": ("China", True),
    "EDEN": ("Denmark", False),
    "EFNL": ("Finland", False),
    "EWQ": ("France", False),
    "EWG": ("Germany", False),
    "EWH": ("Hong_Kong", False),
    "INDA": ("India", True),
    "EIDO": ("Indonesia", True),
    "EIRL": ("Republic_of_Ireland", False),
    "EIS": ("Israel", False),
    "EWI": ("Italy", False),
    "EWJ": ("Japan", False),
    "EWM": ("Malaysia", True),
    "EWW": ("Mexico", True),
    "EWN": ("Netherlands", False),
    "ENZL": ("New_Zealand", False),
    "ENOR": ("Norway", False),
    "EPU": ("Peru", True),
    "EPHE": ("Philippines", True),
    "EPOL": ("Poland", True),
    "QAT": ("Qatar", True),
    "ERUS": ("Russia", True),
    "EWS": ("Singapore", False),
    "EZA": ("South_Africa", True),
    "EWY": ("South_Korea", True),
    "EWP": ("Spain", False),
    "EWD": ("Sweden", False),
    "EWL": ("Switzerland", False),
    "EWT": ("Taiwan", True),
    "THD": ("Thailand", True),
    "TUR": ("Turkey", True),
    "UAE": ("United_Arab_Emirates", True),
    "EWU": ("United_Kingdom", False),
    "EUSA": ("United_States", False),
}

USD_ETF = "UUP"

# R3: "Economy of <country>" pages. Titles that need "the" are listed explicitly.
ECONOMY_TITLE_OVERRIDES: dict[str, str] = {
    "United_Kingdom": "Economy_of_the_United_Kingdom",
    "United_States": "Economy_of_the_United_States",
    "Netherlands": "Economy_of_the_Netherlands",
    "Philippines": "Economy_of_the_Philippines",
    "United_Arab_Emirates": "Economy_of_the_United_Arab_Emirates",
    "Republic_of_Ireland": "Economy_of_the_Republic_of_Ireland",
}


def economy_title(country_title: str) -> str:
    return ECONOMY_TITLE_OVERRIDES.get(country_title, f"Economy_of_{country_title}")


# R8: global fear pages.
FEAR_TITLES: tuple[str, ...] = ("Recession", "Stock_market_crash", "Inflation", "Bank_run")

# Section 2.1.
VIEWS_START = date(2015, 7, 1)
VIEWS_END = date(2026, 9, 30)

PRICE_START = date(2014, 7, 1)
PRICE_END = date(2026, 10, 2)

# Rule 3.2 and 3.3.
INCEPTION_CUTOFF = date(2015, 7, 1)
LIQUIDITY_WINDOW = (date(2014, 7, 1), date(2015, 6, 30))
MIN_MEDIAN_DOLLAR_VOLUME = 5_000_000.0
