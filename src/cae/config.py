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

PRICE_START = date(2014, 7, 1)
PRICE_END = date(2026, 10, 2)

# Rule 3.2 and 3.3.
INCEPTION_CUTOFF = date(2015, 7, 1)
LIQUIDITY_WINDOW = (date(2014, 7, 1), date(2015, 6, 30))
MIN_MEDIAN_DOLLAR_VOLUME = 5_000_000.0
