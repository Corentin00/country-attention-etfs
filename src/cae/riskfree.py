"""Download the daily risk-free rate from the Ken French Data Library (spec section 2.3).

Writes:
  data/raw/F-F_Research_Data_Factors_daily_CSV.zip   original file, as downloaded
  data/raw/rf_daily.csv                              date, rf (decimal, per trading day)
"""

from __future__ import annotations

import io
import re
import zipfile

import pandas as pd
import requests

from cae.config import RAW

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
ZIP_NAME = "F-F_Research_Data_Factors_daily_CSV.zip"


def parse(zip_bytes: bytes) -> pd.DataFrame:
    """Keep the daily rows (YYYYMMDD,Mkt-RF,SMB,HML,RF in percent); RF in decimal."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = [
        line.split(",")
        for line in text.splitlines()
        if re.match(r"^\s*\d{8},", line)
    ]
    return pd.DataFrame(
        {
            "date": pd.to_datetime([r[0].strip() for r in rows], format="%Y%m%d"),
            "rf": [float(r[4]) / 100 for r in rows],
        }
    )


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    (RAW / ZIP_NAME).write_bytes(response.content)
    rf = parse(response.content)
    rf.to_csv(RAW / "rf_daily.csv", index=False)
    print(f"{len(rf):,} days, {rf['date'].min().date()} to {rf['date'].max().date()}")
    print(rf.tail())


if __name__ == "__main__":
    main()
