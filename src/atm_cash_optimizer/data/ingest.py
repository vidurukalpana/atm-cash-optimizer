"""Build the ``nn5_daily`` table: download → parse → impute → scale to LKR → validate → write.

Run with::

    python -m atm_cash_optimizer.data.ingest
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.data.download import download_nn5
from atm_cash_optimizer.data.impute import impute_weekday_median
from atm_cash_optimizer.data.tsf import read_tsf
from atm_cash_optimizer.data.validation import validate_nn5_daily
from atm_cash_optimizer.io import write_table

logger = logging.getLogger(__name__)

TABLE_NAME = "nn5_daily"


def build_nn5_daily(tsf_path: str | Path, nn5_unit_lkr: float) -> pd.DataFrame:
    """Parse, impute, scale and validate the NN5 file; return the ``nn5_daily`` table."""
    df = impute_weekday_median(read_tsf(tsf_path))
    df["withdrawals_lkr"] = df["withdrawals_raw"] * nn5_unit_lkr
    df = df[["machine_id", "date", "withdrawals_raw", "withdrawals_lkr", "is_imputed"]]
    return validate_nn5_daily(df)


def run(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Run the full ingestion and write ``nn5_daily`` to Parquet and DuckDB."""
    machines = load_config("machines")
    df = build_nn5_daily(download_nn5(data_dir), machines.nn5_unit_lkr)
    write_table(df, TABLE_NAME, data_dir)
    logger.info(
        "%s: %d machines, %d days each, %s to %s, %d imputed values",
        TABLE_NAME,
        df["machine_id"].nunique(),
        df.groupby("machine_id").size().iloc[0],
        df["date"].min().date(),
        df["date"].max().date(),
        df["is_imputed"].sum(),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run()
