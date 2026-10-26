"""Tests for the data-ingestion branch: tsf parser, imputation, schema and table I/O."""

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from atm_cash_optimizer.data.impute import impute_weekday_median
from atm_cash_optimizer.data.ingest import build_nn5_daily
from atm_cash_optimizer.data.tsf import read_tsf
from atm_cash_optimizer.data.validation import validate_nn5_daily
from atm_cash_optimizer.io import read_table, write_table

MINI_TSF = Path(__file__).parent / "fixtures" / "mini.tsf"


def test_read_tsf() -> None:
    df = read_tsf(MINI_TSF)

    assert list(df.columns) == ["machine_id", "date", "withdrawals_raw"]
    assert df["machine_id"].unique().tolist() == ["NN5_001", "NN5_002"]
    assert df.groupby("machine_id").size().tolist() == [21, 21]
    assert df["date"].min() == pd.Timestamp("1996-03-18")
    assert df["withdrawals_raw"].isna().sum() == 3  # the three "?" values


def test_impute_uses_same_weekday_median() -> None:
    df = impute_weekday_median(read_tsf(MINI_TSF))
    t1 = df[df["machine_id"] == "NN5_001"].set_index("date")

    # Monday 25 March is missing; the Mondays a week before (=1) and after (=15) are known.
    assert t1.loc["1996-03-25", "withdrawals_raw"] == 8
    assert t1["is_imputed"].sum() == 1
    assert df["withdrawals_raw"].notna().all()


def test_impute_falls_back_to_series_weekday_median() -> None:
    # With a 1-week window, NN5_002's first Monday only sees the next Monday, which is also
    # missing, so it falls back to the median of all its Mondays (= 30).
    df = impute_weekday_median(read_tsf(MINI_TSF), window_weeks=1)
    t2 = df[df["machine_id"] == "NN5_002"].set_index("date")

    assert t2.loc["1996-03-18", "withdrawals_raw"] == 30


def _corrupt(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "duplicate-key":
        return pd.concat([df, df.iloc[[0]]], ignore_index=True)
    if rule == "date-gap":
        return df.drop(index=5)
    if rule == "negative":
        return df.assign(withdrawals_lkr=df["withdrawals_lkr"].where(df.index != 0, -1.0))
    if rule == "null":
        return df.assign(withdrawals_lkr=df["withdrawals_lkr"].where(df.index != 0))
    if rule == "unequal-length":
        return df.drop(index=df.index[-1])
    raise ValueError(rule)


@pytest.mark.parametrize(
    "rule", ["duplicate-key", "date-gap", "negative", "null", "unequal-length"]
)
def test_schema_rejects_bad_data(rule: str) -> None:
    good = build_nn5_daily(MINI_TSF, nn5_unit_lkr=1000)

    with pytest.raises(SchemaErrors):
        validate_nn5_daily(_corrupt(good, rule))


def test_write_and_read_table(tmp_path: Path) -> None:
    df = build_nn5_daily(MINI_TSF, nn5_unit_lkr=1000)

    write_table(df, "nn5_daily", data_dir=tmp_path)

    pd.testing.assert_frame_equal(read_table("nn5_daily", data_dir=tmp_path), df)
    assert (tmp_path / "warehouse.duckdb").is_file()
