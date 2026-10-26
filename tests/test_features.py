"""Tests for the PySpark feature pipeline: label, window counts and no future leakage."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.features.build import build_features
from atm_cash_optimizer.features.spark_session import get_spark

pytestmark = pytest.mark.spark

MODEL = load_config("model")
DATES = pd.date_range("2024-01-01", periods=60, freq="D")  # starts on a Monday
CAL_DATES = pd.date_range(DATES[0], DATES[-1] + pd.Timedelta(days=MODEL.horizon_max))
POYA = pd.to_datetime(["2024-02-03", "2024-02-10", "2024-02-24"])


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = get_spark("tests")
    yield session
    session.stop()


def _daily(spark: SparkSession, y: np.ndarray) -> DataFrame:
    pdf = pd.DataFrame(
        {"machine_id": "A", "date": DATES, "target": "withdrawals", "y": y, "cluster_id": 0}
    )
    return spark.createDataFrame(pdf).withColumn("date", F.to_date("date"))


def _calendar(spark: SparkSession) -> DataFrame:
    pdf = pd.DataFrame({"date": CAL_DATES, "n_poya": CAL_DATES.isin(POYA).astype(int)})
    return spark.createDataFrame(pdf).withColumn("date", F.to_date("date"))


def _features(spark: SparkSession, y: np.ndarray) -> pd.DataFrame:
    out = build_features(_daily(spark, y), _calendar(spark), MODEL).toPandas()
    out["origin_date"] = pd.to_datetime(out["origin_date"])
    return out.sort_values(["origin_date", "h"], ignore_index=True)


def test_y_cum_is_sum_over_horizon(spark: SparkSession) -> None:
    y = np.arange(1.0, len(DATES) + 1)
    series = pd.Series(y, index=DATES)
    f = _features(spark, y)

    for row in f.itertuples():
        window = pd.date_range(row.origin_date + pd.Timedelta(days=1), periods=row.h)
        if window[-1] > DATES[-1]:
            assert pd.isna(row.y_cum)  # future not in the data yet
        else:
            assert row.y_cum == series[window].sum()
    assert f["origin_date"].min() == DATES[27]  # first origin with 28 days of history


def test_window_counts(spark: SparkSession) -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 150, len(DATES))
    series = pd.Series(y, index=DATES)
    f = _features(spark, y)

    for row in f.itertuples():
        window = pd.date_range(row.origin_date + pd.Timedelta(days=1), periods=row.h)
        last4 = series[row.origin_date - pd.Timedelta(days=27) : row.origin_date]
        weekday_mean = last4.groupby(last4.index.day_of_week).mean()

        assert row.n_poya == window.isin(POYA).sum()
        assert row.n_weekend_days == (window.day_of_week >= 5).sum()
        assert row.expected_by_weekday == pytest.approx(weekday_mean[window.day_of_week].sum())


def test_no_future_leakage(spark: SparkSession) -> None:
    origin = DATES[40]
    y = np.arange(1.0, len(DATES) + 1)
    changed = np.where(DATES > origin, y * 100, y)  # only the future differs

    before = _features(spark, y)
    after = _features(spark, changed)

    def at_origin(f: pd.DataFrame) -> pd.DataFrame:
        return f[f["origin_date"] == origin].drop(columns="y_cum").reset_index(drop=True)

    pd.testing.assert_frame_equal(at_origin(before), at_origin(after))
    assert not before["y_cum"].equals(after["y_cum"])  # the label does see the future
