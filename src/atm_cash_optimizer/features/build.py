"""Origin–horizon training rows for the forecasting models (PySpark).

For every machine, target series, origin date ``o`` and horizon ``h = 1..horizon_max`` there is
one row with:

- **origin features** — only data with ``date <= o`` (no future leakage):
  ``y_lag_<k>`` = demand ``k`` days before ``o+1`` (``y_lag_1`` = demand on ``o``),
  ``y_roll_mean_<n>`` / ``y_roll_std_<n>`` over the last ``n`` days,
  ``y_wd_mean_<d>`` = mean demand on weekday ``d`` (0 = Monday) over the last
  ``weekday_mean_weeks`` weeks, ``origin_weekday``, ``origin_month``;
- **window features** over ``o+1 .. o+h`` — calendar facts known in advance:
  ``n_weekend_days``, ``expected_by_weekday`` and the calendar counts (``n_poya``,
  ``n_avurudu_days``, ...; all 0 for ``nn5``);
- **machine features** — ``cluster_id``, ``location_type``, ``machine_type``;
- **label** ``y_cum`` = total demand over ``o+1 .. o+h`` (null when ``o+h`` is past the data).

Targets: ``withdrawals`` (all machines) and ``net`` = withdrawals − deposits (CRMs, ``lk``).

Run with::

    python -m atm_cash_optimizer.features.build            # both tracks
    python -m atm_cash_optimizer.features.build --track lk
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from functools import reduce

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from atm_cash_optimizer.config import ModelConfig, ScenarioConfig, load_config
from atm_cash_optimizer.features.spark_session import get_spark
from atm_cash_optimizer.io import get_data_dir, register_parquet_view, table_path
from atm_cash_optimizer.scenario.calendar import build_lk_calendar

logger = logging.getLogger(__name__)

TRACKS = ("nn5", "lk")
KEYS = ["machine_id", "target"]
WEEKDAYS = range(7)  # 0 = Monday … 6 = Sunday
WEEKEND = (5, 6)

# lk_calendar flag → window-count feature (from the plan)
CALENDAR_COUNTS = {
    "is_public_holiday": "n_public_holidays",
    "is_poya": "n_poya",
    "is_salary_week": "n_salary_days",
    "is_long_weekend_eve": "n_long_weekend_eves",
    "is_no_service_day": "n_no_service_days",
}


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


def load_daily(spark: SparkSession, track: str) -> DataFrame:
    """Long table ``machine_id, date, target, y, cluster_id, location_type, machine_type``."""
    if track == "nn5":
        clusters = spark.read.parquet(str(table_path("machine_clusters")))
        return (
            spark.read.parquet(str(table_path("nn5_daily")))
            .join(clusters.select("machine_id", "cluster_id"), "machine_id")
            .select(
                "machine_id",
                "date",
                F.lit("withdrawals").alias("target"),
                F.col("withdrawals_lkr").alias("y"),
                "cluster_id",
                F.lit(None).cast("string").alias("location_type"),
                F.lit("ATM").alias("machine_type"),
            )
        )

    lk = spark.read.parquet(str(table_path("lk_daily")))
    static = ["cluster_id", "location_type", "machine_type"]
    withdrawals = lk.select(
        "machine_id",
        "date",
        F.lit("withdrawals").alias("target"),
        F.col("withdrawals_lkr").alias("y"),
        *static,
    )
    net = lk.where(F.col("machine_type") == "CRM").select(
        "machine_id",
        "date",
        F.lit("net").alias("target"),
        (F.col("withdrawals_lkr") - F.col("deposits_lkr")).alias("y"),
        *static,
    )
    return withdrawals.unionByName(net)


def calendar_indicators(
    track: str, start: pd.Timestamp, end: pd.Timestamp, scenario: ScenarioConfig
) -> pd.DataFrame:
    """Daily 0/1 indicators, one column per window-count feature, from ``start`` to ``end``.

    ``lk``: from the Sri Lankan calendar, plus ``n_<event>_days`` for every festival in
    ``scenario.yaml`` and ``n_<event>_after_days`` for festivals with windows after the
    holiday (e.g. the quiet days after Avurudu). ``nn5``: the same columns, all 0.
    """
    dates = pd.date_range(start, end, freq="D")
    after_events = [e for e, cfg in scenario.events.items() if any(w.end > 0 for w in cfg.windows)]
    columns = [
        *CALENDAR_COUNTS.values(),
        *[f"n_{e}_days" for e in scenario.events],
        *[f"n_{e}_after_days" for e in after_events],
    ]
    if track == "nn5":
        return pd.DataFrame({"date": dates, **{c: 0 for c in columns}})

    cal = build_lk_calendar(start.date(), end.date(), scenario)
    out = pd.DataFrame({"date": cal["date"]})
    for flag, feature in CALENDAR_COUNTS.items():
        out[feature] = cal[flag].astype(int)
    for event in scenario.events:
        out[f"n_{event}_days"] = cal[f"{event}_day"].notna().astype(int)
    for event in after_events:
        out[f"n_{event}_after_days"] = (cal[f"{event}_day"] > 0).fillna(False).astype(int)
    return out[["date", *columns]]


# --------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------


def _weekday(date_col: Column) -> Column:
    # Spark's dayofweek is 1 = Sunday … 7 = Saturday; convert to 0 = Monday … 6 = Sunday.
    return ((F.dayofweek(date_col) + 5) % 7).cast("int")


def _weekday_count(origin_weekday: Column, h: Column, weekday: int) -> Column:
    """Number of days with ``weekday`` in ``o+1 .. o+h``."""
    first = (F.lit(weekday) - origin_weekday + 7) % 7  # days from o to the next such weekday
    first = F.when(first == 0, 7).otherwise(first)
    return F.when(h >= first, F.floor((h - first) / 7) + 1).otherwise(0).cast("int")


def history_days(model: ModelConfig) -> int:
    """Days of history an origin needs before all its features are defined."""
    return max(max(model.lags), max(model.rolling_windows), 7 * model.weekday_mean_weeks)


def build_features(daily: DataFrame, calendar: DataFrame, model: ModelConfig) -> DataFrame:
    """Build origin–horizon rows (see the module docstring).

    Args:
        daily: ``machine_id, date, target, y`` plus static machine columns; one row per
            machine, target and day, with no gaps.
        calendar: ``date`` plus 0/1 indicator columns named after their window feature;
            must cover every ``o+h``.
        model: Lags, rolling windows, weekday-mean weeks and horizon.
    """
    static = [c for c in daily.columns if c not in (*KEYS, "date", "y")]
    by_date = Window.partitionBy(*KEYS).orderBy("date")

    def last(n: int) -> Window:
        return by_date.rowsBetween(-(n - 1), Window.currentRow)

    wd_span = 7 * model.weekday_mean_weeks
    base = daily.withColumn("weekday", _weekday(F.col("date"))).withColumn(
        "cs", F.sum("y").over(by_date.rowsBetween(Window.unboundedPreceding, Window.currentRow))
    )

    origin_features = [
        *[F.lag("y", k - 1).over(by_date).alias(f"y_lag_{k}") for k in model.lags],
        *[
            col
            for n in model.rolling_windows
            for col in (
                F.avg("y").over(last(n)).alias(f"y_roll_mean_{n}"),
                F.stddev_samp("y").over(last(n)).alias(f"y_roll_std_{n}"),
            )
        ],
        *[
            F.avg(F.when(F.col("weekday") == d, F.col("y")))
            .over(last(wd_span))
            .alias(f"y_wd_mean_{d}")
            for d in WEEKDAYS
        ],
    ]
    origins = (
        base.select(
            *KEYS,
            F.col("date").alias("origin_date"),
            *static,
            "cs",
            F.col("weekday").alias("origin_weekday"),
            F.month("date").alias("origin_month"),
            F.row_number().over(by_date).alias("n_history"),
            *origin_features,
        )
        .where(F.col("n_history") >= history_days(model))
        .drop("n_history")
    )

    horizons = F.sequence(F.lit(1), F.lit(model.horizon_max))
    rows = origins.withColumn("h", F.explode(horizons)).withColumn(
        "target_date", F.date_add("origin_date", F.col("h"))
    )

    # Label: cumulative-sum difference, via a self-join on date = o + h.
    future = base.select(*KEYS, F.col("date").alias("target_date"), F.col("cs").alias("cs_end"))
    rows = rows.join(future, [*KEYS, "target_date"], "left").withColumn(
        "y_cum", F.col("cs_end") - F.col("cs")
    )

    # Window counts: cumulative sums of the calendar indicators, read at o + h and at o.
    indicators = [c for c in calendar.columns if c != "date"]
    cum = calendar.select(
        "date",
        *[
            F.sum(c).over(Window.orderBy("date").rowsBetween(Window.unboundedPreceding, 0)).alias(c)
            for c in indicators
        ],
    )
    at_end = cum.select(
        F.col("date").alias("target_date"), *[F.col(c).alias(f"{c}__end") for c in indicators]
    )
    at_origin = cum.select(
        F.col("date").alias("origin_date"), *[F.col(c).alias(f"{c}__start") for c in indicators]
    )
    rows = rows.join(F.broadcast(at_end), "target_date", "left").join(
        F.broadcast(at_origin), "origin_date", "left"
    )

    counts = {d: _weekday_count(F.col("origin_weekday"), F.col("h"), d) for d in WEEKDAYS}
    expected = reduce(lambda a, b: a + b, [counts[d] * F.col(f"y_wd_mean_{d}") for d in WEEKDAYS])

    return rows.select(
        "machine_id",
        "origin_date",
        "h",
        "target",
        *static,
        "origin_weekday",
        "origin_month",
        *[f"y_lag_{k}" for k in model.lags],
        *[f"y_roll_{s}_{n}" for n in model.rolling_windows for s in ("mean", "std")],
        *[f"y_wd_mean_{d}" for d in WEEKDAYS],
        sum(counts[d] for d in WEEKEND).alias("n_weekend_days"),
        expected.alias("expected_by_weekday"),
        *[(F.col(f"{c}__end") - F.col(f"{c}__start")).cast("int").alias(c) for c in indicators],
        "y_cum",
    )


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


def run(tracks: tuple[str, ...] = TRACKS, spark: SparkSession | None = None) -> None:
    """Build and write ``features_<track>`` for each track (Parquet partitioned by target)."""
    spark = spark or get_spark()
    model = load_config("model")
    scenario = load_config("scenario")

    for track in tracks:
        daily = load_daily(spark, track)
        span = daily.agg(F.min("date").alias("start"), F.max("date").alias("end")).first()
        start = pd.Timestamp(span["start"])
        end = pd.Timestamp(span["end"]) + timedelta(days=model.horizon_max)
        calendar = spark.createDataFrame(calendar_indicators(track, start, end, scenario))
        calendar = calendar.withColumn("date", F.to_date("date"))

        features = build_features(daily, calendar, model)
        path = get_data_dir() / "features" / f"features_{track}"
        features.write.mode("overwrite").partitionBy("target").parquet(str(path))
        register_parquet_view(f"features_{track}", path)

        written = spark.read.parquet(str(path))
        logger.info(
            "features_%s: %d rows, %d columns, targets %s",
            track,
            written.count(),
            len(written.columns),
            sorted(r["target"] for r in written.select("target").distinct().collect()),
        )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Build origin–horizon feature tables.")
    parser.add_argument("--track", choices=[*TRACKS, "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(TRACKS if args.track == "all" else (args.track,))


if __name__ == "__main__":
    main()
