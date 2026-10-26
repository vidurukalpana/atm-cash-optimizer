"""Local Spark session used by the feature pipeline and its tests."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession

from atm_cash_optimizer.config import load_config

DRIVER_MEMORY_ENV = "SPARK_DRIVER_MEMORY"
DEFAULT_DRIVER_MEMORY = "4g"


def get_spark(app_name: str = "atm-cash-optimizer") -> SparkSession:
    """Return a local Spark session.

    Uses ``n_jobs`` cores from ``model.yaml`` (not all cores, so the Mac stays cool), UTC time
    and driver memory from ``$SPARK_DRIVER_MEMORY``.
    """
    cores = load_config("model").n_jobs
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.driver.memory", os.environ.get(DRIVER_MEMORY_ENV, DEFAULT_DRIVER_MEMORY))
        .config("spark.sql.session.timeZone", "UTC")
        # Spark's default of 200 shuffle partitions is meant for clusters; a few per core
        # avoid thousands of tiny tasks on one laptop.
        .config("spark.sql.shuffle.partitions", 2 * cores)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
