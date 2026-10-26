"""Local Spark session used by the feature pipeline and its tests."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession

DRIVER_MEMORY_ENV = "SPARK_DRIVER_MEMORY"
DEFAULT_DRIVER_MEMORY = "4g"
# Spark's default of 200 shuffle partitions is meant for clusters; on one laptop a few
# partitions per core avoid thousands of tiny tasks.
SHUFFLE_PARTITIONS = max(4, 2 * (os.cpu_count() or 2))


def get_spark(app_name: str = "atm-cash-optimizer") -> SparkSession:
    """Return a local Spark session: all cores, UTC, driver memory from ``$SPARK_DRIVER_MEMORY``."""
    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", os.environ.get(DRIVER_MEMORY_ENV, DEFAULT_DRIVER_MEMORY))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", SHUFFLE_PARTITIONS)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
