"""Local Spark session."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession

from atm_cash_optimizer.config import load_config

DRIVER_MEMORY_ENV = "SPARK_DRIVER_MEMORY"
DEFAULT_DRIVER_MEMORY = "4g"


def get_spark(app_name: str = "atm-cash-optimizer") -> SparkSession:
    """Spark on ``model.yaml → n_jobs`` cores, UTC, memory from ``$SPARK_DRIVER_MEMORY``."""
    cores = load_config("model").n_jobs
    spark = (
        SparkSession.builder.master(f"local[{cores}]")
        .appName(app_name)
        .config("spark.driver.memory", os.environ.get(DRIVER_MEMORY_ENV, DEFAULT_DRIVER_MEMORY))
        .config("spark.sql.session.timeZone", "UTC")
        # The default (200) is meant for clusters; on a laptop it makes thousands of tiny tasks.
        .config("spark.sql.shuffle.partitions", 2 * cores)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
