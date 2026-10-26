"""Tests for the EDA branch: demand profiles, machine clustering and the SQL query loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atm_cash_optimizer.analysis import profiles
from atm_cash_optimizer.analysis.clustering import cluster_machines
from atm_cash_optimizer.io import read_sql_queries

START = "2024-01-01"  # a Monday


def _series(machine_id: str, values: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range(START, periods=len(values), freq="D")
    return pd.DataFrame({"machine_id": machine_id, "date": dates, "withdrawals_lkr": values})


def test_weekday_profile_and_cv_on_known_pattern() -> None:
    # A: 10 on every day except Saturday (20). B: constant 5.
    days = pd.date_range(START, periods=56, freq="D")
    df = pd.concat(
        [
            _series("A", np.where(days.day_of_week == 5, 20.0, 10.0)),
            _series("B", np.full(56, 5.0)),
        ]
    )

    raw = profiles.weekday_profile(df)
    normalized = profiles.weekday_profile(df, normalize=True)
    cv = profiles.coefficient_of_variation(df)

    assert raw.loc["A", 5] == 20 and raw.loc["A", 0] == 10
    assert normalized.loc["A", 5] == pytest.approx(20 / (80 / 7))
    assert normalized.loc["B"].tolist() == [1.0] * 7
    assert cv["B"] == 0 and cv["A"] > 0


def test_month_end_effect_on_known_pattern() -> None:
    # Double demand within 3 days of the month boundary, 1 elsewhere.
    days = pd.date_range(START, "2024-06-30", freq="D")
    near_end = (days.day <= 3) | (days.day > days.days_in_month - 3)
    df = _series("A", np.where(near_end, 2.0, 1.0))

    assert profiles.month_end_effect(df)["A"] == pytest.approx(2.0)


def test_holiday_effect_on_known_pattern() -> None:
    # Demand halves on the holiday, normal the day before and after.
    holiday = pd.Timestamp("2024-01-17")
    days = pd.date_range(START, periods=56, freq="D")
    df = _series("A", np.where(days == holiday, 5.0, 10.0))

    effect = profiles.holiday_effect(df, [holiday]).set_index("offset")["ratio"]

    assert effect.to_dict() == {-1: 1.0, 0: 0.5, 1: 1.0}


def test_clustering_is_deterministic_and_finds_steady_machines() -> None:
    rng = np.random.default_rng(0)
    weekly = np.tile([1.0, 1.0, 1.1, 1.5, 1.3, 0.7, 0.8], 12)
    df = pd.concat(
        [_series(f"S{i}", 100 * weekly * rng.normal(1, 0.02, weekly.size)) for i in range(6)]
        + [_series(f"V{i}", 100 * weekly * rng.lognormal(0, 0.6, weekly.size)) for i in range(6)]
    )

    first = cluster_machines(df, seed=42)
    second = cluster_machines(df, seed=42)

    pd.testing.assert_frame_equal(first.clusters, second.clusters)
    labels = first.clusters.set_index("machine_id")["cluster_label"]
    assert set(labels[labels.index.str.startswith("S")]) == {"steady"}
    assert set(labels) <= {"steady", "weekend_heavy", "volatile"}


def test_read_sql_queries(tmp_path: Path) -> None:
    path = tmp_path / "q.sql"
    path.write_text("-- header\n\n-- name: first\nSELECT 1;\n\n-- name: second\nSELECT 2;\n")

    assert read_sql_queries(path) == {"first": "SELECT 1", "second": "SELECT 2"}
