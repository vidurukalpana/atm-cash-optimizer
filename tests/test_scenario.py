"""Tests for the Sri Lanka scenario: calendar, transform, CRMs, denominations, reproducibility."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from atm_cash_optimizer.config import DEFAULT_CONFIG_DIR, load_config
from atm_cash_optimizer.scenario.build import build_scenario
from atm_cash_optimizer.scenario.calendar import build_lk_calendar
from atm_cash_optimizer.scenario.transform import calendar_multiplier, shift_dates

DENOMS = ["5000", "1000", "500"]
CFG = load_config("scenario")


@pytest.fixture(scope="module")
def nn5() -> pd.DataFrame:
    """Three machines, 8 weeks from NN5's first day (includes UK Easter 1996)."""
    dates = pd.date_range("1996-03-18", periods=56, freq="D")
    weekly = np.tile([0.8, 0.8, 1.0, 1.5, 1.3, 0.7, 0.9], 8)
    return pd.concat(
        pd.DataFrame(
            {
                "machine_id": f"NN5_00{i}",
                "date": dates,
                "withdrawals_lkr": 1e6 * (i + 1) * weekly,
                "is_imputed": False,
            }
        )
        for i in range(1, 4)
    )


@pytest.fixture(scope="module")
def clusters() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": ["NN5_001", "NN5_002", "NN5_003"],
            "cluster_id": [0, 1, 2],
            "cluster_label": ["steady", "volatile", "weekend_heavy"],
        }
    )


@pytest.fixture(scope="module")
def scenario(nn5: pd.DataFrame, clusters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_scenario(nn5, clusters, CFG, DENOMS)


def test_weekday_preserved(nn5: pd.DataFrame, scenario: tuple) -> None:
    _, daily = scenario
    original = nn5.sort_values(["machine_id", "date"])["date"].dt.day_of_week.to_numpy()

    assert (daily["date"].dt.day_of_week.to_numpy() == original).all()
    assert daily["date"].min() == pd.Timestamp(CFG.start_date)
    with pytest.raises(ValueError, match="weekdays"):
        shift_dates(nn5, date(2024, 3, 19))  # a Tuesday


def test_uk_holidays_neutralized(scenario: tuple) -> None:
    _, daily = scenario
    flagged = daily.loc[daily["is_neutralized"], "date"] - (
        pd.Timestamp(CFG.start_date) - pd.Timestamp("1996-03-18")
    )

    # Good Friday 1996-04-05 and Easter Monday 1996-04-08, each ±1 day.
    assert pd.Timestamp("1996-04-05") in set(flagged)
    assert pd.Timestamp("1996-04-08") in set(flagged)


def test_calendar_flags() -> None:
    cal = build_lk_calendar(date(2024, 4, 1), date(2024, 5, 31), CFG).set_index("date")

    assert cal.loc["2024-05-23", "is_poya"]  # Vesak Full Moon Poya Day
    assert not cal.loc["2024-05-24", "is_poya"]  # "Day Following Vesak" is a holiday, not Poya
    assert cal.loc["2024-05-24", "is_no_service_day"]
    assert cal.loc["2024-04-10", "is_long_weekend_eve"]  # Thu 11 to Mon 15 April are off
    assert cal.loc["2024-04-10", "avurudu_day"] == -3
    assert cal.loc["2024-04-10", "is_avurudu_window"]


def test_multipliers_only_on_flagged_dates() -> None:
    cal = build_lk_calendar(date(2024, 3, 18), date(2025, 3, 17), CFG)
    multiplier = calendar_multiplier(cal, CFG)

    event_days = cal[[f"{e}_day" for e in CFG.events]].notna().any(axis=1)
    flagged = cal["is_poya"] | cal["is_salary_week"] | cal["is_long_weekend_eve"] | event_days

    assert (multiplier[~flagged] == 1.0).all()
    # A flagged day keeps 1.0 only if it is a window day with no effect (e.g. Avurudu day 0).
    assert (multiplier[flagged] != 1.0).mean() > 0.95


def test_atm_deposits_zero(scenario: tuple) -> None:
    _, daily = scenario

    assert (daily.loc[daily["machine_type"] == "ATM", "deposits_lkr"] == 0).all()
    assert (daily.loc[daily["machine_type"] == "CRM", "deposits_lkr"] > 0).all()
    assert daily["is_simulated"].all()


def test_denomination_shares_sum_to_one(scenario: tuple, tmp_path) -> None:
    _, daily = scenario
    split = daily[[f"withdrawals_{d}_lkr" for d in DENOMS]].sum(axis=1)
    assert np.allclose(split, daily["withdrawals_lkr"])

    data = yaml.safe_load((DEFAULT_CONFIG_DIR / "scenario.yaml").read_text())
    data["denomination_shares"]["default"]["500"] = 0.5
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="sum to 1"):
        load_config("scenario", config_dir=tmp_path)


def test_same_seed_identical_output(nn5: pd.DataFrame, clusters: pd.DataFrame) -> None:
    def digest(seed: int) -> int:
        cfg = CFG.model_copy(update={"seed": seed})
        _, daily = build_scenario(nn5, clusters, cfg, DENOMS)
        return int(pd.util.hash_pandas_object(daily, index=False).sum())

    assert digest(42) == digest(42)
    assert digest(42) != digest(7)


def test_bootstrap_network_size(nn5: pd.DataFrame, clusters: pd.DataFrame) -> None:
    _, daily = build_scenario(nn5, clusters, CFG, DENOMS, n_machines=10, n_days=28)

    assert daily["machine_id"].nunique() == 10
    assert daily.groupby("machine_id").size().eq(28).all()
