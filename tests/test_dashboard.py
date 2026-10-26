"""Smoke tests: every dashboard page renders without errors on a tiny warehouse."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.io import DATA_DIR_ENV, write_table
from atm_cash_optimizer.simulation.metrics import summarize

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
PAGES = [
    "pages/overview.py",
    "pages/cash_status.py",
    "pages/machine_insights.py",
    "pages/results.py",
]
DATES = pd.date_range("2025-01-01", periods=45, freq="D")
SIM_DATES = DATES[5:35]
MACHINES = pd.DataFrame(
    {
        "machine_id": ["LK_001", "LK_002"],
        "machine_type": ["ATM", "CRM"],
        "location_type": ["residential", "branch"],
    }
)
POLICIES = [("optimizer", 0.95), ("threshold", 0.2)]


def _sim_daily() -> pd.DataFrame:
    rows = []
    for policy, param in POLICIES:
        for machine in MACHINES.itertuples():
            for i, day in enumerate(SIM_DATES):
                atm = machine.machine_type == "ATM"
                cash = {"5000": 10e6, "1000": 1e6, "500": 0.4e6} if atm else {"total": 8e6}
                rows.append(
                    {
                        "policy": policy,
                        "param": param,
                        "machine_id": machine.machine_id,
                        "date": day,
                        "withdrawals_demand": 1e6,
                        "withdrawals_served": 1e6,
                        "end_cash": sum(cash.values()),
                        **{f"end_cash_{d}_lkr": v for d, v in cash.items()},
                        "visit": i % 7 == 0,
                        "emergency_visit": False,
                        "stockout": policy == "threshold" and i == len(SIM_DATES) - 1,
                        "vault_full": False,
                    }
                )
    return pd.DataFrame(rows)


def _forecasts() -> pd.DataFrame:
    rows = []
    for machine_id in MACHINES["machine_id"]:
        for origin in SIM_DATES:
            for h in range(1, 15):
                median = 1e6 * h
                rows.append(
                    {
                        "model": "quantile_lgbm",
                        "machine_id": machine_id,
                        "origin_date": origin,
                        "h": h,
                        "target": "withdrawals",
                        **{
                            f"q{round(q * 100):02d}": median * (0.5 + q)
                            for q in (0.05, 0.5, 0.8, 0.9, 0.95, 0.98)
                        },
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module", autouse=True)
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Write the few tables the dashboard reads into a temporary data directory."""
    root = tmp_path_factory.mktemp("data")
    daily = MACHINES.merge(pd.DataFrame({"date": DATES}), how="cross")
    daily["withdrawals_lkr"] = 1e6
    sim_daily = _sim_daily()
    summary = summarize(sim_daily, load_config("costs"), warmup_days=2).merge(
        MACHINES, on="machine_id"
    )
    orders = pd.DataFrame(
        {
            "policy": "optimizer",
            "param": 0.95,
            "machine_id": ["LK_001"],
            "visit_date": [SIM_DATES[-1]],
            "action": ["LOAD"],
            "amount_5000_lkr": [5e6],
            "amount_1000_lkr": [np.nan],
            "remove_amount_lkr": [0.0],
            "emergency": [False],
        }
    )
    for name, table in {
        "lk_daily": daily,
        "sim_daily_lk": sim_daily,
        "sim_summary_lk": summary,
        "orders_lk": orders,
        "forecasts_lk": _forecasts(),
    }.items():
        write_table(table, name, root)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_DIR_ENV, str(root))
        mp.syspath_prepend(str(DASHBOARD))  # the app imports its own `ui` and `data`
        yield root


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page: str) -> None:
    app = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=60)
    app.run()
    app.switch_page(page).run()

    assert not app.exception, app.exception
    assert app.markdown  # header and cards were drawn
