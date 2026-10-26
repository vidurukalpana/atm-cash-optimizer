"""Choose which machines are Cash Recycler Machines (CRMs) and simulate their deposits.

For each CRM, ``deposits_lkr`` on a day is::

    ratio × machine's mean withdrawals × weekday factor × holiday factor × noise

- ``ratio`` is drawn once per CRM from ``crm_deposits.ratio_range``;
- the weekday factor comes from ``crm_deposits.weekday_profile`` (rescaled to mean 1);
- the holiday factor is ``crm_deposits.holiday_multiplier`` on public holidays, else 1;
- noise is lognormal with mean 1.

ATMs get ``deposits_lkr = 0``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atm_cash_optimizer.config import ScenarioConfig
from atm_cash_optimizer.scenario.transform import lognormal_noise


def choose_machine_types(
    machine_ids: list[str], crm_fraction: float, rng: np.random.Generator
) -> pd.Series:
    """``machine_type`` (``"ATM"`` or ``"CRM"``) per machine; ``round(fraction × n)`` CRMs."""
    ids = sorted(machine_ids)
    n_crm = round(crm_fraction * len(ids))
    crms = set(rng.choice(ids, size=n_crm, replace=False)) if n_crm else set()
    return pd.Series(["CRM" if m in crms else "ATM" for m in ids], index=ids, name="machine_type")


def simulate_deposits(df: pd.DataFrame, cfg: ScenarioConfig, rng: np.random.Generator) -> pd.Series:
    """Daily ``deposits_lkr`` aligned with ``df``.

    Args:
        df: Rows sorted by ``machine_id``, ``date`` with ``machine_type``,
            ``withdrawals_lkr`` and ``is_public_holiday``.
    """
    deposits_cfg = cfg.crm_deposits
    profile = np.array(deposits_cfg.weekday_profile)
    profile = profile / profile.mean()

    crm_ids = sorted(df.loc[df["machine_type"] == "CRM", "machine_id"].unique())
    ratio = pd.Series(rng.uniform(*deposits_cfg.ratio_range, size=len(crm_ids)), index=crm_ids)
    mean_withdrawals = df.groupby("machine_id")["withdrawals_lkr"].transform("mean")

    is_crm = (df["machine_type"] == "CRM").to_numpy()
    deposits = (
        df["machine_id"].map(ratio).fillna(0).to_numpy()
        * mean_withdrawals.to_numpy()
        * profile[df["date"].dt.day_of_week.to_numpy()]
        * np.where(df["is_public_holiday"], deposits_cfg.holiday_multiplier, 1.0)
        * lognormal_noise(rng, deposits_cfg.noise_sigma, len(df))
    )
    return pd.Series(np.where(is_crm, deposits, 0.0), index=df.index, name="deposits_lkr")
