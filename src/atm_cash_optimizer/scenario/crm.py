"""Pick the CRMs and simulate their deposits (ATMs get 0).

Daily deposits = ratio (per CRM) × mean withdrawals × weekday factor × holiday factor × noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atm_cash_optimizer.config import ScenarioConfig
from atm_cash_optimizer.scenario.transform import lognormal_noise


def choose_machine_types(
    machine_ids: list[str], crm_fraction: float, rng: np.random.Generator
) -> pd.Series:
    """``"ATM"`` or ``"CRM"`` per machine; ``round(fraction × n)`` are CRMs."""
    ids = sorted(machine_ids)
    n_crm = round(crm_fraction * len(ids))
    crms = set(rng.choice(ids, size=n_crm, replace=False)) if n_crm else set()
    return pd.Series(["CRM" if m in crms else "ATM" for m in ids], index=ids, name="machine_type")


def simulate_deposits(df: pd.DataFrame, cfg: ScenarioConfig, rng: np.random.Generator) -> pd.Series:
    """Daily ``deposits_lkr`` aligned with ``df`` (needs ``machine_type``, ``withdrawals_lkr``,
    ``is_public_holiday``)."""
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
