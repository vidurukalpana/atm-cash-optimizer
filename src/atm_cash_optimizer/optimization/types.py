"""Inputs and outputs of the replenishment optimizer.

Days are counted from the origin ``o`` (the evening the plan is made): day ``t = 1..H`` is
``o + t``. A visit on day ``t`` happens at the start of that day, before customers arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from atm_cash_optimizer.config import CostsConfig

TOTAL = "total"  # the single "denomination" of CRMs and of the nn5 track


@dataclass(frozen=True)
class MachineState:
    """A machine at the end of the origin day.

    ``cash``, ``capacity`` and ``shares`` are keyed by denomination (``"5000"``, ``"1000"``,
    ``"500"``), or by ``"total"`` for CRMs and the nn5 track.
    """

    machine_id: str
    machine_type: str  # "ATM" or "CRM"
    cash: dict[str, float]
    capacity: dict[str, float]
    shares: dict[str, float]


@dataclass(frozen=True)
class ForecastWindow:
    """Cumulative quantile forecasts ``C(o, h)`` for ``h = 1..H``.

    For ATMs these are withdrawals; for CRMs net outflow (withdrawals − deposits).
    """

    origin_date: pd.Timestamp
    quantiles: dict[float, np.ndarray]

    @property
    def horizon(self) -> int:
        return len(next(iter(self.quantiles.values())))

    def get(self, q: float) -> np.ndarray:
        """Forecast of quantile ``q``; raises if it was not forecast."""
        if q not in self.quantiles:
            raise ValueError(f"quantile {q} not available; have {sorted(self.quantiles)}")
        return self.quantiles[q]

    def lower(self, q: float) -> np.ndarray:
        """The largest available quantile ``<= q`` (a safe stand-in for a missing low quantile)."""
        candidates = [k for k in self.quantiles if k <= q + 1e-9]
        if not candidates:
            raise ValueError(f"no quantile at or below {q}; have {sorted(self.quantiles)}")
        return self.quantiles[max(candidates)]


@dataclass(frozen=True)
class PlanSettings:
    """Everything the solver needs besides the machine and its forecast."""

    alpha: float
    costs: CostsConfig
    lead_days: int
    bundle_notes: int
    largest_note: int  # bundle value of the "total" denomination = bundle_notes × largest_note

    def bundle_value(self, denomination: str) -> float:
        """LKR value of one bundle of ``denomination``."""
        note = self.largest_note if denomination == TOTAL else int(denomination)
        return float(self.bundle_notes * note)


@dataclass(frozen=True)
class Visit:
    """One planned visit: what to load (per denomination) and how much to remove (CRM)."""

    day: int
    date: pd.Timestamp
    action: str  # LOAD, REMOVE or LOAD_AND_REMOVE
    load: dict[str, float]
    remove: float = 0.0


@dataclass(frozen=True)
class Plan:
    """The cheapest plan for one machine over the next ``H`` days."""

    machine_id: str
    origin_date: pd.Timestamp
    alpha: float
    visits: list[Visit] = field(default_factory=list)
    objective: float = 0.0
    status: str = ""
    used_slack: bool = False

    def committed(self, lead_days: int) -> Visit | None:
        """The decision to send now (rolling horizon): a visit on the first orderable day."""
        first_day = max(lead_days, 1)
        return next((v for v in self.visits if v.day == first_day), None)
