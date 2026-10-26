"""Optimizer inputs and outputs.

Day ``t = 1..H`` is ``origin + t``; a visit on day ``t`` happens before customers arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from atm_cash_optimizer.config import CostsConfig

TOTAL = "total"  # single "denomination" of CRMs and the nn5 track


@dataclass(frozen=True)
class MachineState:
    """Cash, capacity and demand shares at the end of the origin day.

    Keyed by denomination (``"5000"``, ``"1000"``, ``"500"``) or ``"total"`` (CRMs, nn5).
    """

    machine_id: str
    machine_type: str  # "ATM" or "CRM"
    cash: dict[str, float]
    capacity: dict[str, float]
    shares: dict[str, float]


@dataclass(frozen=True)
class ForecastWindow:
    """Cumulative quantiles for ``h = 1..H``: withdrawals (ATM) or net outflow (CRM)."""

    origin_date: pd.Timestamp
    quantiles: dict[float, np.ndarray]

    @property
    def horizon(self) -> int:
        return len(next(iter(self.quantiles.values())))

    def get(self, q: float) -> np.ndarray:
        """Quantile ``q``; raises if it was not forecast."""
        if q not in self.quantiles:
            raise ValueError(f"quantile {q} not available; have {sorted(self.quantiles)}")
        return self.quantiles[q]

    def lower(self, q: float) -> np.ndarray:
        """Largest available quantile ``<= q`` (a cautious stand-in for a missing one)."""
        candidates = [k for k in self.quantiles if k <= q + 1e-9]
        if not candidates:
            raise ValueError(f"no quantile at or below {q}; have {sorted(self.quantiles)}")
        return self.quantiles[max(candidates)]


@dataclass(frozen=True)
class PlanSettings:
    """Solver settings shared by all machines."""

    alpha: float
    costs: CostsConfig
    lead_days: int
    bundle_notes: int
    largest_note: int  # bundle value of the "total" denomination = bundle_notes × largest_note

    def bundle_value(self, denomination: str) -> float:
        """LKR value of one bundle (``total`` uses the largest note)."""
        note = self.largest_note if denomination == TOTAL else int(denomination)
        return float(self.bundle_notes * note)


@dataclass(frozen=True)
class Visit:
    """One planned visit: load per denomination, and removal for CRMs."""

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
        """The decision to send now: the visit on the first orderable day, if any."""
        first_day = max(lead_days, 1)
        return next((v for v in self.visits if v.day == first_day), None)
