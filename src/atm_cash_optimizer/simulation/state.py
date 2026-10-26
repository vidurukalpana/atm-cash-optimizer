"""What the simulation knows about one machine: fixed facts, daily cash and pending orders."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from atm_cash_optimizer.optimization.types import ForecastWindow

CRM_TARGET_FRACTION = 0.5
"""CRMs are refilled or emptied to half the vault (room for both withdrawals and deposits)."""


@dataclass(frozen=True)
class MachineSpec:
    """Everything fixed about one machine during a simulation run.

    ``withdrawals`` holds the real daily demand per denomination (``"total"`` for CRMs and nn5);
    ``dates[0]`` is the origin before the first simulated day.
    """

    machine_id: str
    machine_type: str
    location_type: str | None
    capacity: dict[str, float]
    shares: dict[str, float]
    dates: pd.DatetimeIndex
    withdrawals: dict[str, np.ndarray]
    deposits: np.ndarray
    no_service_dates: frozenset[pd.Timestamp]
    forecasts: dict[pd.Timestamp, ForecastWindow] = field(default_factory=dict)

    @property
    def is_crm(self) -> bool:
        return self.machine_type == "CRM"

    def initial_cash(self, fill_fraction: float) -> dict[str, float]:
        """ATMs start at ``fill_fraction`` of capacity; CRMs at half the vault."""
        fraction = CRM_TARGET_FRACTION if self.is_crm else fill_fraction
        return {d: fraction * c for d, c in self.capacity.items()}

    def history(self, day: pd.Timestamp) -> pd.DataFrame:
        """Real demand up to and including ``day`` — the only demand a policy may see."""
        keep = self.dates <= day
        return pd.DataFrame(
            {
                "date": self.dates[keep],
                "withdrawals": sum(w[keep] for w in self.withdrawals.values()),
                "deposits": self.deposits[keep],
            }
        )


@dataclass(frozen=True)
class Order:
    """A visit ordered by a policy (or the emergency rule)."""

    machine_id: str
    created: pd.Timestamp
    visit_date: pd.Timestamp
    load: dict[str, float]
    remove: float = 0.0
    emergency: bool = False

    @property
    def action(self) -> str:
        loads = sum(self.load.values()) > 0
        if loads and self.remove > 0:
            return "LOAD_AND_REMOVE"
        return "REMOVE" if self.remove > 0 else "LOAD"


@dataclass
class MachineState:
    """Cash per denomination at the end of the current day, and orders not yet executed."""

    cash: dict[str, float]
    pending: list[Order] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(self.cash.values())

    def has_order_by(self, last_date: pd.Timestamp) -> bool:
        """True if a visit is already ordered for a date up to ``last_date``."""
        return any(o.visit_date <= last_date for o in self.pending)
