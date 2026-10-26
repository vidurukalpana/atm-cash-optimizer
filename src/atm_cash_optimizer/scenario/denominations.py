"""Split total withdrawals into banknote denominations (``withdrawals_<denom>_lkr``)."""

from __future__ import annotations

import pandas as pd

from atm_cash_optimizer.config import ScenarioConfig


def split_denominations(
    df: pd.DataFrame, cfg: ScenarioConfig, denominations: list[str]
) -> pd.DataFrame:
    """Add ``withdrawals_<denom>_lkr`` for each denomination, by location-type shares.

    Shares come from ``denomination_shares[<location_type>]``, else ``["default"]``. The
    columns add up to ``withdrawals_lkr``.

    Raises:
        ValueError: If a share table does not have exactly the track's denominations.
    """
    for location, shares in cfg.denomination_shares.items():
        if set(shares) != set(denominations):
            raise ValueError(
                f"denomination_shares.{location} has {sorted(shares)}, expected {denominations}"
            )

    out = df.copy()
    for denom in denominations:
        share = out["location_type"].map(
            lambda loc, d=denom: cfg.denomination_shares.get(
                loc, cfg.denomination_shares["default"]
            )[d]
        )
        out[f"withdrawals_{denom}_lkr"] = out["withdrawals_lkr"] * share
    return out
