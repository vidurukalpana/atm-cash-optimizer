"""pandera schema for the ``nn5_daily`` table (see the data contracts in the plan)."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa


def _dates_continuous(df: pd.DataFrame) -> bool:
    # Every machine's dates step by exactly one day.
    steps = df.sort_values("date").groupby("machine_id")["date"].diff().dropna()
    return bool((steps == pd.Timedelta(days=1)).all())


def _equal_lengths(df: pd.DataFrame) -> bool:
    return df.groupby("machine_id").size().nunique() == 1


NN5_DAILY_SCHEMA = pa.DataFrameSchema(
    {
        "machine_id": pa.Column(str, nullable=False),
        "date": pa.Column("datetime64[ns]", nullable=False),
        "withdrawals_raw": pa.Column(float, pa.Check.ge(0), nullable=False),
        "withdrawals_lkr": pa.Column(float, pa.Check.ge(0), nullable=False),
        "is_imputed": pa.Column(bool, nullable=False),
    },
    checks=[
        pa.Check(_dates_continuous, error="dates are not continuous per machine"),
        pa.Check(_equal_lengths, error="series have different lengths"),
    ],
    unique=["machine_id", "date"],
    strict=True,
    coerce=True,
)


def validate_nn5_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ``df`` against :data:`NN5_DAILY_SCHEMA`; raise on any failure.

    Raises:
        pandera.errors.SchemaErrors: Listing every failed rule.
    """
    return NN5_DAILY_SCHEMA.validate(df, lazy=True)
