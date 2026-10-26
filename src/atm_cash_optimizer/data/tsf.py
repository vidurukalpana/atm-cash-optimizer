"""Parse ``.tsf`` files (Monash forecasting archive) into a long DataFrame.

Data lines look like ``T1:1996-03-18 00-00-00:13.4,14.7,?,...``; ``?`` is a missing value.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
MISSING = "?"


def _parse_values(text: str) -> np.ndarray:
    return np.array([np.nan if v == MISSING else float(v) for v in text.split(",")])


def read_tsf(path: str | Path) -> pd.DataFrame:
    """Read a daily ``.tsf`` file into ``machine_id`` (``NN5_001``…), ``date`` and
    ``withdrawals_raw`` (NaN where the file has ``?``)."""
    attributes: list[str] = []
    frequency: str | None = None
    frames: list[pd.DataFrame] = []
    in_data = False

    with Path(path).open(encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if not in_data:
                lower = line.lower()
                if lower.startswith("@attribute"):
                    attributes.append(line.split()[1])
                elif lower.startswith("@frequency"):
                    frequency = line.split()[1]
                elif lower.startswith("@data"):
                    if frequency != "daily":
                        raise ValueError(f"expected @frequency daily, got {frequency!r}")
                    if not {"series_name", "start_timestamp"} <= set(attributes):
                        raise ValueError(f"missing series_name/start_timestamp in {attributes}")
                    in_data = True
                continue

            parts = line.split(":")
            if len(parts) != len(attributes) + 1:
                raise ValueError(f"line {line_no}: expected {len(attributes) + 1} fields")
            meta = dict(zip(attributes, parts[:-1], strict=True))
            values = _parse_values(parts[-1])
            start = pd.to_datetime(meta["start_timestamp"], format=TIMESTAMP_FORMAT)

            frames.append(
                pd.DataFrame(
                    {
                        "machine_id": f"NN5_{len(frames) + 1:03d}",
                        "date": pd.date_range(start, periods=len(values), freq="D"),
                        "withdrawals_raw": values,
                    }
                )
            )

    if not frames:
        raise ValueError(f"no series found in {path}")

    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Parsed %d series, %d rows, %d missing values from %s",
        len(frames),
        len(df),
        df["withdrawals_raw"].isna().sum(),
        path,
    )
    return df
