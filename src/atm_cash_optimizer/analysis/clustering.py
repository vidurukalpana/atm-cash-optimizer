"""Group machines with similar demand behaviour and write ``machine_clusters``.

Features per machine (see :func:`machine_features`):

- normalized weekday profile (7 values, mean 1) — the *shape* of the week;
- coefficient of variation — how volatile the machine is;
- level — mean daily withdrawals.

Features are standardized, then KMeans is fitted for every k in :data:`K_RANGE` and the k with
the best silhouette score is kept. Each cluster gets a readable label — ``steady``,
``weekend_heavy`` or ``volatile`` — by rules on its centroid (see :func:`label_clusters`).

Run with::

    python -m atm_cash_optimizer.analysis.clustering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import holidays
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from atm_cash_optimizer.analysis.profiles import (
    VALUE_COL,
    coefficient_of_variation,
    holiday_effect,
    month_end_effect,
    monthly_profile,
    weekday_profile,
)
from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.io import get_reports_dir, markdown_table, read_table, write_table

logger = logging.getLogger(__name__)

K_RANGE = range(2, 7)
"""Numbers of clusters tried (fixed by the development plan)."""
KMEANS_N_INIT = 10
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NN5_HOLIDAY_COUNTRY = ("GB", "ENG")  # NN5 machines are in the UK; England's bank holidays


@dataclass(frozen=True)
class ClusteringResult:
    """Output of :func:`cluster_machines`."""

    clusters: pd.DataFrame
    """``machine_id``, ``cluster_id``, ``cluster_label`` (cluster 0 = largest)."""
    silhouette: dict[int, float]
    """Silhouette score for every k tried."""
    k: int
    """Chosen number of clusters."""
    centroids: pd.DataFrame
    """Mean feature values per cluster (original units), indexed by ``cluster_id``."""


def machine_features(df: pd.DataFrame, value_col: str = VALUE_COL) -> pd.DataFrame:
    """One row per machine: ``wd_0`` … ``wd_6`` (normalized weekday profile), ``cv``, ``level``."""
    profile = weekday_profile(df, value_col, normalize=True)
    profile.columns = [f"wd_{d}" for d in profile.columns]
    level = df.groupby("machine_id")[value_col].mean().rename("level")
    return pd.concat([profile, coefficient_of_variation(df, value_col), level], axis=1)


def weekend_share(features: pd.DataFrame) -> pd.Series:
    """Mean of the normalized Saturday and Sunday profile (``1.0`` = an average day)."""
    return features[["wd_5", "wd_6"]].mean(axis=1).rename("weekend_share")


def label_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Give each cluster one of the labels used by ``configs/scenario.yaml``.

    Rules on the centroids (columns of :func:`machine_features`), applied in order:

    1. the cluster with the lowest ``cv`` → ``steady``;
    2. if there are 3+ clusters, the remaining cluster with the highest weekend share →
       ``weekend_heavy``;
    3. every other cluster → ``volatile``.

    Labels are not unique when k > 3 (several ``volatile`` clusters); ``cluster_id`` is the key.
    """
    labels = dict.fromkeys(centroids.index, "volatile")
    steady = centroids["cv"].idxmin()
    labels[steady] = "steady"
    if len(centroids) >= 3:
        labels[weekend_share(centroids.drop(index=steady)).idxmax()] = "weekend_heavy"
    return {int(cluster_id): label for cluster_id, label in labels.items()}


def cluster_machines(df: pd.DataFrame, seed: int, value_col: str = VALUE_COL) -> ClusteringResult:
    """Cluster machines by demand behaviour, choosing k in :data:`K_RANGE` by silhouette.

    Args:
        df: Long table with ``machine_id``, ``date`` and ``value_col``.
        seed: Random seed for KMeans (same seed → same clusters).
        value_col: Column holding daily demand.
    """
    features = machine_features(df, value_col)
    z = StandardScaler().fit_transform(features)

    fits: dict[int, KMeans] = {}
    silhouette: dict[int, float] = {}
    for k in K_RANGE:
        if k >= len(features):  # silhouette needs at least one more machine than clusters
            break
        fits[k] = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed).fit(z)
        silhouette[k] = float(silhouette_score(z, fits[k].labels_))
    k = max(silhouette, key=silhouette.__getitem__)

    # Renumber clusters by size (0 = largest) so ids are stable and easy to read.
    raw = pd.Series(fits[k].labels_, index=features.index)
    order = {old: new for new, old in enumerate(raw.value_counts().index)}
    cluster_id = raw.map(order).rename("cluster_id")

    centroids = features.groupby(cluster_id).mean()
    labels = label_clusters(centroids)

    clusters = cluster_id.reset_index()
    clusters["cluster_label"] = clusters["cluster_id"].map(labels)
    logger.info("Chose k=%d (silhouette %.3f): %s", k, silhouette[k], labels)
    return ClusteringResult(clusters=clusters, silhouette=silhouette, k=k, centroids=centroids)


def eda_summary(df: pd.DataFrame, result: ClusteringResult) -> str:
    """Build the markdown text of ``reports/eda_summary.md``."""
    years = sorted(df["date"].dt.year.unique())
    holiday_dates = holidays.country_holidays(
        NN5_HOLIDAY_COUNTRY[0], subdiv=NN5_HOLIDAY_COUNTRY[1], years=years
    )
    days_per_machine = df.groupby("machine_id").size().iloc[0]
    cv = coefficient_of_variation(df)
    month_end = month_end_effect(df)

    weekday = weekday_profile(df, normalize=True).mean().to_frame("relative_to_avg_day").T
    weekday.columns = [d[:3] for d in WEEKDAYS]
    monthly = monthly_profile(df, normalize=True).mean().to_frame("relative_to_avg_month").T
    holiday = holiday_effect(df, holiday_dates.keys())

    sizes = result.clusters.groupby("cluster_id").agg(
        cluster_label=("cluster_label", "first"), machines=("machine_id", "size")
    )
    clusters = sizes.join(result.centroids[["level", "cv"]]).join(weekend_share(result.centroids))
    clusters["peak_day"] = (
        result.centroids.filter(like="wd_").idxmax(axis=1).map(lambda c: WEEKDAYS[int(c[-1])])
    )
    clusters["level"] = clusters["level"].map("{:,.0f}".format)
    silhouette = pd.DataFrame([result.silhouette], index=["silhouette"])

    return f"""# EDA summary — nn5 track

Generated by `python -m atm_cash_optimizer.analysis.clustering`. Do not edit by hand.

## Data

- Machines: {df["machine_id"].nunique()}, days per machine: {days_per_machine}
- Dates: {df["date"].min().date()} to {df["date"].max().date()}
- Mean withdrawals: LKR {df[VALUE_COL].mean():,.0f} per machine-day

## Weekly pattern (fleet average, 1.00 = average day)

{markdown_table(weekday)}

## Monthly pattern (fleet average, 1.00 = average month)

{markdown_table(monthly)}

## Volatility (coefficient of variation per machine)

- min {cv.min():.2f}, median {cv.median():.2f}, max {cv.max():.2f}
- most volatile: {", ".join(f"{m} ({v:.2f})" for m, v in cv.nlargest(5).items())}

## Month-end effect

Mean demand in the last 3 and first 3 days of a month vs other days: median ratio
{month_end.median():.2f} (machines range {month_end.min():.2f}–{month_end.max():.2f}).

## Holiday effect ({NN5_HOLIDAY_COUNTRY[0]}-{NN5_HOLIDAY_COUNTRY[1]} bank holidays)

Demand relative to a normal day of the same weekday (offset −1 = day before the holiday).

{markdown_table(holiday, floatfmt="{:.2f}")}

## Machine clusters

Silhouette score per k (higher = better separated):

{markdown_table(silhouette, floatfmt="{:.3f}")}

Chosen **k = {result.k}**. Labels: lowest cv → `steady`; of the rest, highest weekend share
→ `weekend_heavy` (k ≥ 3); others → `volatile`. `weekend_share` 1.00 = an average day.

{markdown_table(clusters.reset_index())}
"""


def run(data_dir: str | Path | None = None, reports_dir: str | Path | None = None) -> None:
    """Cluster the ``nn5_daily`` machines, write ``machine_clusters`` and the EDA summary."""
    df = read_table("nn5_daily", data_dir)
    result = cluster_machines(df, seed=load_config("model").seed)
    write_table(result.clusters, "machine_clusters", data_dir)

    path = get_reports_dir(reports_dir) / "eda_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(eda_summary(df, result), encoding="utf-8")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run()
