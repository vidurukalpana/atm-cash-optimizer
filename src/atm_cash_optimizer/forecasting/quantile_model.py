"""LightGBM quantile models: one global model per quantile (fitted per target by the caller).

Each model learns one quantile of ``y_cum`` (the cumulative demand over ``o+1 .. o+h``) from the
feature table built in :mod:`atm_cash_optimizer.features.build`. ``h`` is a feature, so a single
model covers all horizons. ``cluster_id``, ``location_type`` and ``machine_type`` are handled as
LightGBM categorical features.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from atm_cash_optimizer.config import ModelConfig
from atm_cash_optimizer.forecasting.evaluation import quantile_column

ID_COLUMNS = ["machine_id", "origin_date", "target"]
LABEL = "y_cum"
CATEGORICAL = ["cluster_id", "location_type", "machine_type"]


def feature_columns(features: pd.DataFrame) -> list[str]:
    """All model inputs: every column except ids and the label."""
    return [c for c in features.columns if c not in (*ID_COLUMNS, LABEL)]


def prepare(features: pd.DataFrame, categories: dict[str, list] | None = None) -> pd.DataFrame:
    """Model input matrix: categorical columns as pandas categories, numbers as float32.

    Args:
        categories: Category levels per categorical column; pass the training levels when
            preparing prediction rows so both use the same codes.
    """
    x = features[feature_columns(features)].copy()
    for column in CATEGORICAL:
        levels = None if categories is None else categories[column]
        x[column] = pd.Categorical(x[column].astype("string"), categories=levels)
    numeric = [c for c in x.columns if c not in CATEGORICAL]
    x[numeric] = x[numeric].astype("float32")
    return x


def fit_quantile_models(
    train: pd.DataFrame, cfg: ModelConfig, params: dict | None = None
) -> dict[float, lgb.LGBMRegressor]:
    """Fit one LightGBM model per quantile in ``cfg.quantiles`` on rows with a known label.

    Args:
        train: Feature rows including ``y_cum``.
        cfg: Quantiles, LightGBM hyperparameters, ``n_jobs`` and seed.
        params: Optional overrides of the LightGBM parameters (e.g. fewer trees in tests).
    """
    x = prepare(train)
    y = train[LABEL].to_numpy()
    lgb_params = {
        **cfg.lightgbm.model_dump(),
        "subsample_freq": 1,  # LightGBM only uses `subsample` when bagging is switched on
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.seed,
        "verbose": -1,
        **(params or {}),
    }
    categories = {c: list(x[c].cat.categories) for c in CATEGORICAL}
    models = {}
    for q in cfg.quantiles:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, **lgb_params)
        model.fit(x, y, categorical_feature=CATEGORICAL)
        model.categories_ = categories  # reused for prediction rows
        models[q] = model
    return models


def predict_quantiles(
    models: dict[float, lgb.LGBMRegressor], features: pd.DataFrame
) -> pd.DataFrame:
    """Predict every quantile; returns ``q05``, ``q50``, … columns aligned with ``features``."""
    x = prepare(features, next(iter(models.values())).categories_)
    return pd.DataFrame(
        {quantile_column(q): model.predict(x) for q, model in models.items()},
        index=features.index,
    )


def sort_quantiles(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fix quantile crossing: sort the quantile columns of each row in ascending order."""
    ordered = sorted(predictions.columns, key=lambda c: int(c[1:]))
    values = np.sort(predictions[ordered].to_numpy(), axis=1)
    return pd.DataFrame(values, index=predictions.index, columns=ordered)
