"""Three baseline forecasters, run over the rolling-origin backtest.

All return the cumulative median ``q50`` of ``y_cum(o, h)`` for ``h = 1..H``:

- **A — seasonal naive:** each future day = the same weekday one week before the origin's
  week (for ``h > 7`` the last week simply repeats).
- **B — weekday mean:** each future day = mean of the same weekday over the last
  ``weekday_mean_weeks`` weeks (used by baseline policy 3 later).
- **C — AutoETS** (statsforecast, season length 7): fitted on the last ``ets_train_days`` days,
  refitted every ``retrain_every_days`` origins; on the days between, the fitted model is
  applied to the newest data (``forward``) without re-estimating. Runs on ``n_jobs`` cores.

Run with::

    python -m atm_cash_optimizer.forecasting.baselines                 # both tracks, A B C
    python -m atm_cash_optimizer.forecasting.baselines --track nn5 --models A B
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from atm_cash_optimizer.config import ModelConfig, load_config
from atm_cash_optimizer.forecasting import evaluation
from atm_cash_optimizer.forecasting.backtest import (
    TRACKS,
    SeriesPanel,
    history_slice,
    load_series,
    origin_indices,
)
from atm_cash_optimizer.io import DEFAULT_DATA_DIR, read_table, write_table

logger = logging.getLogger(__name__)

MODELS = ("A", "B", "C")
WEEK = 7
SEASON_LENGTH = 7
MLFLOW_DIR = DEFAULT_DATA_DIR.parent / "mlruns"
EXPERIMENT = "forecast-baselines"
WORKER_THREAD_VARS = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


# --------------------------------------------------------------------------------------------
# Forecasters: (series × days) matrix + origin positions → daily forecasts (series, origins, H)
# --------------------------------------------------------------------------------------------


def _same_weekday_last_week(horizon: int) -> np.ndarray:
    # Day o+k has the weekday of day o+r-7 with r = ((k-1) mod 7) + 1, i.e. one of o-6 .. o.
    return (np.arange(horizon) % WEEK) + 1 - WEEK


def seasonal_naive(values: np.ndarray, origin_idx: np.ndarray, horizon: int) -> np.ndarray:
    """Baseline A: daily forecasts, shape ``(n_series, n_origins, horizon)``."""
    columns = origin_idx[:, None] + _same_weekday_last_week(horizon)[None, :]
    return values[:, columns]


def weekday_mean(
    values: np.ndarray, origin_idx: np.ndarray, horizon: int, weeks: int
) -> np.ndarray:
    """Baseline B: daily forecasts, shape ``(n_series, n_origins, horizon)``."""
    last_week = origin_idx[:, None] + _same_weekday_last_week(horizon)[None, :]
    stacked = [values[:, last_week - WEEK * j] for j in range(weeks)]
    return np.mean(stacked, axis=0)


def _ets_one_series(
    y: np.ndarray, origin_idx: np.ndarray, horizon: int, train_days: int, refit_every: int
) -> np.ndarray:
    from statsforecast.models import AutoETS  # imported in the worker process

    forecasts = np.empty((len(origin_idx), horizon))
    model, fit_start = None, 0
    for n, i in enumerate(origin_idx):
        if n % refit_every == 0:
            window = history_slice(i, train_days)
            model = AutoETS(season_length=SEASON_LENGTH).fit(y[window])
            fit_start = window.start
        # Same parameters, state updated with every day since the fit (still only data <= o).
        forecasts[n] = model.forward(y[fit_start : i + 1], h=horizon)["mean"]
    return forecasts


def auto_ets(
    values: np.ndarray, origin_idx: np.ndarray, horizon: int, model_cfg: ModelConfig
) -> np.ndarray:
    """Baseline C: daily forecasts, shape ``(n_series, n_origins, horizon)``."""
    work = partial(
        _ets_one_series,
        origin_idx=origin_idx,
        horizon=horizon,
        train_days=model_cfg.ets_train_days,
        refit_every=model_cfg.retrain_every_days,
    )
    # Workers inherit these at start-up: one thread each, so n_jobs workers use n_jobs cores.
    for var in WORKER_THREAD_VARS:
        os.environ[var] = "1"
    with ProcessPoolExecutor(max_workers=model_cfg.n_jobs) as pool:
        return np.stack(list(pool.map(work, values)))


def forecast(
    model: str, panel: SeriesPanel, origin_idx: np.ndarray, model_cfg: ModelConfig
) -> np.ndarray:
    """Cumulative ``q50`` forecasts of a baseline, shape ``(n_series, n_origins, H)``."""
    horizon = model_cfg.horizon_max
    if model == "A":
        daily = seasonal_naive(panel.values, origin_idx, horizon)
    elif model == "B":
        daily = weekday_mean(panel.values, origin_idx, horizon, model_cfg.weekday_mean_weeks)
    elif model == "C":
        daily = auto_ets(panel.values, origin_idx, horizon, model_cfg)
    else:
        raise ValueError(f"unknown baseline {model!r}; expected one of {MODELS}")
    return np.cumsum(daily, axis=-1)


def to_frame(
    model: str,
    panel: SeriesPanel,
    origin_idx: np.ndarray,
    q50: np.ndarray,
    quantiles: list[float],
) -> pd.DataFrame:
    """Long ``forecasts_<track>`` rows; quantiles other than the median stay empty."""
    n_series, n_origins, horizon = q50.shape
    out = pd.DataFrame(
        {
            "model": f"baseline_{model}",
            "machine_id": np.repeat(panel.keys["machine_id"].to_numpy(), n_origins * horizon),
            "origin_date": np.tile(np.repeat(panel.dates[origin_idx], horizon), n_series),
            "h": np.tile(np.arange(1, horizon + 1), n_series * n_origins),
            "target": np.repeat(panel.keys["target"].to_numpy(), n_origins * horizon),
        }
    )
    for q in quantiles:
        out[evaluation.quantile_column(q)] = q50.ravel() if q == 0.5 else np.nan
    return out


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


def _mlflow_experiment() -> None:
    import mlflow

    MLFLOW_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=str(MLFLOW_DIR / "artifacts"))
    mlflow.set_experiment(EXPERIMENT)


def _save_forecasts(track: str, new: pd.DataFrame) -> None:
    # Replace only the models that were just run; keep other models' rows in the table.
    name = f"forecasts_{track}"
    try:
        existing = read_table(name)
        new = pd.concat(
            [existing[~existing["model"].isin(new["model"].unique())], new], ignore_index=True
        )
    except FileNotFoundError:
        pass
    write_table(new, name)


def run(tracks: tuple[str, ...] = TRACKS, models: tuple[str, ...] = MODELS) -> None:
    """Backtest the baselines, write ``forecasts_<track>`` and log metrics to MLflow."""
    import mlflow

    model_cfg = load_config("model")
    backtest_cfg = load_config("backtest")
    _mlflow_experiment()

    for track in tracks:
        panel = load_series(track)
        origin_idx = origin_indices(len(panel.dates), backtest_cfg.test_days)
        actual = evaluation.cumulative_actuals_frame(panel, origin_idx, model_cfg.horizon_max)
        segments = evaluation.segment_frame(track, panel, origin_idx, model_cfg.horizon_max)

        frames = []
        for model in models:
            started = time.perf_counter()
            q50 = forecast(model, panel, origin_idx, model_cfg)
            seconds = time.perf_counter() - started
            frame = to_frame(model, panel, origin_idx, q50, model_cfg.quantiles)
            frames.append(frame)

            metrics = evaluation.evaluate(frame, actual, segments, backtest_cfg.report_horizons)
            with mlflow.start_run(run_name=f"{track}-baseline_{model}"):
                mlflow.log_params(
                    {
                        "track": track,
                        "model": f"baseline_{model}",
                        "test_days": backtest_cfg.test_days,
                        "horizon_max": model_cfg.horizon_max,
                        "n_series": len(panel.keys),
                        "n_origins": len(origin_idx),
                        **(
                            {
                                "ets_train_days": model_cfg.ets_train_days,
                                "refit_every_days": model_cfg.retrain_every_days,
                                "n_jobs": model_cfg.n_jobs,
                            }
                            if model == "C"
                            else {}
                        ),
                    }
                )
                mlflow.log_metrics({**evaluation.headline(metrics), "runtime_seconds": seconds})
                mlflow.log_table(metrics, artifact_file=f"metrics_{track}_baseline_{model}.json")
            logger.info(
                "%s baseline_%s: %.1fs, %s", track, model, seconds, evaluation.headline(metrics)
            )

        _save_forecasts(track, pd.concat(frames, ignore_index=True))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Backtest the baseline forecasters.")
    parser.add_argument("--track", choices=[*TRACKS, "all"], default="all")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(TRACKS if args.track == "all" else (args.track,), tuple(args.models))


if __name__ == "__main__":
    main()
