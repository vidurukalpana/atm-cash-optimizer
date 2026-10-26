"""Backtest the LightGBM quantile models, calibrate them and compare with the baselines.

Schedule (per track and target), over the same test period as the baselines:

1. **Retrain dates** every ``quantile_retrain_every_days`` origins, plus one warm-up retrain one
   period before the test starts (its forecasts are only used to calibrate the first test model).
2. **Training rows** at retrain date ``r``: feature rows with ``origin_date <= r − horizon_max``
   (labels fully observed), from the last ``quantile_train_days`` days, every
   ``train_origin_step``-th origin.
3. **Predict** every origin from ``r`` until the next retrain; sort quantiles per row.
4. **Calibrate**: shifts estimated on the most recent out-of-sample window (see
   :mod:`.calibration`) are added, then quantiles are sorted again.

Outputs: ``forecasts_<track>`` (``model = quantile_lgbm``), ``calibration_<track>`` (the
corrections), ``reports/forecast_comparison_<track>.md`` and an MLflow run in
``forecast-quantile``.

Run with::

    python -m atm_cash_optimizer.forecasting.train                 # both tracks
    python -m atm_cash_optimizer.forecasting.train --timing 1      # time 1 retrain, write nothing
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path

import pandas as pd

from atm_cash_optimizer.config import BacktestConfig, ModelConfig, load_config
from atm_cash_optimizer.forecasting import calibration, evaluation
from atm_cash_optimizer.forecasting.backtest import (
    TRACKS,
    load_series,
    origin_indices,
    save_forecasts,
)
from atm_cash_optimizer.forecasting.quantile_model import (
    fit_quantile_models,
    predict_quantiles,
    sort_quantiles,
)
from atm_cash_optimizer.io import (
    get_data_dir,
    get_reports_dir,
    markdown_table,
    read_table,
    write_table,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "quantile_lgbm"
EXPERIMENT = "forecast-quantile"
KEYS = ["machine_id", "origin_date", "h", "target"]
SHIFT_COLUMNS = ["target", "h", "quantile", "n", "coverage", "shift", "applied"]


# --------------------------------------------------------------------------------------------
# Data and schedule
# --------------------------------------------------------------------------------------------


def load_features(track: str) -> pd.DataFrame:
    """Read ``features_<track>`` (all targets) with dates as datetimes and floats as float32."""
    df = pd.read_parquet(get_data_dir() / "features" / f"features_{track}")
    df["origin_date"] = pd.to_datetime(df["origin_date"]).astype("datetime64[ns]")
    df["target"] = df["target"].astype(str)
    floats = df.select_dtypes("float64").columns.drop("y_cum")
    df[floats] = df[floats].astype("float32")
    return df


def retrain_schedule(test_origins: pd.DatetimeIndex, every_days: int) -> list[pd.Timestamp]:
    """Warm-up retrain one period before the test, then every ``every_days`` test origins."""
    warm_up = test_origins[0] - pd.Timedelta(days=every_days)
    return [warm_up, *test_origins[::every_days]]


def training_rows(
    features: pd.DataFrame, retrain_date: pd.Timestamp, cfg: ModelConfig
) -> pd.DataFrame:
    """Rows a model retrained on ``retrain_date`` may learn from (labels fully observed).

    Raises:
        AssertionError: If any label would end after ``retrain_date`` (future leakage).
    """
    last_origin = retrain_date - pd.Timedelta(days=cfg.horizon_max)
    first_origin = last_origin - pd.Timedelta(days=cfg.quantile_train_days)
    origin = features["origin_date"]
    sampled = ((origin - pd.Timestamp(0)).dt.days % cfg.train_origin_step) == 0
    rows = features[
        (origin <= last_origin) & (origin > first_origin) & sampled & features["y_cum"].notna()
    ]
    label_end = rows["origin_date"] + pd.to_timedelta(rows["h"], unit="D")
    assert label_end.max() <= retrain_date, "training label after the retrain date"
    return rows


def _calibration_window(
    out_of_sample: pd.DataFrame, retrain_date: pd.Timestamp, window_days: int
) -> pd.DataFrame:
    # Out-of-sample forecasts whose actual is known on the retrain date, most recent window only.
    label_end = out_of_sample["origin_date"] + pd.to_timedelta(out_of_sample["h"], unit="D")
    known = label_end <= retrain_date
    recent = label_end > retrain_date - pd.Timedelta(days=window_days)
    return out_of_sample[known & recent & out_of_sample["actual"].notna()]


# --------------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------------


def backtest_target(
    features: pd.DataFrame,
    schedule: list[pd.Timestamp],
    end: pd.Timestamp,
    cfg: ModelConfig,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling retrain, predict and calibrate for one target.

    Returns:
        ``(raw, calibrated, shifts)``: raw and calibrated forecasts (``KEYS``, ``actual`` and
        quantile columns) for every origin from the first retrain date to ``end``, and the
        shifts estimated at each retrain date.
    """
    raw_parts, calibrated_parts, shift_parts = [], [], []
    for k, retrain_date in enumerate(schedule):
        next_date = schedule[k + 1] if k + 1 < len(schedule) else end + pd.Timedelta(days=1)
        started = time.perf_counter()
        train = training_rows(features, retrain_date, cfg)
        models = fit_quantile_models(train, cfg, params)

        origin = features["origin_date"]
        rows = features[(origin >= retrain_date) & (origin < next_date)]
        raw = rows[KEYS].assign(actual=rows["y_cum"].to_numpy())
        raw = raw.join(sort_quantiles(predict_quantiles(models, rows)))

        history = pd.concat(raw_parts) if raw_parts else raw.iloc[:0]
        window = _calibration_window(history, retrain_date, cfg.quantile_retrain_every_days)
        shifts = (
            calibration.conformal_shifts(window, cfg.quantiles, cfg.calibration_tolerance)
            if len(window)
            else pd.DataFrame(columns=SHIFT_COLUMNS)
        )
        quantile_cols = [evaluation.quantile_column(q) for q in cfg.quantiles]
        calibrated = calibration.apply_shifts(raw, shifts)
        calibrated[quantile_cols] = sort_quantiles(calibrated[quantile_cols])

        raw_parts.append(raw)
        calibrated_parts.append(calibrated)
        shift_parts.append(shifts.assign(retrain_date=retrain_date))
        logger.info(
            "  retrain %s: %d training rows, %d forecasts, %d shifts applied, %.1fs",
            retrain_date.date(),
            len(train),
            len(raw),
            int(shifts["applied"].astype(bool).sum()),
            time.perf_counter() - started,
        )
    return pd.concat(raw_parts), pd.concat(calibrated_parts), pd.concat(shift_parts)


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def comparison_report(
    track: str,
    metrics: pd.DataFrame,
    pinball: pd.DataFrame,
    coverage_raw: pd.DataFrame,
    coverage_calibrated: pd.DataFrame,
    shifts: pd.DataFrame,
) -> str:
    """Markdown text of ``reports/forecast_comparison_<track>.md``.

    All models are compared on their median (MAE, sMAPE) — the only forecast every model has.
    The range of ``quantile_lgbm`` is judged on its own: pinball loss per quantile and coverage.
    """
    overall = metrics[metrics["segment_type"] == "all"]
    sections = [
        f"# Forecast comparison — {track} track\n",
        "Generated by `python -m atm_cash_optimizer.forecasting.train`.\n",
        "## Median forecast — all models\n\n"
        "Every model is compared on its median (`q50`). Lower is better. sMAPE is unreliable "
        "for `net` (values near zero), so read MAE there.\n",
    ]
    for target in sorted(overall["target"].unique()):
        rows = overall[overall["target"] == target]
        for metric, label, fmt in (
            ("mae", "MAE (LKR)", "{:,.0f}"),
            ("smape", "sMAPE (%)", "{:.1f}"),
        ):
            table = rows.pivot_table(index="model", columns="h", values=metric)
            table.columns = [f"h={h}" for h in table.columns]
            table = markdown_table(table.reset_index(), fmt)
            sections.append(f"### {target}: {label}\n\n{table}\n")

    sections.append(
        f"## Range forecast — `{MODEL_NAME}` only\n\n"
        "The baselines give one number, so only this model has a range. Pinball loss per "
        "quantile (lower is better; outer quantiles naturally have smaller values, so compare "
        "within a row, across horizons or between runs).\n"
    )
    for target in sorted(pinball["target"].unique()):
        table = pinball[pinball["target"] == target].pivot_table(
            index="quantile", columns="h", values="pinball"
        )
        table.columns = [f"h={h}" for h in table.columns]
        table.index = [evaluation.quantile_column(q) for q in table.index]
        table = markdown_table(table.rename_axis("quantile").reset_index(), "{:,.0f}")
        sections.append(f"### {target}: pinball loss (LKR)\n\n{table}\n")

    for name, cov in (
        ("before calibration", coverage_raw),
        ("after calibration", coverage_calibrated),
    ):
        sections.append(
            f"## Coverage of `{MODEL_NAME}` quantiles {name}\n\n"
            "Share of actuals at or below each quantile; should be close to its level.\n\n"
            f"{markdown_table(cov, '{:.3f}')}\n"
        )
    applied = int(shifts["applied"].fillna(False).astype(bool).sum())
    sections.append(
        "## Calibration corrections\n\n"
        f"{applied} of {len(shifts)} checks (retrain × target × horizon × quantile) needed a "
        f"shift. Details in the `calibration_{track}` table.\n"
    )
    return "\n".join(sections)


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


def _coverage(
    forecasts: pd.DataFrame,
    actual: pd.DataFrame,
    segments: pd.DataFrame,
    model_cfg: ModelConfig,
    backtest_cfg: BacktestConfig,
) -> pd.DataFrame:
    merged = forecasts.drop(columns="actual").merge(actual, on=evaluation.KEYS)
    merged = merged.merge(segments, on=evaluation.KEYS)
    merged = merged[merged["h"].isin(backtest_cfg.report_horizons) & merged["actual"].notna()]
    return calibration.coverage_by_segment(
        merged, model_cfg.quantiles, list(evaluation.SEGMENT_TYPES)
    )


def _log_mlflow(
    track: str,
    model_cfg: ModelConfig,
    n_retrains: int,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    shifts: pd.DataFrame,
    report: Path,
    seconds: float,
) -> None:
    import mlflow

    settings = (
        "quantile_retrain_every_days",
        "quantile_train_days",
        "train_origin_step",
        "calibration_tolerance",
        "n_jobs",
        "seed",
    )
    lgbm = metrics[(metrics["model"] == MODEL_NAME) & (metrics["segment_type"] == "all")]
    overall_coverage = coverage[coverage["segment_type"] == "all"]
    values = {**evaluation.headline(lgbm), "runtime_seconds": seconds}
    for row in lgbm.itertuples():
        values[f"{row.target}_pinball_avg_h{row.h}"] = row.pinball  # mean over all quantiles
    for row in overall_coverage.itertuples():
        for column in (c for c in overall_coverage.columns if c.startswith("cov_")):
            values[f"{row.target}_{column}"] = getattr(row, column)

    with mlflow.start_run(run_name=f"{track}-{MODEL_NAME}"):
        mlflow.log_params(
            {
                "track": track,
                "model": MODEL_NAME,
                "retrains": n_retrains,
                **{name: getattr(model_cfg, name) for name in settings},
                **{f"lgbm_{k}": v for k, v in model_cfg.lightgbm.model_dump().items()},
            }
        )
        mlflow.log_metrics(values)
        mlflow.log_artifact(str(report))
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / f"calibration_{track}.csv"
            shifts.to_csv(csv, index=False)
            mlflow.log_artifact(str(csv))


def run_track(
    track: str, model_cfg: ModelConfig, backtest_cfg: BacktestConfig, timing: int | None = None
) -> None:
    """Backtest, calibrate, evaluate and report one track.

    With ``timing = N`` only the first N retrains run, a time estimate is logged and nothing is
    written.
    """
    started = time.perf_counter()
    panel = load_series(track)
    origin_idx = origin_indices(len(panel.dates), backtest_cfg.test_days)
    test_origins = panel.dates[origin_idx]
    full_schedule = retrain_schedule(test_origins, model_cfg.quantile_retrain_every_days)
    schedule = full_schedule[:timing] if timing else full_schedule
    features = load_features(track)

    parts = []
    for target in sorted(features["target"].unique()):
        logger.info("%s / %s: %d retrains", track, target, len(schedule))
        target_rows = features[features["target"] == target]
        parts.append(backtest_target(target_rows, schedule, test_origins[-1], model_cfg))
    seconds = time.perf_counter() - started
    if timing:
        estimate = seconds / timing * len(full_schedule) / 60
        logger.info(
            "TIMING %s: %d retrain(s) in %.0fs, full track ≈ %.0f min",
            track,
            timing,
            seconds,
            estimate,
        )
        return

    def test_period(frames: list[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(frames, ignore_index=True)
        return df[df["origin_date"] >= test_origins[0]]

    raw = test_period([p[0] for p in parts])
    calibrated = test_period([p[1] for p in parts])
    shifts = pd.concat([p[2] for p in parts], ignore_index=True)

    quantile_cols = [evaluation.quantile_column(q) for q in model_cfg.quantiles]
    forecasts = calibrated[KEYS + quantile_cols].assign(model=MODEL_NAME)
    save_forecasts(track, forecasts[["model", *KEYS, *quantile_cols]])
    write_table(shifts, f"calibration_{track}")

    actual = evaluation.cumulative_actuals_frame(panel, origin_idx, model_cfg.horizon_max)
    segments = evaluation.segment_frame(track, panel, origin_idx, model_cfg.horizon_max)
    metrics = evaluation.evaluate(
        read_table(f"forecasts_{track}"), actual, segments, backtest_cfg.report_horizons
    )
    coverage_raw = _coverage(raw, actual, segments, model_cfg, backtest_cfg)
    coverage_calibrated = _coverage(calibrated, actual, segments, model_cfg, backtest_cfg)
    pinball = evaluation.pinball_by_quantile(
        calibrated.drop(columns="actual").merge(actual, on=evaluation.KEYS),
        model_cfg.quantiles,
        backtest_cfg.report_horizons,
    )

    report = get_reports_dir() / f"forecast_comparison_{track}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        comparison_report(track, metrics, pinball, coverage_raw, coverage_calibrated, shifts),
        encoding="utf-8",
    )
    _log_mlflow(
        track, model_cfg, len(schedule), metrics, coverage_calibrated, shifts, report, seconds
    )
    logger.info("%s done in %.0fs; report %s", track, seconds, report)


def run(tracks: tuple[str, ...] = TRACKS, timing: int | None = None) -> None:
    """Run :func:`run_track` for each track."""
    model_cfg = load_config("model")
    backtest_cfg = load_config("backtest")
    if not timing:
        evaluation.use_mlflow_experiment(EXPERIMENT)
    for track in tracks:
        run_track(track, model_cfg, backtest_cfg, timing)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Backtest the LightGBM quantile models.")
    parser.add_argument("--track", choices=[*TRACKS, "all"], default="all")
    parser.add_argument(
        "--timing", type=int, metavar="N", help="only run N retrains and print a time estimate"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(TRACKS if args.track == "all" else (args.track,), args.timing)


if __name__ == "__main__":
    main()
