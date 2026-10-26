# atm-cash-optimizer

Bank-side cash planning system for ATMs and Cash Recycler Machines: probabilistic demand
forecasting, cost-optimized replenishment by denomination, vendor order files and reconciliation.

> Work in progress — see [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for the roadmap
> and [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) for a plain-language explanation of the project.

## Setup

Requires Python 3.11 (and Java 17/21 for the Spark steps).

```bash
# one-time setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# lint
ruff check .
ruff format --check .

# fast test suite
pytest -m "not slow"
```

## Data

```bash
# download NN5, impute, scale to LKR, validate → data/processed/nn5_daily.parquet + data/warehouse.duckdb
python -m atm_cash_optimizer.data.ingest
```

See [docs/data.md](docs/data.md) for the source, fields and imputation method.

## Exploratory analysis

```bash
# cluster machines → machine_clusters table + reports/eda_summary.md
python -m atm_cash_optimizer.analysis.clustering
```

Charts and SQL queries (`sql/eda_queries.sql`): `notebooks/01_eda.ipynb`. Findings and decisions: [docs/eda.md](docs/eda.md).

## Sri Lanka scenario (simulated)

```bash
# lk_calendar + lk_daily: NN5 moved to 2024–2026 with Sri Lankan holidays, festivals, CRMs, banknotes
python -m atm_cash_optimizer.scenario.build
```

All `lk` data is **simulated** — see [docs/scenario.md](docs/scenario.md).

## Features (PySpark, needs Java 17)

```bash
# origin–horizon training rows → data/features/features_nn5/ and features_lk/ (+ DuckDB views)
python -m atm_cash_optimizer.features.build          # --track nn5 | lk | all
```

Columns and rules are documented in `src/atm_cash_optimizer/features/build.py`.

## Baseline forecasts

```bash
# backtest baselines A/B/C on the last 280 days → forecasts_nn5, forecasts_lk + MLflow (~3 min, 4 cores)
python -m atm_cash_optimizer.forecasting.baselines      # --track nn5|lk|all --models A B C

# browse the results
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

## Quantile forecasts (LightGBM)

```bash
# time one retrain round first (writes nothing), then the full backtest per track
python -m atm_cash_optimizer.forecasting.train --track lk --timing 1
python -m atm_cash_optimizer.forecasting.train --track nn5     # ~2.5 min, 4 cores
python -m atm_cash_optimizer.forecasting.train --track lk      # ~4.5 min, 4 cores
```

Writes `forecasts_<track>` (`model = quantile_lgbm`), `calibration_<track>`,
`reports/forecast_comparison_<track>.md` and MLflow experiment `forecast-quantile`.
