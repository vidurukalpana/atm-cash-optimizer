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
