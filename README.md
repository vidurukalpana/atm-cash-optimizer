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

<!-- ## Features (PySpark, needs Java 17)

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


if i wnat to continue this converstaion? how to do that?

You can come back to this conversation from the terminal. Open it in the project folder first, because conversations are saved per folder:

cd "/Users/admin/Desktop/Viduz Diary/My Repositories/atm-cash-optimizer"

Then pick one of these:

┌──────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────┐
│             Command              │                                    What it does                                    │
├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ claude --continue (or claude -c) │ Reopens your most recent conversation in this folder, so this one if it's the last │
├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ claude --resume (or claude -r)   │ Shows a list of past conversations to choose from                                  │
├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────┤
│ /resume                          │ The same list, from inside a running Claude session                                │
└──────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────┘

Even if you start a fresh conversation, the important context isn't lost:
- Your progress is in the repo. docs/DEVELOPMENT_PLAN.md has every finished branch ticked (1–6), and the docs (data.md, eda.md, scenario.md) record the decisions. A new session can read those and pick up at branch 7.
- I keep notes between conversations in my memory files, such as your preference for no Makefile, minimal tests, and that I never stage or commit without asking. Those load automatically in any new conversation in this folder.

A very long conversation gets summarized automatically as it grows, so continuing this one works fine. Starting fresh for a new branch is fine too. Just say "continue with the next branch in the development plan." -->