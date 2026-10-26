"""Run every pipeline step in order: data → scenario → features → forecasts → simulation.

A full run keeps ``n_jobs`` cores busy for roughly 20 minutes; nothing is scheduled.

Run: ``python -m atm_cash_optimizer.run_all [--steps NAME ...] [--list]``
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable

from atm_cash_optimizer import setup_logging
from atm_cash_optimizer.analysis import clustering
from atm_cash_optimizer.data import ingest
from atm_cash_optimizer.features import build as features
from atm_cash_optimizer.forecasting import baselines, train
from atm_cash_optimizer.scenario import build as scenario
from atm_cash_optimizer.simulation import run as simulation

logger = logging.getLogger(__name__)

STEPS: dict[str, Callable[[], object]] = {
    "ingest": ingest.run,
    "clustering": clustering.run,
    "scenario": scenario.run,
    "features": features.run,
    "baselines": baselines.run,
    "quantile": train.run,
    "simulation_nn5": lambda: simulation.run("nn5"),
    "simulation_lk": lambda: simulation.run("lk"),
}


def run(steps: list[str] | None = None) -> None:
    """Run ``steps`` (default: all) in pipeline order and log each step's time."""
    selected = [name for name in STEPS if steps is None or name in steps]
    total = time.perf_counter()
    for name in selected:
        started = time.perf_counter()
        logger.info("=== %s ===", name)
        STEPS[name]()
        logger.info("=== %s done in %.0fs ===", name, time.perf_counter() - started)
    minutes = (time.perf_counter() - total) / 60
    logger.info("All %d steps done in %.0f min", len(selected), minutes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole pipeline in order.")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), help="only these steps")
    parser.add_argument("--list", action="store_true", help="print the steps and exit")
    args = parser.parse_args()
    if args.list:
        print("\n".join(STEPS))
        return
    setup_logging()
    run(args.steps)


if __name__ == "__main__":
    main()
