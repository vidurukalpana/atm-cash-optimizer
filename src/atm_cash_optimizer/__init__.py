"""Bank-side cash planning for ATMs and Cash Recycler Machines (CRMs)."""

import logging

__version__ = "0.1.0"


def setup_logging() -> None:
    """Log INFO messages to the terminal; called by every command-line entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
