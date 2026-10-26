"""Download the NN5 Daily dataset (with missing values) from Zenodo.

Source: Monash Time Series Forecasting Archive, Zenodo record 4656110
(https://zenodo.org/records/4656110). The zip is checked against a known MD5 and unzipped into
``data/raw/``. Nothing is downloaded if the ``.tsf`` file is already there.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path

from atm_cash_optimizer.io import get_data_dir

logger = logging.getLogger(__name__)

NN5_URL = (
    "https://zenodo.org/records/4656110/files/nn5_daily_dataset_with_missing_values.zip?download=1"
)
NN5_MD5 = "2529489455b2d9f730fbb3c65d39df88"
NN5_ZIP = "nn5_daily_dataset_with_missing_values.zip"
NN5_TSF = "nn5_daily_dataset_with_missing_values.tsf"


def md5sum(path: Path) -> str:
    """Return the hex MD5 digest of a file."""
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_nn5(data_dir: str | Path | None = None) -> Path:
    """Make sure the NN5 ``.tsf`` file exists in ``data/raw/`` and return its path.

    Raises:
        ValueError: If the downloaded zip does not match :data:`NN5_MD5`.
    """
    raw_dir = get_data_dir(data_dir) / "raw"
    tsf_path = raw_dir / NN5_TSF
    if tsf_path.is_file():
        logger.info("NN5 already present at %s, skipping download", tsf_path)
        return tsf_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / NN5_ZIP
    if not zip_path.is_file():
        logger.info("Downloading NN5 from %s", NN5_URL)
        urllib.request.urlretrieve(NN5_URL, zip_path)

    actual = md5sum(zip_path)
    if actual != NN5_MD5:
        zip_path.unlink()
        raise ValueError(f"MD5 mismatch for {zip_path.name}: expected {NN5_MD5}, got {actual}")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(NN5_TSF, raw_dir)
    logger.info("Extracted %s", tsf_path)
    return tsf_path
