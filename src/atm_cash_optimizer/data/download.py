"""Download NN5 Daily (with missing values) from Zenodo record 4656110 into ``data/raw/``."""

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
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_nn5(data_dir: str | Path | None = None) -> Path:
    """Return the path of the NN5 ``.tsf`` file, downloading it (MD5-checked) if missing."""
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
