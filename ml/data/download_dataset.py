"""
Download the AI4I 2020 Predictive Maintenance Dataset from UCI ML Repository.

Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
Records: 10,000
Features: 14 (6 sensor features + metadata + failure flags)
Target: Machine failure (binary) + failure type (multi-class)
"""

import urllib.request
import zipfile
import os
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
ZIP_PATH = RAW_DIR / "ai4i2020.zip"
CSV_FILENAME = "ai4i2020.csv"


def download_dataset() -> Path:
    """Download and extract the AI4I 2020 dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / CSV_FILENAME

    if csv_path.exists():
        print(f"Dataset already exists at {csv_path}")
        return csv_path

    print(f"Downloading dataset from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    print("Download complete. Extracting ...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)

    # The UCI zip may contain the CSV under a sub-folder; find it
    found = list(RAW_DIR.rglob("*.csv"))
    if not found:
        raise FileNotFoundError("CSV not found in extracted archive.")

    extracted = found[0]
    extracted.rename(csv_path)
    ZIP_PATH.unlink(missing_ok=True)

    print(f"Dataset saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    download_dataset()
