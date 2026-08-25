from pathlib import Path

import pandas as pd


def load_raw_data(raw_data_dir: Path | str) -> pd.DataFrame:
    """Load all raw parquet files below a data-lake directory."""
    raw_dir = Path(raw_data_dir)
    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    return pd.concat(
        (pd.read_parquet(path) for path in parquet_files),
        ignore_index=True,
    )