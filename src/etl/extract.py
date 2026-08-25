from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


def load_raw_data(raw_data_dir: Path | str) -> pd.DataFrame:
    """Load all raw parquet files below a data-lake directory."""
    raw_dir = Path(raw_data_dir)
    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")

    dataset = ds.dataset(
        [str(path) for path in parquet_files],
        format="parquet",
        partitioning=None,
    )
    return dataset.to_table().to_pandas()