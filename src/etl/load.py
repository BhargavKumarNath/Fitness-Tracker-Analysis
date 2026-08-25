from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def write_processed_data(df: pd.DataFrame, output_dir: Path | str) -> Path:
    """Write processed records partitioned by year and month."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(destination),
        format="parquet",
        partitioning=["year", "month"],
        existing_data_behavior="overwrite_or_ignore",
    )
    return destination