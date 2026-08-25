from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, month, to_date, year

from src.config import get_runtime_paths
from src.etl.run import run_etl
from src.etl.transform import transform_data


def main() -> None:
    paths = get_runtime_paths()
    run_etl(paths["raw_data_dir"], paths["processed_data_dir"])


if __name__ == "__main__":
    main()
