"""
Data Loader Module
Loads raw flight datasets into PySpark DataFrames with schema inference and validation.
"""

import os
from pathlib import Path
from typing import Optional
from pyspark.sql import DataFrame, SparkSession
from src.spark_utils import get_spark_session


def find_raw_dataset(raw_dir: str = "data/raw") -> Path:
    """
    Scans the raw data folder to automatically discover the flight dataset file.
    
    Args:
        raw_dir (str): Path to raw data directory.
        
    Returns:
        Path: Path to discovered CSV dataset.
        
    Raises:
        FileNotFoundError: If no valid CSV dataset is found.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory '{raw_dir}' does not exist.")

    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found inside '{raw_dir}'. Please place the flight CSV dataset there."
        )

    # Prefer flights.csv if present, otherwise take the first CSV file found
    for f in csv_files:
        if f.name.lower() == "flights.csv":
            return f
    return csv_files[0]


def load_raw_data(
    spark: Optional[SparkSession] = None,
    file_path: Optional[str] = None
) -> tuple[DataFrame, SparkSession]:
    """
    Loads the real flight dataset using PySpark.
    
    Args:
        spark (Optional[SparkSession]): Existing SparkSession or None to create one.
        file_path (Optional[str]): Specific path to CSV file or None for auto-discovery.
        
    Returns:
        tuple[DataFrame, SparkSession]: Loaded PySpark DataFrame and SparkSession.
    """
    if spark is None:
        spark = get_spark_session("FlightDataLoader")

    if file_path is None:
        target_path = find_raw_dataset()
    else:
        target_path = Path(file_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Specified dataset file '{file_path}' does not exist.")

    print(f"[INFO] Loading raw flight dataset from: {target_path}")
    
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(target_path))
    )

    print(f"[INFO] Successfully loaded PySpark DataFrame.")
    print(f"[INFO] Dataset Columns ({len(df.columns)}): {df.columns}")
    return df, spark


if __name__ == "__main__":
    df, spark = load_raw_data()
    print(f"[INFO] Total Rows: {df.count()}")
    df.printSchema()
    df.show(5)
    spark.stop()
