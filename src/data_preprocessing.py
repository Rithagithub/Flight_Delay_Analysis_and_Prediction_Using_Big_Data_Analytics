"""
Data Preprocessing Module
Cleans raw flight data, handles missing values/duplicates/cancellations,
engineers pre-departure features, defines the target variable (IsDelayed),
and persists the processed dataset.
"""

import os
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, when, floor, concat_ws, lit, create_map
)
from itertools import chain
from src.spark_utils import get_spark_session, CARRIER_NAMES
from src.data_loader import load_raw_data


def preprocess_flight_data(
    df: DataFrame,
    save_processed: bool = True,
    processed_dir: str = "data/processed"
) -> DataFrame:
    """
    Cleans raw flight data and creates feature engineering columns using PySpark.
    
    Processing Steps:
    1. Removes duplicate records.
    2. Filters out cancelled flights (Cancelled == 0).
    3. Handles missing/null values in departure delay and categorical columns.
    4. Extracts DepartureHour from CRSDepTime (HHMM format -> HH).
    5. Extracts ArrivalHour from CRSArrTime (HHMM format -> HH).
    6. Constructs Route feature (Origin -> Destination).
    7. Maps Carrier code to readable Airline Name.
    8. Defines target variable IsDelayed: 1.0 if DepDelay > 15 minutes else 0.0.
    
    Args:
        df (DataFrame): Raw PySpark DataFrame.
        save_processed (bool): Whether to persist processed dataset to disk.
        processed_dir (str): Output folder for processed data.
        
    Returns:
        DataFrame: Cleaned and feature-engineered PySpark DataFrame.
    """
    original_count = df.count()
    print(f"\n[INFO] Starting Data Preprocessing...")
    print(f"[INFO] Initial Raw Records Count: {original_count:,}")

    # 1. Drop duplicate rows
    df_clean = df.dropDuplicates()
    after_dedup_count = df_clean.count()
    duplicates_removed = original_count - after_dedup_count
    print(f"[INFO] Duplicate Rows Removed: {duplicates_removed:,}")

    # 2. Filter out cancelled flights (Cancelled == 0)
    # Cancelled flights never departed, so predicting departure delay applies only to operated flights
    if "Cancelled" in df_clean.columns:
        df_clean = df_clean.filter(col("Cancelled") == 0)

    # 3. Filter missing/null values in critical columns
    required_cols = ["DepDelay", "Carrier", "OriginAirportName", "DestAirportName", "CRSDepTime", "Month", "DayOfWeek"]
    for c in required_cols:
        if c in df_clean.columns:
            df_clean = df_clean.filter(col(c).isNotNull())

    # 4. Feature Engineering: DepartureHour and ArrivalHour
    # CRSDepTime is in integer format HHMM (e.g. 1539 -> 15, 710 -> 7, 2400 -> 0)
    df_clean = df_clean.withColumn("DepartureHour", floor(col("CRSDepTime") / 100))
    df_clean = df_clean.withColumn(
        "DepartureHour",
        when(col("DepartureHour") >= 24, 0).otherwise(col("DepartureHour"))
    )

    if "CRSArrTime" in df_clean.columns:
        df_clean = df_clean.withColumn("ArrivalHour", floor(col("CRSArrTime") / 100))
        df_clean = df_clean.withColumn(
            "ArrivalHour",
            when(col("ArrivalHour") >= 24, 0).otherwise(col("ArrivalHour"))
        )

    # 5. Feature Engineering: Route
    df_clean = df_clean.withColumn(
        "Route",
        concat_ws(" -> ", col("OriginAirportName"), col("DestAirportName"))
    )

    # 6. Map Carrier code to full Airline Name
    carrier_mapping_expr = create_map([lit(x) for x in chain(*CARRIER_NAMES.items())])
    df_clean = df_clean.withColumn(
        "AirlineName",
        when(carrier_mapping_expr[col("Carrier")].isNotNull(), carrier_mapping_expr[col("Carrier")])
        .otherwise(col("Carrier"))
    )

    # 7. Target Variable Definition:
    # Standard FAA/DOT Definition: Flight is considered delayed if departure delay > 15 minutes
    df_clean = df_clean.withColumn(
        "IsDelayed",
        when(col("DepDelay") > 15, 1.0).otherwise(0.0)
    )

    # Ensure numeric data types for MLlib
    df_clean = df_clean.withColumn("DepartureHour", col("DepartureHour").cast("int"))
    df_clean = df_clean.withColumn("DayOfWeek", col("DayOfWeek").cast("int"))
    df_clean = df_clean.withColumn("Month", col("Month").cast("int"))
    df_clean = df_clean.withColumn("DayofMonth", col("DayofMonth").cast("int"))
    df_clean = df_clean.withColumn("DepDelay", col("DepDelay").cast("double"))
    df_clean = df_clean.withColumn("IsDelayed", col("IsDelayed").cast("double"))

    cleaned_count = df_clean.count()
    delayed_count = df_clean.filter(col("IsDelayed") == 1.0).count()
    ontime_count = cleaned_count - delayed_count
    delay_rate = (delayed_count / cleaned_count) * 100 if cleaned_count > 0 else 0.0

    print(f"\n" + "=" * 55)
    print(f"       DATA PREPROCESSING SUMMARY RESULTS")
    print(f"=" * 55)
    print(f" Original Records Count : {original_count:,}")
    print(f" Cleaned Valid Records  : {cleaned_count:,}")
    print(f" Delayed Flights (>15m) : {delayed_count:,} ({delay_rate:.2f}%)")
    print(f" On-Time Flights (<=15m): {ontime_count:,} ({100 - delay_rate:.2f}%)")
    print(f"=" * 55)

    # 8. Save Processed Data
    if save_processed:
        out_dir = Path(processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete cleaned parquet dataset
        parquet_path = out_dir / "flights_clean.parquet"
        print(f"[INFO] Saving processed dataset to Parquet: {parquet_path}")
        df_clean.coalesce(1).write.mode("overwrite").parquet(str(parquet_path))

        # Save sampled CSV (50,000 records) for rapid dashboard visualization & inspection
        sample_csv_path = out_dir / "flights_sample.csv"
        print(f"[INFO] Saving lightweight inspection sample CSV: {sample_csv_path}")
        sample_pd = df_clean.sample(fraction=0.2, seed=42).limit(50000).toPandas()
        sample_pd.to_csv(str(sample_csv_path), index=False)
        print(f"[INFO] Sample CSV saved ({len(sample_pd):,} rows).")

    return df_clean


if __name__ == "__main__":
    df_raw, spark = load_raw_data()
    df_clean = preprocess_flight_data(df_raw, save_processed=True)
    spark.stop()
