"""
Flight Delay Analysis and Prediction Using Big Data Analytics
Master CLI Execution Script

Usage:
    python run.py --all           # Runs end-to-end pipeline (Preprocessing, Analytics, Training, Test Prediction)
    python run.py --preprocess    # Runs Data Loading & Preprocessing
    python run.py --analyze       # Runs Big Data Analytics & Chart Generation
    python run.py --train         # Trains and Evaluates PySpark MLlib Random Forest Model
    python run.py --predict       # Runs Sample Real-Time Delay Prediction
    python run.py --dashboard     # Launches Streamlit Web Dashboard
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.spark_utils import get_spark_session
from src.data_loader import load_raw_data
from src.data_preprocessing import preprocess_flight_data
from src.analysis import run_flight_analytics
from src.train_model import train_and_evaluate_model
from src.prediction import predict_flight_delay


def main():
    parser = argparse.ArgumentParser(
        description="Flight Delay Analysis & Prediction Master Runner"
    )
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (Preprocess, Analyze, Train, Predict)")
    parser.add_argument("--preprocess", action="store_true", help="Clean raw dataset and create features")
    parser.add_argument("--analyze", action="store_true", help="Execute PySpark Big Data analytics and create charts")
    parser.add_argument("--train", action="store_true", help="Train PySpark MLlib Random Forest model")
    parser.add_argument("--predict", action="store_true", help="Run interactive sample flight delay prediction")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit Web Dashboard")

    args = parser.parse_args()

    # Default to --all if no arguments provided
    if not any([args.all, args.preprocess, args.analyze, args.train, args.predict, args.dashboard]):
        args.all = True

    print("\n" + "#" * 70)
    print("  FLIGHT DELAY ANALYSIS AND PREDICTION USING BIG DATA ANALYTICS")
    print("#" * 70)

    spark = None

    if args.all or args.preprocess or args.analyze or args.train:
        spark = get_spark_session("FlightDelayMasterRunner")

    # 1. Preprocessing Step
    df_clean = None
    if args.all or args.preprocess:
        print("\n>>> [STEP 1/4] Data Loading & Preprocessing...")
        df_raw, _ = load_raw_data(spark)
        df_clean = preprocess_flight_data(df_raw, save_processed=True)

    # 2. Analytics Step
    if args.all or args.analyze:
        print("\n>>> [STEP 2/4] PySpark Big Data Analytics...")
        if df_clean is None:
            parquet_path = PROJECT_ROOT / "data" / "processed" / "flights_clean.parquet"
            if parquet_path.exists():
                df_clean = spark.read.parquet(str(parquet_path))
            else:
                df_raw, _ = load_raw_data(spark)
                df_clean = preprocess_flight_data(df_raw, save_processed=True)
        run_flight_analytics(df_clean)

    # 3. Model Training Step
    if args.all or args.train:
        print("\n>>> [STEP 3/4] PySpark MLlib Model Training & Evaluation...")
        if df_clean is None:
            parquet_path = PROJECT_ROOT / "data" / "processed" / "flights_clean.parquet"
            if parquet_path.exists():
                df_clean = spark.read.parquet(str(parquet_path))
            else:
                df_raw, _ = load_raw_data(spark)
                df_clean = preprocess_flight_data(df_raw, save_processed=True)
        train_and_evaluate_model(df_clean)

    # 4. Prediction Verification Step
    if args.all or args.predict:
        print("\n>>> [STEP 4/4] Verifying Real-Time Flight Delay Prediction...")
        sample_res = predict_flight_delay(
            carrier="AA",
            origin_airport="Dallas/Fort Worth International",
            dest_airport="Los Angeles International",
            departure_hour=17,
            day_of_week=5,
            month=7,
            day_of_month=25
        )
        print("\n[PREDICTION RESULT]")
        for k, v in sample_res.items():
            print(f"  {k:22}: {v}")

    if spark is not None:
        spark.stop()

    # 5. Dashboard Launch Step
    if args.dashboard:
        print("\n>>> Launching Streamlit Web Dashboard...")
        subprocess.run(["streamlit", "run", "dashboard/app.py"], check=True)

    print("\n" + "#" * 70)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("  To view the dashboard, run: streamlit run dashboard/app.py")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
