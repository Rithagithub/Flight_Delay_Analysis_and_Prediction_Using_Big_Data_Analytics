"""
Model Training & Evaluation Module
Trains a PySpark MLlib RandomForestClassifier with balanced class weighting on 80% training data,
evaluates on 20% test data with authentic classification metrics
(Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix),
and saves the trained PipelineModel to disk.
"""

import os
import json
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from src.spark_utils import get_spark_session
from src.data_loader import load_raw_data
from src.data_preprocessing import preprocess_flight_data
from src.feature_engineering import build_feature_pipeline


def train_and_evaluate_model(
    df: DataFrame,
    model_save_dir: str = "models/flight_delay_model",
    reports_dir: str = "outputs/reports"
) -> dict:
    """
    Trains and evaluates the PySpark MLlib Random Forest flight delay prediction model.
    
    Args:
        df (DataFrame): Cleaned flight DataFrame.
        model_save_dir (str): Directory where trained PipelineModel is persisted.
        reports_dir (str): Directory where evaluation metrics report is saved.
        
    Returns:
        dict: Real computed evaluation metrics.
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 65)
    print("       PYSPARK MLLIB RANDOM FOREST MODEL TRAINING")
    print("=" * 65)

    # 1. 80/20 Train-Test Split with reproducible seed
    print(f"[INFO] Splitting dataset into 80% Training and 20% Testing sets (seed=42)...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"[INFO] Training Records Count : {train_count:,}")
    print(f"[INFO] Testing Records Count  : {test_count:,}")

    # 2. Compute Balanced Class Weights (handling ~19.3% vs ~80.7% class imbalance)
    delayed_train = train_df.filter(col("IsDelayed") == 1.0).count()
    ontime_train = train_count - delayed_train
    w_delayed = train_count / (2.0 * delayed_train) if delayed_train > 0 else 1.0
    w_ontime = train_count / (2.0 * ontime_train) if ontime_train > 0 else 1.0

    print(f"[INFO] Class Distribution in Training Set: Delayed={delayed_train:,}, On-Time={ontime_train:,}")
    print(f"[INFO] Balanced Class Weights: Delayed={w_delayed:.2f}, On-Time={w_ontime:.2f}")

    train_df = train_df.withColumn(
        "classWeight",
        when(col("IsDelayed") == 1.0, lit(w_delayed)).otherwise(lit(w_ontime))
    )

    # 3. Build ML Pipeline stages
    feature_stages, feature_names = build_feature_pipeline()
    
    # 4. Instantiate RandomForestClassifier with class weighting
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="IsDelayed",
        weightCol="classWeight",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=50,
        maxDepth=8,
        seed=42
    )

    pipeline = Pipeline(stages=feature_stages + [rf])

    # 5. Train the Pipeline Model on Spark
    print(f"[INFO] Fitting PySpark MLlib Pipeline on training data...")
    pipeline_model = pipeline.fit(train_df)
    print(f"[INFO] Model training completed successfully.")

    # 6. Model Evaluation on Test Data
    print(f"[INFO] Evaluating model on unseen test data...")
    predictions = pipeline_model.transform(test_df)

    # Multi-class evaluators (Accuracy, Precision, Recall, F1)
    acc_evaluator = MulticlassClassificationEvaluator(labelCol="IsDelayed", predictionCol="prediction", metricName="accuracy")
    prec_evaluator = MulticlassClassificationEvaluator(labelCol="IsDelayed", predictionCol="prediction", metricName="weightedPrecision")
    rec_evaluator = MulticlassClassificationEvaluator(labelCol="IsDelayed", predictionCol="prediction", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="IsDelayed", predictionCol="prediction", metricName="f1")

    # Binary evaluators (ROC-AUC, PR-AUC)
    roc_evaluator = BinaryClassificationEvaluator(labelCol="IsDelayed", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr_evaluator = BinaryClassificationEvaluator(labelCol="IsDelayed", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

    accuracy = acc_evaluator.evaluate(predictions)
    precision = prec_evaluator.evaluate(predictions)
    recall = rec_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)
    roc_auc = roc_evaluator.evaluate(predictions)
    pr_auc = pr_evaluator.evaluate(predictions)

    # 7. Confusion Matrix Calculation
    tp = predictions.filter((predictions.IsDelayed == 1.0) & (predictions.prediction == 1.0)).count()
    fp = predictions.filter((predictions.IsDelayed == 0.0) & (predictions.prediction == 1.0)).count()
    tn = predictions.filter((predictions.IsDelayed == 0.0) & (predictions.prediction == 0.0)).count()
    fn = predictions.filter((predictions.IsDelayed == 1.0) & (predictions.prediction == 0.0)).count()

    print("\n" + "=" * 65)
    print("             REAL MODEL EVALUATION METRICS")
    print("=" * 65)
    print(f" Test Accuracy        : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f" Weighted Precision   : {precision:.4f} ({precision * 100:.2f}%)")
    print(f" Weighted Recall      : {recall:.4f} ({recall * 100:.2f}%)")
    print(f" Weighted F1-Score    : {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print(f" Area Under ROC (AUC) : {roc_auc:.4f}")
    print(f" Area Under PR (AUPR) : {pr_auc:.4f}")
    print("-" * 65)
    print(" Confusion Matrix:")
    print(f"   True Positives  (TP) : {tp:,} (Correctly predicted Delayed)")
    print(f"   True Negatives  (TN) : {tn:,} (Correctly predicted On-Time)")
    print(f"   False Positives (FP) : {fp:,} (On-Time misclassified as Delayed)")
    print(f"   False Negatives (FN) : {fn:,} (Delayed misclassified as On-Time)")
    print("=" * 65)

    metrics_dict = {
        "model_name": "PySpark MLlib RandomForestClassifier",
        "num_trees": 50,
        "max_depth": 8,
        "train_records": train_count,
        "test_records": test_count,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }
    }

    # 8. Save Metrics Report to JSON and TXT
    json_report_path = reports_path / "model_evaluation_report.json"
    with open(json_report_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    txt_report_path = reports_path / "model_evaluation_report.txt"
    with open(txt_report_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write("   FLIGHT DELAY MODEL EVALUATION REPORT (PySpark MLlib)\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Algorithm           : Random Forest Classifier (Balanced Weights)\n")
        f.write(f"Trees & Max Depth   : 50 Trees, Max Depth 8\n")
        f.write(f"Training Sample Size: {train_count:,} flights (80%)\n")
        f.write(f"Testing Sample Size : {test_count:,} flights (20%)\n\n")
        f.write(f"Accuracy            : {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        f.write(f"Weighted Precision  : {precision:.4f} ({precision * 100:.2f}%)\n")
        f.write(f"Weighted Recall     : {recall:.4f} ({recall * 100:.2f}%)\n")
        f.write(f"Weighted F1-Score   : {f1_score:.4f} ({f1_score * 100:.2f}%)\n")
        f.write(f"ROC-AUC             : {roc_auc:.4f}\n")
        f.write(f"PR-AUC              : {pr_auc:.4f}\n\n")
        f.write("Confusion Matrix Breakdown:\n")
        f.write(f"  TP (Delayed as Delayed): {tp:,}\n")
        f.write(f"  TN (On-Time as On-Time): {tn:,}\n")
        f.write(f"  FP (On-Time as Delayed): {fp:,}\n")
        f.write(f"  FN (Delayed as On-Time): {fn:,}\n")

    print(f"[INFO] Evaluation report saved to: {json_report_path}")

    # 9. Save Trained ML Pipeline Model
    model_path = Path(model_save_dir)
    print(f"[INFO] Persisting trained PySpark MLlib PipelineModel to: {model_path}")
    pipeline_model.write().overwrite().save(str(model_path))
    print(f"[INFO] Model saved successfully.")

    # 10. Verify reloadability
    reloaded_model = PipelineModel.load(str(model_path))
    print(f"[INFO] Verified: Model reloaded successfully with {len(reloaded_model.stages)} pipeline stages.")

    return metrics_dict


if __name__ == "__main__":
    spark = get_spark_session("FlightDelayModelTraining")
    
    parquet_path = Path("data/processed/flights_clean.parquet")
    if parquet_path.exists():
        print(f"[INFO] Loading processed parquet data from: {parquet_path}")
        df = spark.read.parquet(str(parquet_path))
    else:
        df_raw, _ = load_raw_data(spark)
        df = preprocess_flight_data(df_raw, save_processed=True)

    metrics = train_and_evaluate_model(df)
    spark.stop()
