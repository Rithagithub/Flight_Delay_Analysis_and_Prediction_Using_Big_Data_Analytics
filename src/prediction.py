"""
Prediction Module
Loads the saved PySpark MLlib PipelineModel and performs interactive,
real-time flight delay predictions from user-selected flight parameters.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from src.spark_utils import get_spark_session, CARRIER_NAMES


class FlightDelayPredictor:
    """
    Flight Delay Predictor using trained PySpark MLlib PipelineModel.
    """

    def __init__(self, model_path: str = "models/flight_delay_model", spark: Optional[SparkSession] = None):
        """
        Initializes the predictor by loading the trained PySpark MLlib PipelineModel.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. Please run training first via 'python src/train_model.py'."
            )
        
        self.spark = spark if spark is not None else get_spark_session("FlightDelayPredictor")
        print(f"[INFO] Loading trained PipelineModel from: {self.model_path}")
        self.model = PipelineModel.load(str(self.model_path))
        print(f"[INFO] PipelineModel loaded successfully.")

    def predict(
        self,
        carrier: str,
        origin_airport: str,
        dest_airport: str,
        departure_hour: int,
        day_of_week: int,
        month: int,
        day_of_month: int = 15
    ) -> Dict[str, Any]:
        """
        Runs flight delay prediction for given flight attributes.
        
        Args:
            carrier (str): Airline carrier code (e.g. 'AA', 'DL', 'WN').
            origin_airport (str): Name of Origin airport.
            dest_airport (str): Name of Destination airport.
            departure_hour (int): Scheduled departure hour (0-23).
            day_of_week (int): Day of week (1=Mon ... 7=Sun).
            month (int): Month (1-12).
            day_of_month (int): Day of month (1-31).
            
        Returns:
            Dict[str, Any]: Prediction result containing prediction label, human readable text, and probabilities.
        """
        # Create single-row Spark DataFrame matching the training schema exactly
        input_data = [(
            str(carrier),
            str(origin_airport),
            str(dest_airport),
            int(departure_hour),
            int(day_of_week),
            int(month),
            int(day_of_month)
        )]
        
        schema = [
            "Carrier",
            "OriginAirportName",
            "DestAirportName",
            "DepartureHour",
            "DayOfWeek",
            "Month",
            "DayofMonth"
        ]

        df_input = self.spark.createDataFrame(input_data, schema=schema)

        # Transform using the loaded MLlib Pipeline
        transformed = self.model.transform(df_input)
        row = transformed.select("prediction", "probability").collect()[0]

        prediction_val = float(row["prediction"])
        prob_vector = row["probability"].toArray()
        prob_ontime = float(prob_vector[0])
        prob_delayed = float(prob_vector[1])

        status_text = "Likely Delayed" if prediction_val == 1.0 else "Likely On Time"
        
        # Determine delay risk tier
        if prob_delayed >= 0.40:
            risk = "High"
        elif prob_delayed >= 0.20:
            risk = "Moderate"
        else:
            risk = "Low"

        carrier_full = CARRIER_NAMES.get(carrier, carrier)

        return {
            "carrier": carrier,
            "carrier_name": carrier_full,
            "origin_airport": origin_airport,
            "dest_airport": dest_airport,
            "departure_hour": departure_hour,
            "day_of_week": day_of_week,
            "month": month,
            "day_of_month": day_of_month,
            "prediction": prediction_val,
            "result_text": status_text,
            "probability_delayed": round(prob_delayed * 100, 2),
            "probability_ontime": round(prob_ontime * 100, 2),
            "risk_level": risk
        }


def predict_flight_delay(
    carrier: str,
    origin_airport: str,
    dest_airport: str,
    departure_hour: int,
    day_of_week: int,
    month: int,
    day_of_month: int = 15,
    model_path: str = "models/flight_delay_model"
) -> Dict[str, Any]:
    """
    Convenience function for one-off flight delay predictions.
    """
    predictor = FlightDelayPredictor(model_path=model_path)
    result = predictor.predict(
        carrier=carrier,
        origin_airport=origin_airport,
        dest_airport=dest_airport,
        departure_hour=departure_hour,
        day_of_week=day_of_week,
        month=month,
        day_of_month=day_of_month
    )
    return result


if __name__ == "__main__":
    predictor = FlightDelayPredictor()
    sample_result = predictor.predict(
        carrier="WN",
        origin_airport="Chicago Midway International",
        dest_airport="Dallas/Fort Worth International",
        departure_hour=18,
        day_of_week=5,
        month=7,
        day_of_month=20
    )
    print("\n[INFO] Sample Flight Delay Prediction Result:")
    for k, v in sample_result.items():
        print(f"  {k}: {v}")
