"""
Big Data Analytics Module
Performs scalable distributed analytics using PySpark DataFrame APIs
(select, filter, groupBy, agg, orderBy, withColumn) and generates interactive
visualizations with Plotly.
"""

import os
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, avg, round, sum as spark_sum, when
import plotly.express as px
import plotly.graph_objects as go
from src.spark_utils import get_spark_session
from src.data_loader import load_raw_data
from src.data_preprocessing import preprocess_flight_data


def run_flight_analytics(df: DataFrame, charts_dir: str = "outputs/charts") -> dict:
    """
    Executes full Big Data analytics pipeline on flight dataset using PySpark aggregations.
    
    Analyses included:
    1. Overall summary statistics (KPIs)
    2. Airline performance (delay rate, average delay, total volume)
    3. Airport delay analysis (origin & destination delay hotspots)
    4. Temporal patterns (monthly seasonality, day of week trends, hourly patterns)
    5. Route congestion analysis (top delayed routes)
    
    Args:
        df (DataFrame): Cleaned PySpark DataFrame.
        charts_dir (str): Output folder for saving generated charts.
        
    Returns:
        dict: Aggregated analytics results as pandas DataFrames/dictionaries.
    """
    out_dir = Path(charts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("       STARTING PYSPARK BIG DATA ANALYTICS")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. OVERALL METRICS
    # -------------------------------------------------------------
    total_flights = df.count()
    delayed_flights = df.filter(col("IsDelayed") == 1.0).count()
    ontime_flights = total_flights - delayed_flights
    delay_rate = (delayed_flights / total_flights) * 100
    avg_delay = df.select(avg("DepDelay")).collect()[0][0] or 0.0

    print(f"\n[1] Overall Flight Metrics:")
    print(f"    - Total Flights Analyzed: {total_flights:,}")
    print(f"    - Total Delayed (>15 min): {delayed_flights:,} ({delay_rate:.2f}%)")
    print(f"    - Total On-Time (<=15 min): {ontime_flights:,} ({100 - delay_rate:.2f}%)")
    print(f"    - Average Departure Delay: {avg_delay:.2f} minutes")

    overall_kpis = {
        "total_flights": total_flights,
        "delayed_flights": delayed_flights,
        "ontime_flights": ontime_flights,
        "delay_rate": delay_rate,
        "avg_delay": avg_delay
    }

    # -------------------------------------------------------------
    # 2. AIRLINE PERFORMANCE ANALYSIS
    # -------------------------------------------------------------
    airline_agg = (
        df.groupBy("Carrier", "AirlineName")
        .agg(
            count("*").alias("TotalFlights"),
            spark_sum("IsDelayed").alias("DelayedFlights"),
            round(avg("IsDelayed") * 100, 2).alias("DelayRatePct"),
            round(avg("DepDelay"), 2).alias("AvgDelayMin")
        )
        .orderBy(col("DelayRatePct").desc())
    )
    airline_df = airline_agg.toPandas()
    print(f"\n[2] Top 5 Airlines with Highest Delay Rates:")
    print(airline_df[["AirlineName", "TotalFlights", "DelayRatePct", "AvgDelayMin"]].head(5).to_string(index=False))

    # -------------------------------------------------------------
    # 3. AIRPORT DELAY ANALYSIS (ORIGIN & DESTINATION)
    # -------------------------------------------------------------
    origin_agg = (
        df.groupBy("OriginAirportName")
        .agg(
            count("*").alias("TotalFlights"),
            spark_sum("IsDelayed").alias("DelayedFlights"),
            round(avg("IsDelayed") * 100, 2).alias("DelayRatePct"),
            round(avg("DepDelay"), 2).alias("AvgDelayMin")
        )
        .filter(col("TotalFlights") >= 500) # Filter significant airports
        .orderBy(col("DelayRatePct").desc())
    )
    origin_df = origin_agg.toPandas()
    print(f"\n[3] Top 5 Origin Airports with Highest Delay Rate:")
    print(origin_df[["OriginAirportName", "TotalFlights", "DelayRatePct", "AvgDelayMin"]].head(5).to_string(index=False))

    dest_agg = (
        df.groupBy("DestAirportName")
        .agg(
            count("*").alias("TotalFlights"),
            spark_sum("IsDelayed").alias("DelayedFlights"),
            round(avg("IsDelayed") * 100, 2).alias("DelayRatePct"),
            round(avg("DepDelay"), 2).alias("AvgDelayMin")
        )
        .filter(col("TotalFlights") >= 500)
        .orderBy(col("DelayRatePct").desc())
    )
    dest_df = dest_agg.toPandas()

    # -------------------------------------------------------------
    # 4. TEMPORAL ANALYSIS (MONTH, DAY OF WEEK, DEPARTURE HOUR)
    # -------------------------------------------------------------
    month_agg = (
        df.groupBy("Month")
        .agg(
            count("*").alias("TotalFlights"),
            round(avg("IsDelayed") * 100, 2).alias("DelayRatePct"),
            round(avg("DepDelay"), 2).alias("AvgDelayMin")
        )
        .orderBy("Month")
    )
    month_df = month_agg.toPandas()

    dow_agg = (
        df.groupBy("DayOfWeek")
        .agg(
            count("*").alias("TotalFlights"),
            round(avg("IsDelayed") * 100, 2).alias("DelayRatePct"),
            round(avg("DepDelay"), 2).alias("AvgDelayMin")
        )
        .orderBy("DayOfWeek")
    )
    dow_df = dow_agg.toPandas()
    dow_map = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    dow_df["DayName"] = dow_df["DayOfWeek"].map(dow_map)

    hour_agg = (
        df.groupBy("DepartureHour")
        .agg(
            count("*").alias("TotalFlights"),
            round(avg("IsDelayed") * 100, 2).alias("DelayRatePct"),
            round(avg("DepDelay"), 2).alias("AvgDelayMin")
        )
        .orderBy("DepartureHour")
    )
    hour_df = hour_agg.toPandas()
    print(f"\n[4] Hourly Delay Pattern Summary:")
    print(hour_df[["DepartureHour", "TotalFlights", "DelayRatePct", "AvgDelayMin"]].head(6).to_string(index=False))

    # -------------------------------------------------------------
    # 5. ROUTE DELAY ANALYSIS
    # -------------------------------------------------------------
    route_agg = (
        df.groupBy("Route")
        .agg(
            count("*").alias("TotalFlights"),
            spark_sum("IsDelayed").alias("DelayedFlights"),
            round(avg("IsDelayed") * 100, 2).alias("DelayRatePct"),
            round(avg("DepDelay"), 2).alias("AvgDelayMin")
        )
        .filter(col("TotalFlights") >= 100) # Meaningful routes
        .orderBy(col("DelayRatePct").desc())
    )
    route_df = route_agg.toPandas()
    print(f"\n[5] Top 5 Most Delayed Flight Routes:")
    print(route_df[["Route", "TotalFlights", "DelayRatePct", "AvgDelayMin"]].head(5).to_string(index=False))

    # -------------------------------------------------------------
    # 6. GENERATE & SAVE PLOTLY VISUALIZATIONS
    # -------------------------------------------------------------
    print(f"\n[INFO] Generating interactive charts and saving to {charts_dir}...")

    # Chart 1: Delayed vs On-Time Flights (Donut Chart)
    fig_overview = go.Figure(data=[go.Pie(
        labels=["On-Time (<= 15 min)", "Delayed (> 15 min)"],
        values=[ontime_flights, delayed_flights],
        hole=0.45,
        marker_colors=["#2ecc71", "#e74c3c"],
        textinfo="label+percent",
        hoverinfo="label+value+percent"
    )])
    fig_overview.update_layout(title="Overall Flight Punctuality Distribution", template="plotly_white")
    fig_overview.write_html(str(out_dir / "flight_overview_donut.html"))

    # Chart 2: Delay Rate by Airline
    fig_airline = px.bar(
        airline_df,
        x="DelayRatePct",
        y="AirlineName",
        orientation="h",
        color="DelayRatePct",
        color_continuous_scale="Reds",
        title="Flight Delay Rate by Airline (%)",
        labels={"DelayRatePct": "Delay Rate (%)", "AirlineName": "Airline"},
        template="plotly_white"
    )
    fig_airline.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_airline.write_html(str(out_dir / "delay_rate_by_airline.html"))

    # Chart 3: Hourly Delay Trends
    fig_hour = px.line(
        hour_df,
        x="DepartureHour",
        y="DelayRatePct",
        markers=True,
        title="Flight Delay Rate by Scheduled Departure Hour",
        labels={"DepartureHour": "Departure Hour (24-Hour Format)", "DelayRatePct": "Delay Rate (%)"},
        template="plotly_white"
    )
    fig_hour.write_html(str(out_dir / "delay_rate_by_hour.html"))

    # Chart 4: Day of Week Delay Patterns
    fig_dow = px.bar(
        dow_df,
        x="DayName",
        y="DelayRatePct",
        color="DelayRatePct",
        color_continuous_scale="Viridis",
        title="Flight Delay Rate by Day of Week",
        labels={"DayName": "Day of Week", "DelayRatePct": "Delay Rate (%)"},
        template="plotly_white"
    )
    fig_dow.write_html(str(out_dir / "delay_rate_by_day_of_week.html"))

    # Chart 5: Top 10 Delayed Origin Airports
    top_airports = origin_df.head(10)
    fig_airport = px.bar(
        top_airports,
        x="DelayRatePct",
        y="OriginAirportName",
        orientation="h",
        color="DelayRatePct",
        color_continuous_scale="Plasma",
        title="Top 10 Origin Airports by Delay Rate (%)",
        labels={"DelayRatePct": "Delay Rate (%)", "OriginAirportName": "Origin Airport"},
        template="plotly_white"
    )
    fig_airport.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_airport.write_html(str(out_dir / "top_delayed_airports.html"))

    # Chart 6: Top 10 Delayed Routes
    top_routes = route_df.head(10)
    fig_route = px.bar(
        top_routes,
        x="DelayRatePct",
        y="Route",
        orientation="h",
        color="DelayRatePct",
        color_continuous_scale="Inferno",
        title="Top 10 Most Delayed Flight Routes (%)",
        labels={"DelayRatePct": "Delay Rate (%)", "Route": "Flight Route"},
        template="plotly_white"
    )
    fig_route.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_route.write_html(str(out_dir / "top_delayed_routes.html"))

    print(f"[INFO] All interactive charts saved successfully to '{charts_dir}'.")

    return {
        "kpis": overall_kpis,
        "airline_df": airline_df,
        "origin_df": origin_df,
        "dest_df": dest_df,
        "month_df": month_df,
        "dow_df": dow_df,
        "hour_df": hour_df,
        "route_df": route_df
    }


if __name__ == "__main__":
    df_raw, spark = load_raw_data()
    df_clean = preprocess_flight_data(df_raw, save_processed=True)
    results = run_flight_analytics(df_clean)
    spark.stop()
