"""
Flight Delay Analytics & Prediction Dashboard
Academic BDA CBP Streamlit Application
Big Data Analytics using PySpark and Machine Learning
"""

import sys
import os
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path so src modules can be imported
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.spark_utils import CARRIER_NAMES
from src.prediction import FlightDelayPredictor

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Flight Delay Analytics & Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .prediction-box-delayed {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
    .prediction-box-ontime {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """
    Loads the sample/processed data for dashboard visualization.
    """
    sample_path = project_root / "data" / "processed" / "flights_sample.csv"
    if not sample_path.exists():
        # Fallback to loading from raw data if sample hasn't been generated yet
        raw_path = project_root / "data" / "raw" / "flights.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path, nrows=50000)
            df = df[df["Cancelled"] == 0]
            df["DepartureHour"] = (df["CRSDepTime"] // 100).clip(0, 23)
            df["IsDelayed"] = (df["DepDelay"] > 15).astype(float)
            df["Route"] = df["OriginAirportName"] + " -> " + df["DestAirportName"]
            df["AirlineName"] = df["Carrier"].map(CARRIER_NAMES).fillna(df["Carrier"])
            return df
        else:
            return None
    
    df = pd.read_csv(sample_path)
    if "AirlineName" not in df.columns:
        df["AirlineName"] = df["Carrier"].map(CARRIER_NAMES).fillna(df["Carrier"])
    return df


@st.cache_data
def load_metrics_report():
    """
    Loads model evaluation metrics from outputs/reports.
    """
    report_path = project_root / "outputs" / "reports" / "model_evaluation_report.json"
    if report_path.exists():
        with open(report_path, "r") as f:
            return json.load(f)
    return None


@st.cache_resource
def get_predictor():
    """
    Caches the PySpark MLlib PipelineModel predictor to avoid reloading on each interaction.
    """
    model_path = project_root / "models" / "flight_delay_model"
    if model_path.exists():
        try:
            return FlightDelayPredictor(model_path=str(model_path))
        except Exception as e:
            st.sidebar.error(f"Error loading PySpark model: {e}")
            return None
    return None


def main():
    # Header Section
    st.markdown('<div class="main-title">✈️ Flight Delay Analytics & Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Big Data Analytics using PySpark and Machine Learning</div>', unsafe_allow_html=True)

    # Load Data
    df = load_data()
    if df is None:
        st.error("⚠️ Dataset not found! Please make sure `data/raw/flights.csv` exists and preprocessing has been run.")
        st.info("Run `python run.py --all` in terminal to execute full preprocessing and training.")
        return

    # Sidebar Navigation & Filters
    st.sidebar.header("🔍 Dashboard Filters")

    # Airline filter
    all_airlines = ["All Airlines"] + sorted(list(df["AirlineName"].dropna().unique()))
    selected_airline = st.sidebar.selectbox("Select Airline:", all_airlines)

    # Origin filter
    all_origins = ["All Origins"] + sorted(list(df["OriginAirportName"].dropna().unique()))
    selected_origin = st.sidebar.selectbox("Select Origin Airport:", all_origins)

    # Destination filter
    all_dests = ["All Destinations"] + sorted(list(df["DestAirportName"].dropna().unique()))
    selected_dest = st.sidebar.selectbox("Select Destination Airport:", all_dests)

    # Month filter
    month_names = {0: "All Months", 1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                   7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
    selected_month_name = st.sidebar.selectbox("Select Month:", list(month_names.values()))
    selected_month = [k for k, v in month_names.items() if v == selected_month_name][0]

    # Day of Week filter
    dow_names = {0: "All Days", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
    selected_dow_name = st.sidebar.selectbox("Select Day of Week:", list(dow_names.values()))
    selected_dow = [k for k, v in dow_names.items() if v == selected_dow_name][0]

    # Apply Filters to Data
    filtered_df = df.copy()
    if selected_airline != "All Airlines":
        filtered_df = filtered_df[filtered_df["AirlineName"] == selected_airline]
    if selected_origin != "All Origins":
        filtered_df = filtered_df[filtered_df["OriginAirportName"] == selected_origin]
    if selected_dest != "All Destinations":
        filtered_df = filtered_df[filtered_df["DestAirportName"] == selected_dest]
    if selected_month != 0:
        filtered_df = filtered_df[filtered_df["Month"] == selected_month]
    if selected_dow != 0:
        filtered_df = filtered_df[filtered_df["DayOfWeek"] == selected_dow]

    # Handle empty filter case
    if len(filtered_df) == 0:
        st.warning("⚠️ No flights match the selected filter combination. Please reset or adjust filters.")
        filtered_df = df

    # Tabs
    tab_analytics, tab_prediction, tab_model = st.tabs([
        "📊 Big Data Analytics",
        "🎯 Predict Flight Delay",
        "🤖 ML Model & Metrics"
    ])

    # =========================================================================
    # TAB 1: BIG DATA ANALYTICS
    # =========================================================================
    with tab_analytics:
        # KPI Metric Cards
        total_fl = len(filtered_df)
        delayed_fl = int(filtered_df["IsDelayed"].sum())
        ontime_fl = total_fl - delayed_fl
        delay_rate_pct = (delayed_fl / total_fl * 100) if total_fl > 0 else 0.0
        avg_dep_delay = filtered_df["DepDelay"].mean() if total_fl > 0 else 0.0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Flights", f"{total_fl:,}")
        col2.metric("Delayed Flights", f"{delayed_fl:,}", delta=f"{delay_rate_pct:.1f}%", delta_color="inverse")
        col3.metric("On-Time Flights", f"{ontime_fl:,}", delta=f"{100-delay_rate_pct:.1f}%")
        col4.metric("Delay Rate", f"{delay_rate_pct:.2f}%")
        col5.metric("Avg Departure Delay", f"{avg_dep_delay:.1f} mins")

        st.markdown("---")

        # Row 1 Charts: Punctuality Overview & Airline Delay Rate
        r1_c1, r1_c2 = st.columns([1, 1.2])

        with r1_c1:
            st.subheader("Flight Punctuality Overview")
            fig_pie = go.Figure(data=[go.Pie(
                labels=["On-Time (≤ 15 min)", "Delayed (> 15 min)"],
                values=[ontime_fl, delayed_fl],
                hole=0.45,
                marker_colors=["#10b981", "#ef4444"],
                textinfo="label+percent",
                hoverinfo="label+value+percent"
            )])
            fig_pie.update_layout(showlegend=True, margin=dict(t=30, b=0, l=0, r=0), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        with r1_c2:
            st.subheader("Delay Rate by Airline (%)")
            airline_stats = (
                filtered_df.groupby("AirlineName")
                .agg(Total=("IsDelayed", "count"), DelayRate=("IsDelayed", lambda x: x.mean() * 100))
                .reset_index()
                .sort_values(by="DelayRate", ascending=True)
            )
            fig_air = px.bar(
                airline_stats,
                x="DelayRate",
                y="AirlineName",
                orientation="h",
                color="DelayRate",
                color_continuous_scale="Reds",
                labels={"DelayRate": "Delay Rate (%)", "AirlineName": ""},
            )
            fig_air.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=320, coloraxis_showscale=False)
            st.plotly_chart(fig_air, use_container_width=True)

        st.markdown("---")

        # Row 2 Charts: Temporal Patterns
        r2_c1, r2_c2 = st.columns(2)

        with r2_c1:
            st.subheader("Hourly Delay Pattern (24-Hour Timeline)")
            hour_stats = (
                filtered_df.groupby("DepartureHour")
                .agg(DelayRate=("IsDelayed", lambda x: x.mean() * 100), Total=("IsDelayed", "count"))
                .reset_index()
            )
            fig_hr = px.line(
                hour_stats,
                x="DepartureHour",
                y="DelayRate",
                markers=True,
                labels={"DepartureHour": "Scheduled Departure Hour", "DelayRate": "Delay Rate (%)"}
            )
            fig_hr.update_traces(line_color="#3b82f6", line_width=3)
            fig_hr.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig_hr, use_container_width=True)

        with r2_c2:
            st.subheader("Delay Rate by Day of Week")
            dow_stats = (
                filtered_df.groupby("DayOfWeek")
                .agg(DelayRate=("IsDelayed", lambda x: x.mean() * 100))
                .reset_index()
            )
            dow_stats["DayName"] = dow_stats["DayOfWeek"].map(
                {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
            )
            fig_dow = px.bar(
                dow_stats,
                x="DayName",
                y="DelayRate",
                color="DelayRate",
                color_continuous_scale="Blues",
                labels={"DayName": "Day", "DelayRate": "Delay Rate (%)"}
            )
            fig_dow.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_dow, use_container_width=True)

        st.markdown("---")

        # Row 3 Charts: Airports & Routes
        r3_c1, r3_c2 = st.columns(2)

        with r3_c1:
            st.subheader("Top 10 Origin Airports with Highest Delays")
            origin_stats = (
                filtered_df.groupby("OriginAirportName")
                .agg(Total=("IsDelayed", "count"), DelayRate=("IsDelayed", lambda x: x.mean() * 100))
                .query("Total >= 50")
                .reset_index()
                .sort_values(by="DelayRate", ascending=False)
                .head(10)
                .sort_values(by="DelayRate", ascending=True)
            )
            fig_orig = px.bar(
                origin_stats,
                x="DelayRate",
                y="OriginAirportName",
                orientation="h",
                color="DelayRate",
                color_continuous_scale="Oranges",
                labels={"DelayRate": "Delay Rate (%)", "OriginAirportName": ""}
            )
            fig_orig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=340, coloraxis_showscale=False)
            st.plotly_chart(fig_orig, use_container_width=True)

        with r3_c2:
            st.subheader("Top 10 Most Delayed Flight Routes")
            route_stats = (
                filtered_df.groupby("Route")
                .agg(Total=("IsDelayed", "count"), DelayRate=("IsDelayed", lambda x: x.mean() * 100))
                .query("Total >= 30")
                .reset_index()
                .sort_values(by="DelayRate", ascending=False)
                .head(10)
                .sort_values(by="DelayRate", ascending=True)
            )
            fig_rt = px.bar(
                route_stats,
                x="DelayRate",
                y="Route",
                orientation="h",
                color="DelayRate",
                color_continuous_scale="Purples",
                labels={"DelayRate": "Delay Rate (%)", "Route": ""}
            )
            fig_rt.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=340, coloraxis_showscale=False)
            st.plotly_chart(fig_rt, use_container_width=True)

    # =========================================================================
    # TAB 2: INTERACTIVE FLIGHT DELAY PREDICTION
    # =========================================================================
    with tab_prediction:
        st.subheader("🎯 Real-Time Flight Delay Prediction (PySpark MLlib)")
        st.write("Input flight details to predict whether the flight is likely to be delayed (>15 minutes) using the trained Random Forest Classifier.")

        pred_col1, pred_col2 = st.columns(2)

        carrier_inv_map = {v: k for k, v in CARRIER_NAMES.items()}
        carrier_options = sorted(list(CARRIER_NAMES.values()))
        origin_options = sorted(list(df["OriginAirportName"].dropna().unique()))
        dest_options = sorted(list(df["DestAirportName"].dropna().unique()))

        with pred_col1:
            input_airline_name = st.selectbox("Operating Airline", carrier_options, index=0)
            input_carrier_code = carrier_inv_map.get(input_airline_name, "AA")

            input_origin = st.selectbox("Origin Airport", origin_options, index=origin_options.index("Hartsfield-Jackson Atlanta International") if "Hartsfield-Jackson Atlanta International" in origin_options else 0)
            input_dest = st.selectbox("Destination Airport", dest_options, index=dest_options.index("Los Angeles International") if "Los Angeles International" in dest_options else 1)

        with pred_col2:
            input_dep_hour = st.slider("Scheduled Departure Hour (0-23)", 0, 23, 17)
            
            day_map = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}
            input_day_name = st.selectbox("Day of Week", list(day_map.keys()), index=4)
            input_day_of_week = day_map[input_day_name]

            month_select_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                                "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
            input_month_name = st.selectbox("Month of Year", list(month_select_map.keys()), index=6)
            input_month = month_select_map[input_month_name]

            input_day_of_month = st.number_input("Day of Month (1-31)", min_value=1, max_value=31, value=15)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Predict Delay Risk", type="primary", use_container_width=True):
            predictor = get_predictor()
            if predictor is None:
                st.error("⚠️ Trained PySpark model could not be loaded. Please ensure training has been executed via `python src/train_model.py`.")
            else:
                with st.spinner("Evaluating flight characteristics with PySpark MLlib Pipeline..."):
                    try:
                        res = predictor.predict(
                            carrier=input_carrier_code,
                            origin_airport=input_origin,
                            dest_airport=input_dest,
                            departure_hour=input_dep_hour,
                            day_of_week=input_day_of_week,
                            month=input_month,
                            day_of_month=input_day_of_month
                        )

                        if res["prediction"] == 1.0:
                            st.markdown(f"""
                            <div class="prediction-box-delayed">
                                <h2 style="color: #ef4444; margin-bottom: 5px;">⚠️ Flight is Likely to be DELAYED</h2>
                                <p style="font-size: 1.1rem; color: #374151;">
                                    <b>Delay Probability:</b> {res['probability_delayed']:.1f}% &nbsp;|&nbsp;
                                    <b>Risk Tier:</b> <span style="color:#dc2626; font-weight:bold;">{res['risk_level']}</span>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box-ontime">
                                <h2 style="color: #16a34a; margin-bottom: 5px;">✅ Flight is Likely ON-TIME</h2>
                                <p style="font-size: 1.1rem; color: #374151;">
                                    <b>On-Time Probability:</b> {res['probability_ontime']:.1f}% &nbsp;|&nbsp;
                                    <b>Delay Probability:</b> {res['probability_delayed']:.1f}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.write("#### Prediction Confidence Breakdown:")
                        prob_cols = st.columns(2)
                        prob_cols[0].metric("On-Time Probability", f"{res['probability_ontime']:.1f}%")
                        prob_cols[1].metric("Delay Probability", f"{res['probability_delayed']:.1f}%")
                        st.progress(res['probability_delayed'] / 100.0)

                    except Exception as ex:
                        st.error(f"Prediction error: {ex}")

    # =========================================================================
    # TAB 3: MODEL PERFORMANCE & METRICS
    # =========================================================================
    with tab_model:
        st.subheader("🤖 PySpark MLlib Random Forest Model Evaluation")
        st.write("Evaluation results calculated on the 20% unseen test dataset.")

        metrics = load_metrics_report()
        if metrics:
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            m_c2.metric("Weighted Precision", f"{metrics['precision'] * 100:.2f}%")
            m_c3.metric("Weighted Recall", f"{metrics['recall'] * 100:.2f}%")
            m_c4.metric("Weighted F1-Score", f"{metrics['f1_score'] * 100:.2f}%")

            st.markdown("---")

            cm_col1, cm_col2 = st.columns(2)

            with cm_col1:
                st.subheader("Confusion Matrix")
                cm = metrics["confusion_matrix"]
                cm_matrix = [
                    [cm["true_negatives"], cm["false_positives"]],
                    [cm["false_negatives"], cm["true_positives"]]
                ]
                fig_cm = px.imshow(
                    cm_matrix,
                    labels=dict(x="Predicted Label", y="Actual Label", color="Flights"),
                    x=["On-Time (0)", "Delayed (1)"],
                    y=["On-Time (0)", "Delayed (1)"],
                    text_auto=True,
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=320)
                st.plotly_chart(fig_cm, use_container_width=True)

            with cm_col2:
                st.subheader("Model Specifications")
                st.markdown(f"""
                - **Algorithm:** `{metrics.get('model_name', 'Random Forest Classifier')}`
                - **Number of Decision Trees:** `{metrics.get('num_trees', 50)}`
                - **Max Tree Depth:** `{metrics.get('max_depth', 8)}`
                - **Training Set Size:** `{metrics.get('train_records', 0):,} flights (80%)`
                - **Test Set Size:** `{metrics.get('test_records', 0):,} flights (20%)`
                - **Area Under ROC (AUC):** `{metrics.get('roc_auc', 0.0):.4f}`
                - **Area Under PR:** `{metrics.get('pr_auc', 0.0):.4f}`
                - **Feature Pipeline:** `StringIndexer -> OneHotEncoder -> VectorAssembler`
                - **Data Leakage Prevention:** Strictly pre-departure features only.
                """)
        else:
            st.info("Model evaluation report not found. Run `python src/train_model.py` to generate authentic evaluation metrics.")


if __name__ == "__main__":
    main()
