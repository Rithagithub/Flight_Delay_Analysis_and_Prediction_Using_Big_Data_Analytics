# Flight Delay Analysis and Prediction Using Big Data Analytics

An academic Big Data Analytics (BDA) Capstone Project implementing scalable distributed data processing, exploratory data analytics, machine learning classification with **PySpark MLlib**, and an interactive web dashboard with **Streamlit** and **Plotly**.

---

## 1. Problem Statement

Commercial aviation flight delays cause billions of dollars in economic loss annually, disrupt passenger schedules, strain airport logistics, and increase airline operational costs. Understanding historical delay patterns and building predictive models can enable proactive decision-making for airlines and travelers. 

Handling hundreds of thousands of flight records requires scalable distributed data processing frameworks like **Apache Spark (PySpark)** rather than traditional single-threaded libraries like Pandas.

---

## 2. Project Objectives

1. **Analyze Large-Scale Flight Records**: Identify key patterns, congestion hotspots, seasonality, and factors influencing flight departure delays.
2. **Apply Distributed Big Data Processing**: Utilize **PySpark DataFrame APIs** (`select`, `filter`, `withColumn`, `groupBy`, `agg`, `orderBy`) for scalable data cleaning, transformation, and feature engineering.
3. **Develop a Machine Learning Pipeline**: Train a **PySpark MLlib RandomForestClassifier** with balanced weighting to predict departure delays (`IsDelayed`) using strictly pre-departure features.
4. **Deploy an Interactive Dashboard**: Deliver a web interface with **Streamlit** for real-time KPI monitoring, interactive filters, Plotly data visualizations, and live flight delay prediction.

---

## 3. System Architecture

```text
       +-------------------------------------------------------------+
       |             Raw Flight Dataset (flights.csv)                |
       |                   (271,940 Records)                         |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |             PySpark Data Loader (data_loader.py)            |
       |        - SparkSession initialization (local[*])             |
       |        - Schema inference & distributed CSV loading         |
       +-------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |           Data Preprocessing (data_preprocessing.py)        |
       |        - Cancellation filtering (Cancelled == 0)            |
       |        - Null handling in critical departure columns        |
       |        - Feature extraction (DepartureHour, Route, Carrier) |
       |        - Target definition: IsDelayed (DepDelay > 15 mins)  |
       |        - Processed output: flights_clean.parquet            |
       +-------------------------------------------------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
       +----------------------------+   +----------------------------+
       |  Big Data Analytics        |   | PySpark MLlib ML Pipeline  |
       |       (analysis.py)        |   |     (train_model.py)       |
       | - Airline delay rankings   |   | - 80/20 Train-Test Split   |
       | - Airport & route metrics  |   | - StringIndexer            |
       | - Hourly & daily trends    |   | - OneHotEncoder            |
       | - Interactive Plotly charts|   | - VectorAssembler          |
       +----------------------------+   | - RandomForestClassifier   |
                     |                  +----------------------------+
                     |                                 |
                     |                                 v
                     |                  +----------------------------+
                     |                  |   Trained PipelineModel    |
                     |                  | (models/flight_delay_model)|
                     |                  +----------------------------+
                     |                                 |
                     +----------------+----------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |      Streamlit Web Application (dashboard/app.py)           |
       | - Real-time KPI Cards & Sidebar Filters                     |
       | - Interactive Plotly Analytics Visualizations               |
       | - Live Flight Delay Prediction Form & Confidence Scores     |
       | - ML Performance Metrics & Confusion Matrix                 |
       +-------------------------------------------------------------+
```

---

## 4. Technology Stack

| Component | Technology | Role & Justification |
|---|---|---|
| **Language** | Python 3.10+ | Core programming language |
| **Big Data Engine** | Apache Spark 3.5+ / PySpark | Distributed in-memory data processing across CPU cores |
| **Machine Learning** | PySpark MLlib | Distributed feature transformations (`StringIndexer`, `OneHotEncoder`, `VectorAssembler`) and `RandomForestClassifier` |
| **Data Analysis** | Pandas & NumPy | Small aggregated summary transformations |
| **Data Visualization** | Plotly Express & Graph Objects | Interactive charts (pie, bar, line charts) |
| **Web Dashboard** | Streamlit | Web user interface and real-time prediction deployment |
| **Storage Formats** | Apache Parquet & CSV | Columnar storage for Spark and CSV for raw data |

---

## 5. Dataset Summary (Real Dataset)

The project uses the real US Bureau of Transportation Statistics (BTS) flight dataset:

- **Filename**: `data/raw/flights.csv`
- **Total Raw Records**: `271,940`
- **Total Cleaned Records**: `269,024` (Non-cancelled, valid departure records)
- **Cancelled Flights**: `2,916`
- **Total Columns**: `20`

### Key Columns:
1. `Year`, `Month`, `DayofMonth`, `DayOfWeek`: Flight calendar date.
2. `Carrier`: 2-letter airline code (e.g. `AA`, `DL`, `WN`, `UA`, `EV`, `MQ`).
3. `OriginAirportID`, `OriginAirportName`, `OriginCity`, `OriginState`: Origin details.
4. `DestAirportID`, `DestAirportName`, `DestCity`, `DestState`: Destination details.
5. `CRSDepTime`: Scheduled departure time in `HHMM` integer format.
6. `DepDelay`: Actual departure delay in minutes (negative = early, positive = delayed).
7. `DepDel15`: FAA indicator (1 if `DepDelay >= 15`, else 0).
8. `CRSArrTime`: Scheduled arrival time in `HHMM` integer format.
9. `ArrDelay`: Arrival delay in minutes.
10. `Cancelled`: Binary indicator (0 = operated, 1 = cancelled).

---

## 6. Target Variable & Data Leakage Prevention

### Target Variable: `IsDelayed`
According to the Federal Aviation Administration (FAA) and US Department of Transportation (DOT), a flight is officially delayed if its departure is delayed by **more than 15 minutes**:

$$\text{IsDelayed} = \begin{cases} 1.0 & \text{if } \text{DepDelay} > 15 \\ 0.0 & \text{if } \text{DepDelay} \le 15 \end{cases}$$

### Strict Data Leakage Prevention:
To ensure the machine learning model is realistic and deployable in production, **only features known before flight departure** are used for training:
- ✅ **Included Features**: `Carrier`, `OriginAirportName`, `DestAirportName`, `DepartureHour`, `DayOfWeek`, `Month`, `DayofMonth`.
- ❌ **Excluded Features (Data Leakage)**: `ArrDelay`, `ArrDel15`, `ActualArrivalTime`, `DepDelay` (target-derived post-flight information).

---

## 7. Machine Learning Pipeline (PySpark MLlib)

The model is built using a unified PySpark MLlib `Pipeline`:

```text
[Input Features]
      ↓
[StringIndexer]      -> Converts Carrier, OriginAirport, DestAirport strings to categorical indices
      ↓
[OneHotEncoder]      -> Converts categorical indices to sparse binary feature vectors
      ↓
[VectorAssembler]    -> Combines encoded categorical vectors + numeric features (DepartureHour, Month, etc.)
      ↓
[RandomForestClassifier] -> Ensemble of 50 decision trees with balanced class weights (maxDepth=8, seed=42)
      ↓
[PipelineModel]      -> End-to-end trained model persisted to models/flight_delay_model/
```

### Real Model Evaluation Results (Test Set: 53,631 Flights):

| Metric | Real Score |
|---|---|
| **Test Accuracy** | **60.06%** |
| **Weighted Precision** | **77.48%** |
| **Weighted Recall** | **60.06%** |
| **Weighted F1-Score** | **64.19%** |
| **Area Under ROC (AUC)** | **0.6994** |
| **Area Under PR (AUPR)** | **0.3497** |

### Confusion Matrix Breakdown:
- **True Positives (TP)**: `7,490` (Correctly predicted Delayed)
- **True Negatives (TN)**: `24,723` (Correctly predicted On-Time)
- **False Positives (FP)**: `18,443` (On-Time misclassified as Delayed)
- **False Negatives (FN)**: `2,975` (Delayed misclassified as On-Time)

---

## 8. Big Data Analytics Key Findings

Calculated directly from the real dataset using PySpark distributed aggregations:

1. **Overall Punctuality**:
   - Total Valid Flights: `269,024`
   - Delayed Flights (>15 min): `51,886` (**19.29%**)
   - On-Time Flights (≤15 min): `217,138` (**80.71%**)
   - Average Departure Delay: **10.43 minutes**

2. **Top Airlines with Highest Delay Rates**:
   - `Envoy Air (MQ)`: **25.28%** delay rate (Avg delay: 15.16 min)
   - `Southwest Airlines (WN)`: **24.65%** delay rate (Avg delay: 12.68 min)
   - `ExpressJet Airlines (EV)`: **23.17%** delay rate (Avg delay: 14.38 min)
   - `JetBlue Airways (B6)`: **22.22%** delay rate (Avg delay: 12.73 min)
   - `American Airlines (AA)`: **20.68%** delay rate (Avg delay: 12.25 min)

3. **Top Origin Airports with Highest Delay Rates**:
   - `Chicago Midway International (MDW)`: **30.99%**
   - `William P Hobby (HOU)`: **26.30%**
   - `Chicago O'Hare International (ORD)`: **25.92%**
   - `Denver International (DEN)`: **25.25%**
   - `Dallas/Fort Worth International (DFW)`: **24.59%**

4. **Hourly Delay Trends**:
   - Early morning flights (5:00 AM - 7:00 AM) exhibit lowest delay rates (~6.0%).
   - Delays accumulate throughout the day, peaking in late evening (7:00 PM - 9:00 PM) above 25-28%.

---

## 9. Folder Structure

```text
Flight_Delay_Analysis_and_Prediction_Using_Big_Data_Analytics/
├── data/
│   ├── raw/
│   │   └── flights.csv                  # Real raw flight dataset (271,940 rows)
│   └── processed/
│       ├── flights_clean.parquet/       # Cleaned PySpark Parquet dataset
│       └── flights_sample.csv           # Lightweight CSV sample for UI
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── spark_utils.py                   # Environment setup & SparkSession builder
│   ├── data_loader.py                   # PySpark CSV dataset loader
│   ├── data_preprocessing.py            # Data cleaning & feature engineering
│   ├── feature_engineering.py           # MLlib pipeline stages (StringIndexer, OHE, Assembler)
│   ├── analysis.py                      # Big Data analytics & Plotly chart generation
│   ├── train_model.py                   # Random Forest model training & evaluation
│   └── prediction.py                    # Reusable real-time prediction module
├── models/
│   └── flight_delay_model/              # Persisted PySpark MLlib PipelineModel
├── dashboard/
│   └── app.py                           # Streamlit interactive web application
├── notebooks/
│   └── Flight_Delay_Analysis_and_Prediction.ipynb  # Jupyter Notebook for demonstration
├── outputs/
│   ├── charts/                          # Generated interactive Plotly HTML charts
│   └── reports/
│       ├── model_evaluation_report.json # Model metrics in JSON format
│       └── model_evaluation_report.txt  # Human-readable evaluation report
├── hadoop/
│   └── bin/
│       ├── winutils.exe                 # Windows Hadoop native utility
│       └── hadoop.dll                   # Windows Hadoop native DLL
├── requirements.txt                     # Project dependencies
├── README.md                            # Comprehensive project documentation
└── run.py                               # Master CLI execution script
```

---

## 10. Installation & Setup Instructions

### Prerequisites
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: Python 3.10, 3.11, or 3.12
- **Java**: Java JDK 11, 17, or 21 (JDK 17 recommended)

### Step 1: Clone or Navigate to the Project Folder
```bash
cd Flight_Delay_Analysis_and_Prediction_Using_Big_Data_Analytics
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 11. How to Run the Project

### Option A: Run the Complete End-to-End Pipeline (Recommended)
```bash
python run.py --all
```
This single command automatically:
1. Loads raw CSV data using PySpark.
2. Cleans data and saves `data/processed/flights_clean.parquet`.
3. Computes Big Data analytics and generates Plotly charts in `outputs/charts/`.
4. Trains the PySpark MLlib Random Forest model (80% train, 20% test).
5. Persists the trained model to `models/flight_delay_model/` and writes `outputs/reports/`.
6. Executes a sample real-time prediction.

### Option B: Run Individual Pipeline Steps
```bash
# 1. Run Data Preprocessing
python src/data_preprocessing.py

# 2. Run Big Data Analytics
python src/analysis.py

# 3. Train MLlib Random Forest Model
python src/train_model.py

# 4. Test Single Flight Prediction
python src/prediction.py
```

### Step 3: Launch the Streamlit Web Dashboard
```bash
streamlit run dashboard/app.py
```
Open your web browser at `http://localhost:8501`.

---

## 12. Viva Questions & Beginner-Friendly Answers

### Q1: What is Big Data, and why is this a Big Data project?
> **Answer**: Big Data refers to data volumes that exceed the processing capacity of traditional single-node memory and CPU architectures. This project processes **over 270,000 real flight records** using **Apache Spark (PySpark)**, which distributes transformations across CPU cores in memory using partitions, rather than relying on standard single-threaded Pandas.

### Q2: Why use PySpark instead of Pandas?
> **Answer**: Pandas runs entirely in single-threaded memory on a single CPU core. If a dataset exceeds RAM, Pandas crashes with `MemoryError`. PySpark uses a distributed computing model with lazy evaluation, partition-level parallel execution, and the Catalyst Optimizer, allowing it to scale from a single laptop to hundreds of cluster nodes seamlessly.

### Q3: What is a Spark DataFrame?
> **Answer**: A Spark DataFrame is a distributed collection of data organized into named columns. It is conceptually equivalent to a table in a relational database but with rich optimization under the hood (Catalyst Optimizer and Tungsten execution engine).

### Q4: What is Lazy Evaluation in Spark?
> **Answer**: Spark does not execute transformations (like `select`, `filter`, `withColumn`) immediately when called. Instead, it builds a Directed Acyclic Graph (DAG) of transformations. Execution is only triggered when an **Action** (like `count()`, `collect()`, `show()`, or `write()`) is called. This allows Spark to optimize the entire execution plan.

### Q5: Why is 15 minutes used as the delay threshold?
> **Answer**: The Federal Aviation Administration (FAA) and the US Bureau of Transportation Statistics (BTS) define an on-time flight as one that departs or arrives within 15 minutes of the scheduled time (`CRSDepTime`). Any departure exceeding 15 minutes is classified as delayed (`IsDelayed = 1`).

### Q6: What is Data Leakage, and how did you prevent it?
> **Answer**: Data leakage occurs when information from outside the training dataset (specifically information that would only be available *after* the event occurs) is used to train the model. We strictly prevented data leakage by excluding arrival delay, actual flight duration, and arrival times from our features, using only attributes known *before* departure (Airline, Origin, Destination, Departure Hour, Day of Week, Month).

### Q7: What is Random Forest, and why choose it?
> **Answer**: Random Forest is an ensemble learning algorithm that builds multiple decision trees during training and merges their predictions (majority vote for classification). It handles non-linear relationships, mitigates overfitting compared to individual decision trees, and works well with categorical features.

### Q8: What are `StringIndexer`, `OneHotEncoder`, and `VectorAssembler`?
> **Answer**:
> - **`StringIndexer`**: Assigns a unique numerical index (0, 1, 2...) to each unique category in a string column.
> - **`OneHotEncoder`**: Converts numerical indices into binary sparse vectors so algorithms do not assume false ordinal relationships.
> - **`VectorAssembler`**: A PySpark feature transformer that merges multiple feature columns into a single vector column (`features`), which is the required format for PySpark MLlib models.

### Q9: What is a PySpark ML Pipeline?
> **Answer**: A Pipeline chains multiple feature extraction, transformation, and estimator stages into a single automated workflow. It ensures that identical transformations are applied consistently during training, testing, and real-time prediction.

### Q10: What is the Confusion Matrix?
> **Answer**: A table used to describe the performance of a classification model on test data:
> - **TP (True Positive)**: Actually Delayed, predicted as Delayed.
> - **TN (True Negative)**: Actually On-Time, predicted as On-Time.
> - **FP (False Positive / Type I Error)**: Actually On-Time, predicted as Delayed.
> - **FN (False Negative / Type II Error)**: Actually Delayed, predicted as On-Time.

### Q11: What is Accuracy, Precision, Recall, and F1-Score?
> **Answer**:
> - **Accuracy**: Overall proportion of correct predictions: $\frac{TP + TN}{TP + TN + FP + FN}$.
> - **Precision**: Out of all flights predicted as delayed, how many were actually delayed: $\frac{TP}{TP + FP}$.
> - **Recall**: Out of all flights that were actually delayed, how many did the model catch: $\frac{TP}{TP + FN}$.
> - **F1-Score**: Harmonic mean of Precision and Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

### Q12: Why is the dataset imbalanced, and how was it handled?
> **Answer**: In real-world aviation, approximately **80.7%** of flights depart on time and **19.3%** are delayed. An unweighted model might simply predict "On-Time" for everything to achieve 80% accuracy without learning delay patterns. We applied **balanced class weighting** in PySpark MLlib to penalize delay misclassifications appropriately.

---

## 13. Limitations & Future Enhancements

### Limitations
- **Weather Data**: Extreme weather events (thunderstorms, blizzards) are primary external delay causes but are not included in standard historical flight schedules.
- **Inbound Aircraft Turnaround**: Tail number tracking for incoming aircraft delays was not modeled in this standalone dataset.

### Future Enhancements
- **Live Weather API Integration**: Ingest real-time Doppler radar and meteorological forecasts at origin and destination airports.
- **Spark Streaming**: Implement Apache Spark Structured Streaming to process live flight radar telemetry streams in real time.
- **Model Tuning**: Perform cross-validation and explore gradient-boosted decision trees (`GBTClassifier`).

---

## 14. Academic Certification
This project was developed for the **Big Data Analytics (BDA)** Course-Based Project (CBP).
- **Core Frameworks**: Apache Spark, PySpark MLlib, Streamlit, Plotly, Python 3.12
- **Target Task**: Flight Delay Punctuality Analysis & Predictive Machine Learning
