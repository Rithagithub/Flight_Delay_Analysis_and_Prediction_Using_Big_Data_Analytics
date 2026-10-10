"""
Spark Utilities Module
Handles Java environment setup, Windows winutils configuration,
Python executable alignment, and SparkSession creation for seamless distributed execution.
"""

import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession

# Known carrier code to airline full name mapping
CARRIER_NAMES = {
    '9E': 'Endeavor Air',
    'AA': 'American Airlines',
    'AS': 'Alaska Airlines',
    'B6': 'JetBlue Airways',
    'DL': 'Delta Air Lines',
    'EV': 'ExpressJet Airlines',
    'F9': 'Frontier Airlines',
    'FL': 'AirTran Airways',
    'HA': 'Hawaiian Airlines',
    'MQ': 'Envoy Air (American Eagle)',
    'OO': 'SkyWest Airlines',
    'UA': 'United Airlines',
    'US': 'US Airways',
    'VX': 'Virgin America',
    'WN': 'Southwest Airlines',
    'YV': 'Mesa Airlines'
}


def setup_environment():
    """
    Configures JAVA_HOME, HADOOP_HOME, PYSPARK_PYTHON, and PYSPARK_DRIVER_PYTHON.
    """
    # Align Python executable across driver and worker processes
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # 1. Setup JAVA_HOME
    if "JAVA_HOME" not in os.environ or not os.path.exists(os.environ["JAVA_HOME"]):
        java_root = Path(r"C:\Program Files\Java")
        if java_root.exists():
            jdks = sorted(list(java_root.glob("jdk*")), reverse=True)
            if jdks:
                os.environ["JAVA_HOME"] = str(jdks[0])
        else:
            possible_jdk_paths = [
                r"C:\Program Files\Java\jdk-17",
                r"C:\Program Files\Java\jdk-21",
                r"C:\Program Files\Java\jdk-11",
            ]
            for p in possible_jdk_paths:
                if os.path.exists(p):
                    os.environ["JAVA_HOME"] = p
                    break

    if "JAVA_HOME" in os.environ and os.path.exists(os.environ["JAVA_HOME"]):
        java_bin = os.path.join(os.environ["JAVA_HOME"], "bin")
        if java_bin not in os.environ["PATH"]:
            os.environ["PATH"] = java_bin + ";" + os.environ["PATH"]

    # 2. Setup HADOOP_HOME (for Windows winutils.exe)
    project_root = Path(__file__).resolve().parent.parent
    local_hadoop = project_root / "hadoop"
    if local_hadoop.exists():
        os.environ["HADOOP_HOME"] = str(local_hadoop)
        hadoop_bin = str(local_hadoop / "bin")
        if hadoop_bin not in os.environ["PATH"]:
            os.environ["PATH"] = hadoop_bin + ";" + os.environ["PATH"]


def get_spark_session(app_name: str = "FlightDelayAnalysisBDA", memory: str = "4g") -> SparkSession:
    """
    Creates and returns a local SparkSession configured for high-performance Big Data analytics.
    
    Args:
        app_name (str): Application name for Spark logs.
        memory (str): Driver memory allocation (e.g. '4g').
        
    Returns:
        SparkSession: Configured active Spark session.
    """
    setup_environment()
    
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.hadoop.io.native.lib.available", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
