"""
Feature Engineering Module
Constructs the PySpark MLlib Pipeline stages:
- StringIndexer: Encodes categorical string columns into numerical indices.
- OneHotEncoder: Maps categorical indices to binary vectors.
- VectorAssembler: Merges all feature columns into a single dense/sparse feature vector.
"""

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline


def build_feature_pipeline() -> tuple[list, list[str]]:
    """
    Builds the PySpark MLlib Feature Engineering Pipeline stages.
    
    Features used (strictly pre-departure to prevent data leakage):
    - Categorical: Carrier, OriginAirportName, DestAirportName
    - Numerical: DepartureHour, DayOfWeek, Month, DayofMonth
    
    Returns:
        tuple[list, list[str]]: List of pipeline stages and input feature names.
    """
    categorical_cols = ["Carrier", "OriginAirportName", "DestAirportName"]
    numeric_cols = ["DepartureHour", "DayOfWeek", "Month", "DayofMonth"]

    stages = []

    # 1. StringIndexer for categorical columns
    # handleInvalid="keep" ensures unseen categories in user testing get assigned a separate index instead of erroring
    index_output_cols = [f"{c}_idx" for c in categorical_cols]
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]
    stages.extend(indexers)

    # 2. OneHotEncoder for indexed categorical features
    ohe_output_cols = [f"{c}_vec" for c in categorical_cols]
    encoder = OneHotEncoder(
        inputCols=index_output_cols,
        outputCols=ohe_output_cols,
        handleInvalid="keep"
    )
    stages.append(encoder)

    # 3. VectorAssembler to combine all feature vectors
    assembler_inputs = ohe_output_cols + numeric_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="keep"
    )
    stages.append(assembler)

    all_features = categorical_cols + numeric_cols
    return stages, all_features
