import streamlit as st
import pandas as pd
import torch
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import Imputer
import atexit

# Initialize Spark
spark = SparkSession.builder \
    .appName("Loan Default Prediction") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Load data
DATA_PATH = "path/to/your/loan_data.csv"

@st.cache
def load_data_spark():
    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    return df

df_spark = load_data_spark()

# Data Preprocessing
def preprocess_data(df):
    # Handle missing values
    imputer = Imputer(inputCols=df.columns, outputCols=[f"{c}_imputed" for c in df.columns])
    df = imputer.fit(df).transform(df)

    # Feature Engineering
    feature_columns = [f"{c}_imputed" for c in df.columns if c != "Loan_Status"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    # Scaling features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    df = scaler.fit(df).transform(df)

    return df.select("scaled_features", "Loan_Status")

df_processed = preprocess_data(df_spark)

# Model Training
def train_model(df):
    pipeline = Pipeline(stages=[
        LogisticRegression(featuresCol="scaled_features", labelCol="Loan_Status")
    ])
    model = pipeline.fit(df)
    return model

model = train_model(df_processed)

# Load PyTorch Model
# Assume your PyTorch model is defined elsewhere and loaded here
pytorch_model = torch.load("path/to/your/pytorch_model.pth")

# Streamlit UI
st.title("Loan Default Prediction App")

def user_input_features():
    # User inputs
    loan_amount = st.sidebar.number_input("Loan Amount", value=10000, min_value=0)
    rate_of_interest = st.sidebar.number_input("Rate of Interest", value=5.0, min_value=0.0)
    property_value = st.sidebar.number_input("Property Value", value=150000, min_value=0)
    income = st.sidebar.number_input("Annual Income", value=50000, min_value=0)
    credit_score = st.sidebar.number_input("Credit Score", value=700, min_value=300, max_value=850)
    ltv = st.sidebar.number_input("Loan to Value Ratio", value=70.0, min_value=0.0, max_value=100.0)

    data = {
        'loan_amount': loan_amount,
        'rate_of_interest': rate_of_interest,
        'property_value': property_value,
        'income': income,
        'Credit_Score': credit_score,
        'LTV': ltv
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_features = user_input_features()

# Make predictions using the PyTorch model
input_tensor = torch.tensor(input_features.values, dtype=torch.float32)
with torch.no_grad():
    prediction = pytorch_model(input_tensor)
    predicted_class = prediction.argmax(dim=1).item()  # Get the index of the max log-probability

# Display prediction results
st.subheader("Prediction Results")
if predicted_class == 1:
    st.write("The loan is likely to be defaulted.")
else:
    st.write("The loan is likely to be repaid.")

# Clean up Spark context at exit
atexit.register(lambda: spark.stop())
