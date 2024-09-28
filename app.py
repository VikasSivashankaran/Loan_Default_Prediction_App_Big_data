import streamlit as st
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import os
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px
import atexit
from io import StringIO
import sys

# ---------------------------
# Streamlit App Configuration
# ---------------------------
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Loan Default Prediction Application")

# Sidebar for user inputs
st.sidebar.header("User Input for Prediction")

# ---------------------------
# Spark Configuration and Initialization
# ---------------------------
@st.cache_resource
def initialize_spark():
    # Set the path to your Python executable
    os.environ['PYSPARK_PYTHON'] = r'C:\Users\GowthamMaheswar\AppData\Local\Programs\Python\Python312\python.exe'
    os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\GowthamMaheswar\AppData\Local\Programs\Python\Python312\python.exe'
    
    # Create Spark configuration and context
    conf = SparkConf() \
        .setAppName('Loan_Default_Prediction') \
        .setMaster("local[*]") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g") \
        .set("spark.network.timeout", "800s") \
        .set("spark.executor.cores", "2")
    
    sc = SparkContext.getOrCreate(conf=conf)
    
    # Create SQLContext from SparkContext
    sql_context = SQLContext(sc)
    return sc, sql_context

# Initialize Spark
sc, sql_context = initialize_spark()

# ---------------------------
# Data Loading and Display
# ---------------------------
st.header("Dataset Schema")

# Define the path to your CSV file
DATA_PATH = 'Loan_Default.csv'  # Update this path if your CSV is located elsewhere

# Check if the file exists
if not os.path.exists(DATA_PATH):
    st.error(f"The file `{DATA_PATH}` was not found. Please ensure it exists in the application directory.")
    st.stop()

# Load data using Spark
@st.cache_data
def load_data_spark(filepath):
    df_spark = sql_context.read.csv(filepath, header=True, inferSchema=True)
    return df_spark.toPandas()  # Collect the DataFrame to Pandas

df_pandas = load_data_spark(DATA_PATH)

# Display the schema
with st.expander("View Schema"):
    st.text(df_pandas.dtypes)

# Display sample data
st.header("Sample Data")
st.dataframe(df_pandas.head())

# ---------------------------
# Data Preprocessing with PySpark
# ---------------------------
st.header("Data Preprocessing")

# Convert to Spark DataFrame for preprocessing
df_spark = sql_context.createDataFrame(df_pandas)

columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
output_columns = columns_to_impute

# Handle missing values using Imputer
imputer = Imputer(inputCols=columns_to_impute, outputCols=output_columns)

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV'],
                            outputCol='features')

# Standardize the features
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

# Create a pipeline for preprocessing
pipeline = Pipeline(stages=[imputer, assembler, scaler])

# Fit the pipeline to the data
with st.spinner("Preprocessing data..."):
    model = pipeline.fit(df_spark)
    df_transformed = model.transform(df_spark)
st.success("Data preprocessing completed successfully.")

# ---------------------------
# Train-Test Split
# ---------------------------
train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=42)

st.write(f"**Training Data Count:** {train_data.count()}")
st.write(f"**Test Data Count:** {test_data.count()}")

# ---------------------------
# Logistic Regression Model
# ---------------------------
st.header("Logistic Regression Model")

# Logistic Regression model
lr = LogisticRegression(featuresCol='scaled_features', labelCol='Status')

# Fit the model to training data
with st.spinner("Training Logistic Regression model..."):
    lr_model = lr.fit(train_data)
st.success("Logistic Regression model trained successfully.")

# Make predictions on test data
predictions = lr_model.transform(test_data)

# Evaluate the model using ROC-AUC
evaluator = BinaryClassificationEvaluator(labelCol='Status', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
roc_auc = evaluator.evaluate(predictions)

# Calculate prediction accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Status", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)

st.subheader("Model Evaluation Metrics")
col1, col2 = st.columns(2)
col1.metric("ROC-AUC", f"{roc_auc:.4f}")
col2.metric("Accuracy", f"{accuracy:.4f}")

# ---------------------------
# PyTorch Model Definition and Training
# ---------------------------
st.header("PyTorch Neural Network Model")

class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Binary classification

    def forward(self, x):
        return self.fc(x)

@st.cache_resource
def train_pytorch_model(filepath):
    # Load the dataset into pandas for PyTorch training
    df = pd.read_csv(filepath)
    
    # Preprocess the data: Handle missing values
    columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
    df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())
    
    # Prepare features and labels
    X = df[['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV']].values
    y = df['Status'].values  # Assuming 'Status' is the column to predict
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create a DataLoader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create and train the model
    input_dim = X.shape[1]  # Number of features
    pytorch_model = SimpleModel(input_dim)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
    
    # Train the model
    for epoch in range(10):  # Number of epochs
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = pytorch_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        st.write(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')
    
    # Save the trained PyTorch model
    torch.save(pytorch_model.state_dict(), 'loan_prediction_model.pth')  # Save the model state_dict
    return pytorch_model

# Train PyTorch model
with st.spinner("Training PyTorch model..."):
    pytorch_model = train_pytorch_model(DATA_PATH)
    st.snow()
st.success("PyTorch model trained and saved successfully.")

# Load the pre-trained PyTorch model
def load_pytorch_model(filepath, input_dim):
    model = SimpleModel(input_dim)
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    return model

pytorch_model = load_pytorch_model('loan_prediction_model.pth', input_dim=6)

# ---------------------------
# User Input for Prediction
# ---------------------------
st.header("Loan Eligibility Prediction")

# Collect user input for prediction using Streamlit widgets
def user_input_features():
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, value=10000.0)
    rate_of_interest = st.sidebar.number_input("Rate of Interest", min_value=0.0, value=5.0)
    property_value = st.sidebar.number_input("Property Value", min_value=0.0, value=200000.0)
    income = st.sidebar.number_input("Income", min_value=0.0, value=50000.0)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=700)
    ltv = st.sidebar.number_input("Loan-to-Value (LTV)", min_value=0.0, max_value=100.0, value=80.0)
    
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
atexit.register(lambda: sc.stop())
