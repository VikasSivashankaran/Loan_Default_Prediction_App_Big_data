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
import sys
from io import StringIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ---------------------------
# Streamlit App Configuration
# ---------------------------

st.set_page_config(
    page_title="Loan Default Prediction with 3D Plot",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Loan Default Prediction Application with 3D Plot")

# Sidebar for user inputs
st.sidebar.header("User Input for Prediction")

# ---------------------------
# Set Java Path for PySpark
# ---------------------------

def setup_java_environment():
    """
    Sets up the Java environment by defining the JAVA_HOME variable dynamically.
    """
    java_home_path = r"C:\Program Files\Java\jdk1.8.0_202"  # Update with your JDK path
    os.environ['JAVA_HOME'] = java_home_path
    os.environ['PATH'] = java_home_path + r"\bin;" + os.environ['PATH']

setup_java_environment()

# ---------------------------
# Spark Configuration and Initialization
# ---------------------------

def initialize_spark():
    """
    Initializes and returns SparkContext and SQLContext.
    Sets environment variables dynamically based on the current Python executable.
    """
    # Dynamically get the current Python executable path
    python_exec = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_exec
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_exec

    # Create Spark configuration and context
    conf = SparkConf() \
        .setAppName('Loan_Default_Prediction') \
        .setMaster("local[*]") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g") \
        .set("spark.network.timeout", "800s") \
        .set("spark.executor.cores", "2")
    
    # Initialize SparkContext
    sc = SparkContext.getOrCreate(conf=conf)
    
    # Create SQLContext from SparkContext
    sql_context = SQLContext(sc)
    return sc, sql_context

# Initialize Spark
sc, sql_context = initialize_spark()

# ---------------------------
# Data Loading and Display
# ---------------------------

def load_data_pandas(filepath):
    """
    Loads data into a Pandas DataFrame and imputes missing values.
    """
    df = pd.read_csv(filepath)
    # Columns to impute
    columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
    # Impute missing values with column mean
    df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())
    return df

def load_data_spark(filepath):
    """
    Loads data into a Spark DataFrame.
    """
    df_spark = sql_context.read.csv(filepath, header=True, inferSchema=True)
    return df_spark

# File uploader for Loan_Default.csv
uploaded_file = st.sidebar.file_uploader("Upload Loan_Default.csv", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with open("Loan_Default.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    data_path = "Loan_Default.csv"
else:
    st.warning("Please upload the `Loan_Default.csv` file to proceed.")
    st.stop()

# Load data using Spark
df_spark = load_data_spark(data_path)

st.header("Dataset Schema")
with st.expander("View Schema"):
    # Capture the schema as a string
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    df_spark.printSchema()
    sys.stdout = old_stdout
    schema_str = mystdout.getvalue()
    st.text(schema_str)

st.header("Sample Data")
st.dataframe(df_spark.limit(5).toPandas())

# ---------------------------
# Data Preprocessing with PySpark
# ---------------------------

st.header("Data Preprocessing")

columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
output_columns = [f"{col}_imputed" for col in columns_to_impute]

# Handle missing values using Imputer
imputer = Imputer(inputCols=columns_to_impute, outputCols=output_columns, strategy="mean")

# Assemble features into a single vector
assembler = VectorAssembler(
    inputCols=['loan_amount', 'rate_of_interest_imputed', 'property_value_imputed', 'income_imputed', 'Credit_Score', 'LTV_imputed'],
    outputCol='features'
)

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
# 3D Plot Section
# ---------------------------

st.header("3D Plot of Loan Data")

def plot_3d(df):
    """
    Plots a 3D scatter plot for loan_amount, income, and property_value.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot data
    x = df['loan_amount'].toPandas()
    y = df['income'].toPandas()
    z = df['property_value'].toPandas()

    # Color by credit score (assuming higher credit score means safer loans)
    color_map = df['Credit_Score'].toPandas()
    
    scatter = ax.scatter(x, y, z, c=color_map, cmap='coolwarm')

    ax.set_xlabel('Loan Amount')
    ax.set_ylabel('Income')
    ax.set_zlabel('Property Value')

    plt.colorbar(scatter, label='Credit Score')

    st.pyplot(fig)

# Call the 3D plot function
plot_3d(df_transformed)

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

def train_pytorch_model(filepath):
    """
    Trains a simple PyTorch neural network model on the dataset.
    """
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

    # Initialize the model
    model = SimpleModel(X.shape[1])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

    # Save the model after training
    torch.save(model.state_dict(), "pytorch_model.pth")
    return model

# Train the PyTorch model
if st.button("Train PyTorch Model"):
    pytorch_model = train_pytorch_model(data_path)
    st.success("PyTorch model trained and saved successfully.")

# ---------------------------
# End of Application
# ---------------------------
