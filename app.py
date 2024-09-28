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

@st.cache_data
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

st.balloons()

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
    pytorch_model = train_pytorch_model(data_path)
    st.snow()
st.success("PyTorch model trained and saved successfully.")

# Load the pre-trained PyTorch model
def load_pytorch_model(filepath, input_dim):
    """
    Loads the trained PyTorch model from the specified filepath.
    """
    model = SimpleModel(input_dim)
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
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
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Convert user input into tensor
user_input_tensor = torch.tensor(input_df.values, dtype=torch.float32)

# Make prediction using PyTorch model
with torch.no_grad():
    output = pytorch_model(user_input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()  # Get the predicted class

# Interpret the prediction
if predicted_class == 1:
    prediction_text = "The loan is likely to be **sanctioned**."
else:
    prediction_text = "The loan is likely to be **rejected**."

st.subheader("Prediction")
st.write(prediction_text)

# ---------------------------
# 3D Visualizations
# ---------------------------

st.header("3D Visualizations")

# Load data using pandas for visualization
df_pandas = load_data_pandas(data_path)

# Sample 100 rows
if len(df_pandas) >= 100:
    sampled_df = df_pandas.sample(n=100, random_state=42)
else:
    sampled_df = df_pandas.copy()

# Drop rows with NaN in 'loan_amount', 'rate_of_interest', or 'age'
sampled_df = sampled_df.dropna(subset=['loan_amount', 'rate_of_interest', 'age'])

# Ensure that 'rate_of_interest' has no negative or zero values if required
# For example, replace negative values with a small positive value to avoid size issues
sampled_df['rate_of_interest'] = sampled_df['rate_of_interest'].apply(lambda x: x if x > 0 else 0.1)

# 3D Line Plot
st.subheader("3D Line Plot")
fig_line = px.line_3d(
    sampled_df,
    x="loan_amount",
    y="rate_of_interest",
    z="age",
    title="3D Line Plot of Loan Amount, Rate of Interest, and Age"
)
st.plotly_chart(fig_line, use_container_width=True)

# 3D Scatter Plot
st.subheader("3D Scatter Plot")
fig_scatter = px.scatter_3d(
    sampled_df,
    x="loan_amount",
    y="rate_of_interest",
    z="age", 
    color='age',
    size='rate_of_interest',
    symbol='loan_amount',
    title="3D Scatter Plot of Loan Amount, Rate of Interest, and Age"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------
# Cleanup
# ---------------------------

# Stop the SparkContext when the app stops
def stop_spark():
    sc.stop()

atexit.register(stop_spark)

# ---------------------------
# Footer or Additional Information
# ---------------------------

st.markdown("""
---
**Technologies Used:**
- [Streamlit](https://streamlit.io/)
- [PySpark](https://spark.apache.org/docs/latest/api/python/index.html)
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [Plotly](https://plotly.com/)
""")
