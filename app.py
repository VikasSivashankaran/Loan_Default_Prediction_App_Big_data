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

# ---------------------------
# Spark Configuration and Initialization
# ---------------------------

@st.cache_resource
def initialize_spark():
    os.environ['PYSPARK_PYTHON'] = r'C:\Users\GowthamMaheswar\AppData\Local\Programs\Python\Python312\python.exe'
    os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\GowthamMaheswar\AppData\Local\Programs\Python\Python312\python.exe'
    
    conf = SparkConf() \
        .setAppName('Loan_Default_Prediction') \
        .setMaster("local[*]") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g") \
        .set("spark.network.timeout", "800s") \
        .set("spark.executor.cores", "2")
    
    sc = SparkContext.getOrCreate(conf=conf)
    sql_context = SQLContext(sc)
    return sc, sql_context

# Initialize Spark
sc, sql_context = initialize_spark()

# ---------------------------
# Data Loading and Display
# ---------------------------

DATA_PATH = "Loan_Default.csv"  # Specify your data path here

@st.cache_data
def load_data_pandas(filepath):
    df = pd.read_csv(filepath)
    columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
    df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())
    return df

def load_data_spark(filepath):
    df_spark = sql_context.read.csv(filepath, header=True, inferSchema=True)
    return df_spark

# Load data using Spark
df_spark = load_data_spark(DATA_PATH)

st.header("Dataset Schema")
with st.expander("View Schema"):
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
output_columns = columns_to_impute

imputer = Imputer(inputCols=columns_to_impute, outputCols=output_columns)
assembler = VectorAssembler(inputCols=['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV'],
                            outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

pipeline = Pipeline(stages=[imputer, assembler, scaler])

with st.spinner("Preprocessing data..."):
    model = pipeline.fit(df_spark)
    df_transformed = model.transform(df_spark)
st.success("Data preprocessing completed successfully.")

train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=42)

st.write(f"**Training Data Count:** {train_data.count()}")
st.write(f"**Test Data Count:** {test_data.count()}")

# ---------------------------
# Logistic Regression Model
# ---------------------------

st.header("Logistic Regression Model")

lr = LogisticRegression(featuresCol='scaled_features', labelCol='Status')

with st.spinner("Training Logistic Regression model..."):
    lr_model = lr.fit(train_data)
st.success("Logistic Regression model trained successfully.")

predictions = lr_model.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol='Status', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
roc_auc = evaluator.evaluate(predictions)

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
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)

@st.cache_resource
def train_pytorch_model(filepath):
    df = pd.read_csv(filepath)
    columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
    df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())
    
    X = df[['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV']].values
    y = df['Status'].values
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    input_dim = X.shape[1]
    pytorch_model = SimpleModel(input_dim)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
    
    for epoch in range(10):
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
    
    torch.save(pytorch_model.state_dict(), 'loan_prediction_model.pth')
    return pytorch_model

with st.spinner("Training PyTorch model..."):
    pytorch_model = train_pytorch_model(DATA_PATH)
st.success("PyTorch model trained and saved successfully.")

def load_pytorch_model(filepath, input_dim):
    model = SimpleModel(input_dim)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

pytorch_model = load_pytorch_model('loan_prediction_model.pth', input_dim=6)

# ---------------------------
# User Input for Prediction
# ---------------------------

st.header("Loan Eligibility Prediction")

def user_input_features():
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, value=10000.0)
    rate_of_interest = st.sidebar.number_input("Rate of Interest", min_value=0.0, value=5.0)
    property_value = st.sidebar.number_input("Property Value", min_value=0.0, value=200000.0)
    income = st.sidebar.number_input("Income", min_value=0.0, value=50000.0)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=700)
    ltv = st.sidebar.number_input("Loan-to-Value (LTV)", min_value=0.0, max_value=100.0, value=80.0)

    return [loan_amount, rate_of_interest, property_value, income, credit_score, ltv]

input_data = user_input_features()

input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    prediction = pytorch_model(input_tensor)
    predicted_class = torch.argmax(prediction, dim=1).item()

st.subheader("Prediction Result")
st.write("Predicted Class:", "Default" if predicted_class == 1 else "Non-Default")

# ---------------------------
# Plotting with Plotly
# ---------------------------

st.header("Visualizations")

# Load data for visualization using pandas
loan_data = load_data_pandas(DATA_PATH)

# 1. Distribution of Loan Amounts by Status
fig1 = px.histogram(loan_data, x='loan_amount', color='Status', barmode='overlay',
                    labels={'Status': 'Loan Status'},
                    title="Distribution of Loan Amounts by Status")
st.plotly_chart(fig1)

if 'age' in loan_data.columns:
    sampled_df = loan_data.sample(n=min(100, len(loan_data)))  # Sampling for better performance
    
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
else:
    st.warning("The 'age' column is not available in the dataset for 3D plots.")

# ---------------------------
# Clean-up
# ---------------------------

atexit.register(sc.stop)
