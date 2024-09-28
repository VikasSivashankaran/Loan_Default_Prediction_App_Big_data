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

@st.cache_data
def load_data_pandas(filepath):
    df = pd.read_csv(filepath)
    columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
    df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())
    return df

def load_data_spark(filepath):
    df_spark = sql_context.read.csv(filepath, header=True, inferSchema=True)
    return df_spark

# File uploader for Loan_Default.csv
uploaded_file = st.sidebar.file_uploader("Upload Loan_Default.csv", type=["csv"])

if uploaded_file is not None:
    with open("Loan_Default.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    data_path = "Loan_Default.csv"
else:
    st.warning("Please upload the `Loan_Default.csv` file to proceed.")
    st.stop()

df_spark = load_data_spark(data_path)

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
assembler = VectorAssembler(inputCols=['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV'], outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

pipeline = Pipeline(stages=[imputer, assembler, scaler])

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
    pytorch_model = train_pytorch_model(data_path)
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
    rate_of_interest = st.sidebar.number_input("Rate of Interest (%)", min_value=0.0, value=5.0)
    property_value = st.sidebar.number_input("Property Value", min_value=0.0, value=100000.0)
    income = st.sidebar.number_input("Annual Income", min_value=0.0, value=30000.0)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=600)
    ltv = st.sidebar.number_input("Loan to Value Ratio (%)", min_value=0.0, max_value=100.0, value=80.0)

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

input_data = user_input_features()

# Prediction
st.subheader("Predicted Loan Eligibility")
if st.button("Predict"):
    # Spark Prediction
    spark_input_data = sql_context.createDataFrame(input_data)
    spark_input_data = model.transform(spark_input_data)
    pred_label = spark_input_data.select('prediction').collect()[0][0]
    
    # PyTorch Prediction
    pytorch_input = torch.tensor(input_data.values, dtype=torch.float32)
    with torch.no_grad():
        pytorch_output = pytorch_model(pytorch_input)
        _, predicted_class = torch.max(pytorch_output, 1)

    st.write(f"**Spark Model Prediction:** {'Approved' if pred_label == 1 else 'Not Approved'}")
    st.write(f"**PyTorch Model Prediction:** {'Approved' if predicted_class.item() == 1 else 'Not Approved'}")

# ---------------------------
# Visualization of Predictions
# ---------------------------

st.header("Prediction Visualization")

# Create a DataFrame for visualization
predictions_df = pd.DataFrame(predictions.select('loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV', 'prediction').collect(), columns=['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV', 'prediction'])

# 3D Plotting with Plotly
fig = px.scatter_3d(predictions_df, x='loan_amount', y='property_value', z='income', color='prediction',
                     labels={'prediction': 'Prediction', 'loan_amount': 'Loan Amount', 'property_value': 'Property Value', 'income': 'Annual Income'},
                     title='3D Visualization of Predictions')
st.plotly_chart(fig)

# 3D Line Plotting
fig_line = px.line_3d(predictions_df, x='loan_amount', y='property_value', z='income', color='prediction',
                       labels={'prediction': 'Prediction', 'loan_amount': 'Loan Amount', 'property_value': 'Property Value', 'income': 'Annual Income'},
                       title='3D Line Visualization of Predictions')
st.plotly_chart(fig_line)

# ---------------------------
# Cleanup and Shutdown
# ---------------------------

def cleanup():
    sc.stop()

atexit.register(cleanup)
