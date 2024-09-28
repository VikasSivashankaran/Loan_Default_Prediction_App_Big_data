import streamlit as st
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px

# Streamlit title
st.title('Loan Default Prediction App')

# Set up PySpark environment
@st.cache_resource
def initialize_spark():
    conf = SparkConf() \
        .setAppName('Loan_Default_Prediction') \
        .setMaster("local[*]") \
        .set("spark.executor.memory", "4g") \
        .set("spark.driver.memory", "4g") \
        .set("spark.network.timeout", "800s") \
        .set("spark.executor.cores", "2")
    sc = SparkContext(conf=conf)
    sql_context = SQLContext(sc)
    return sc, sql_context

# Load dataset and initialize pipeline
@st.cache_data
def load_and_preprocess_data(sql_context):
    # Load dataset into a Spark DataFrame
    df_spark = sql_context.read.csv('Loan_Default.csv', header=True, inferSchema=True)

    # Identify columns for imputation
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
    model = pipeline.fit(df_spark)
    df_transformed = model.transform(df_spark)

    return df_transformed

# PyTorch model definition
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Binary classification

    def forward(self, x):
        return self.fc(x)

# Custom loan approval logic
def custom_loan_logic(loan_amount, income, credit_score):
    if income < 50000 and loan_amount > 500000:
        return "Rejected: Low income for the requested loan amount."
    if credit_score < 600:
        return "Rejected: Poor credit score."
    return "Accepted: Eligible for loan."

# Initialize Spark and load data
sc, sql_context = initialize_spark()
df_transformed = load_and_preprocess_data(sql_context)

# User inputs
st.subheader('Enter the following details to check loan eligibility:')
loan_amount = st.number_input("Loan Amount:", min_value=0.0, value=100000.0)
rate_of_interest = st.number_input("Rate of Interest:", min_value=0.0, value=5.0)
property_value = st.number_input("Property Value:", min_value=0.0, value=300000.0)
income = st.number_input("Income:", min_value=0.0, value=60000.0)
credit_score = st.number_input("Credit Score:", min_value=0.0, value=700.0)
ltv = st.number_input("Loan-to-Value (LTV):", min_value=0.0, value=80.0)

# Prepare input for PyTorch and custom logic
user_input_dict = {
    'loan_amount': loan_amount,
    'rate_of_interest': rate_of_interest,
    'property_value': property_value,
    'income': income,
    'Credit_Score': credit_score,
    'LTV': ltv
}
user_input_df = pd.DataFrame([user_input_dict])

# Apply custom loan logic
decision = custom_loan_logic(loan_amount, income, credit_score)
st.write(f"Custom Loan Decision: {decision}")

# PyTorch prediction
input_dim = user_input_df.shape[1]
pytorch_model = SimpleModel(input_dim)
pytorch_model.load_state_dict(torch.load('loan_prediction_model.pth'))
pytorch_model.eval()

user_input_tensor = torch.tensor(user_input_df.values, dtype=torch.float32)
with torch.no_grad():
    output = pytorch_model(user_input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

# Interpret PyTorch prediction
if predicted_class == 1:
    st.write("Based on PyTorch Model: Loan likely to be sanctioned.")
else:
    st.write("Based on PyTorch Model: Loan likely to be rejected.")

# Logistic Regression using PySpark (optional evaluation on preloaded test set)
if st.button("Evaluate with PySpark"):
    # Split the data into training and test sets
    train_data, test_data = df_transformed.randomSplit([0.8, 0.2])

    # Logistic Regression model
    lr = LogisticRegression(featuresCol='scaled_features', labelCol='Status')

    # Fit the model to training data
    lr_model = lr.fit(train_data)

    # Make predictions on test data
    predictions = lr_model.transform(test_data)

    # Evaluate the model using ROC-AUC
    evaluator = BinaryClassificationEvaluator(labelCol='Status', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
    roc_auc = evaluator.evaluate(predictions)
    st.write(f"PySpark ROC-AUC: {roc_auc}")

    # Calculate prediction accuracy
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Status", predictionCol="prediction", metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(predictions)
    st.write(f"PySpark Accuracy: {accuracy}")

# Visualization with Plotly
if st.button("Show 3D Visualization"):
    df = pd.read_csv('Loan_Default.csv')
    sampled_df = df.sample(n=100, random_state=42)  # Randomly select 100 rows

    # 3D Scatter Plot
    fig = px.scatter_3d(sampled_df, x="loan_amount", y="rate_of_interest", z="age", 
                        color='age', size='rate_of_interest', symbol='loan_amount')
    st.plotly_chart(fig)

# Stop Spark context when Streamlit app ends
st.write("Stopping Spark Context...")
sc.stop()
