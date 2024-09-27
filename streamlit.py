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

st.snow()
# Streamlit configuration
st.title("Loan Default Prediction App")
st.write("This app uses both PySpark and PyTorch models for loan default prediction.")

# Set the path to your Python executable for PySpark
os.environ['PYSPARK_PYTHON'] = r'C:\Users\GowthamMaheswar\AppData\Local\Programs\Python\Python312\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = r'C:\Users\GowthamMaheswar\AppData\Local\Programs\Python\Python312\python.exe'

# Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Binary classification

    def forward(self, x):
        return self.fc(x)

# Function to initialize Spark and run Spark-related operations
def run_spark_model():
    try:
        # Create Spark configuration and context
        conf = SparkConf() \
            .setAppName('Loan_Default_Prediction') \
            .setMaster("local[*]") \
            .set("spark.executor.memory", "4g") \
            .set("spark.driver.memory", "4g") \
            .set("spark.network.timeout", "800s") \
            .set("spark.executor.cores", "2")

        sc = SparkContext(conf=conf)
        sql_context = SQLContext(sc)

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
        st.write(f"ROC-AUC (PySpark Model): {roc_auc}")

        # Calculate prediction accuracy
        accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Status", predictionCol="prediction", metricName="accuracy")
        accuracy = accuracy_evaluator.evaluate(predictions)
        st.write(f"Accuracy (PySpark Model): {accuracy}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        sc.stop()

# Button to run Spark model
if st.button("Run Spark Model"):
    run_spark_model()  # Run Spark only when the button is pressed

# Load the dataset into pandas for PyTorch training
df = pd.read_csv('Loan_Default.csv')

# Preprocess the data: Handle missing values
columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())

# Collect user input for prediction
st.write("Enter details to check loan eligibility:")
loan_amount = st.number_input("Loan Amount:", value=100000.0)
rate_of_interest = st.number_input("Rate of Interest:", value=5.0)
property_value = st.number_input("Property Value:", value=200000.0)
income = st.number_input("Income:", value=50000.0)
credit_score = st.number_input("Credit Score:", value=700)
ltv = st.number_input("Loan-to-Value (LTV):", value=80.0)

# Prepare user input for PyTorch model
user_input_dict = {
    'loan_amount': loan_amount,
    'rate_of_interest': rate_of_interest,
    'property_value': property_value,
    'income': income,
    'Credit_Score': credit_score,
    'LTV': ltv
}

# Convert the user input into a DataFrame for PyTorch
user_input_df = pd.DataFrame([user_input_dict])

# Convert DataFrame to tensor
user_input_tensor = torch.tensor(user_input_df.values, dtype=torch.float32)

# Load the pre-trained PyTorch model
try:
    pytorch_model = SimpleModel(input_dim=len(user_input_df.columns))  # Ensure to initialize the model
    pytorch_model.load_state_dict(torch.load('loan_prediction_model.pth', map_location=torch.device('cpu')))  # Load model weights
    pytorch_model.eval()  # Set the model to evaluation mode

    # Make prediction using PyTorch model
    with torch.no_grad():
        output = pytorch_model(user_input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()  # Get the predicted class

    # Interpret the prediction
    if predicted_class == 1:
        st.success("Based on your inputs, the loan is likely to be sanctioned (PyTorch model).")
    else:
        st.error("Based on your inputs, the loan is likely to be rejected (PyTorch model).")

except Exception as e:
    st.error(f"Error loading PyTorch model: {e}")

# Visualization
st.write("3D Visualization of Data")

sampled_df = df.sample(n=100, random_state=42)  # Randomly select 100 rows

# 3D Line Plot
fig_line = px.line_3d(sampled_df, x="loan_amount", y="rate_of_interest", z="income")
st.plotly_chart(fig_line)

# 3D Scatter Plot
fig_scatter = px.scatter_3d(sampled_df, x="loan_amount", y="rate_of_interest", z="income", 
                            color='income', size='rate_of_interest', symbol='loan_amount')
st.plotly_chart(fig_scatter)
