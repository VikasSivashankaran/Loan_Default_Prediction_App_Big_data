import os
import pandas as pd
import torch
import torch.nn as nn
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import plotly.express as px
import streamlit as st

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

sc = SparkContext(conf=conf)
sql_context = SQLContext(sc)

# Load dataset into a Spark DataFrame
df_spark = sql_context.read.csv('Loan_Default.csv', header=True, inferSchema=True)

# Preprocess the data
columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
imputer = Imputer(inputCols=columns_to_impute, outputCols=columns_to_impute)
assembler = VectorAssembler(inputCols=['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV'],
                            outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
pipeline = Pipeline(stages=[imputer, assembler, scaler])
model = pipeline.fit(df_spark)
df_transformed = model.transform(df_spark)

# Split the data into training and test sets
train_data, test_data = df_transformed.randomSplit([0.8, 0.2])
lr = LogisticRegression(featuresCol='scaled_features', labelCol='Status')
lr_model = lr.fit(train_data)
predictions = lr_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol='Status', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
roc_auc = evaluator.evaluate(predictions)

# Load the dataset into pandas for PyTorch training
df = pd.read_csv('Loan_Default.csv')
df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())
X = df[['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV']].values
y = df['Status'].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create a DataLoader
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)

input_dim = X.shape[1]
pytorch_model = SimpleModel(input_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = pytorch_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Save the trained PyTorch model
torch.save(pytorch_model, 'loan_prediction_model.pth')

# Load the pre-trained PyTorch model
pytorch_model = torch.load('loan_prediction_model.pth')
pytorch_model.eval()

# Streamlit app
st.title("Loan Default Prediction App")
st.write(f"ROC-AUC from Logistic Regression: {roc_auc}")

# User input
loan_amount = st.number_input("Loan Amount:", min_value=0.0, format="%.2f")
rate_of_interest = st.number_input("Rate of Interest:", min_value=0.0, format="%.2f")
property_value = st.number_input("Property Value:", min_value=0.0, format="%.2f")
income = st.number_input("Income:", min_value=0.0, format="%.2f")
credit_score = st.number_input("Credit Score:", min_value=0, max_value=850)
ltv = st.number_input("Loan-to-Value (LTV):", min_value=0.0, format="%.2f")

if st.button("Predict"):
    user_input_dict = {
        'loan_amount': loan_amount,
        'rate_of_interest': rate_of_interest,
        'property_value': property_value,
        'income': income,
        'Credit_Score': credit_score,
        'LTV': ltv
    }
    
    user_input_df = pd.DataFrame([user_input_dict])
    user_input_tensor = torch.tensor(user_input_df.values, dtype=torch.float32)

    with torch.no_grad():
        output = pytorch_model(user_input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    if predicted_class == 1:
        st.success("Based on your inputs, the loan is likely to be sanctioned (PyTorch model).")
    else:
        st.error("Based on your inputs, the loan is likely to be rejected (PyTorch model).")

# Visualization
st.header("Data Visualization")
sampled_df = df.sample(n=100, random_state=42)

# 3D Line Plot
fig_line = px.line_3d(sampled_df, x="loan_amount", y="rate_of_interest", z="age")
st.plotly_chart(fig_line)

# 3D Scatter Plot
fig_scatter = px.scatter_3d(sampled_df, x="loan_amount", y="rate_of_interest", z="age", 
                    color='age', size='rate_of_interest', symbol='loan_amount')
st.plotly_chart(fig_scatter)

# Stop the SparkContext
sc.stop()
