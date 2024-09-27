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

# Create SQLContext from SparkContext
sql_context = SQLContext(sc)

# Load dataset into a Spark DataFrame
df_spark = sql_context.read.csv('Loan_Default.csv', header=True, inferSchema=True)

# Show schema to understand the data structure
df_spark.printSchema()

# Show a few rows
df_spark.show(5)

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
print(f"ROC-AUC: {roc_auc}")

# Calculate prediction accuracy
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="Status", predictionCol="prediction", metricName="accuracy")
accuracy = accuracy_evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Stop the SparkContext
sc.stop()

# Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Binary classification

    def forward(self, x):
        return self.fc(x)

# Load the dataset into pandas for PyTorch training
df = pd.read_csv('Loan_Default.csv')

# Preprocess the data: Handle missing values
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
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = pytorch_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the trained PyTorch model
torch.save(pytorch_model, 'loan_prediction_model.pth')  # Save the model

# Load the pre-trained PyTorch model
pytorch_model = torch.load('loan_prediction_model.pth')
pytorch_model.eval()  # Set the model to evaluation mode

# Collect user input for prediction
print("Please enter the following details to check loan eligibility:")
loan_amount = float(input("Loan Amount: "))
rate_of_interest = float(input("Rate of Interest: "))
property_value = float(input("Property Value: "))
income = float(input("Income: "))
credit_score = float(input("Credit Score: "))
ltv = float(input("Loan-to-Value (LTV): "))

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

# Make prediction using PyTorch model
with torch.no_grad():
    output = pytorch_model(user_input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()  # Get the predicted class

# Interpret the prediction
if predicted_class == 1:
    print("Based on your inputs, the loan is likely to be sanctioned (PyTorch model).")
else:
    print("Based on your inputs, the loan is likely to be rejected (PyTorch model).")


sampled_df = df.sample(n=100, random_state=42)  # Randomly select 100 rows

# 3D Line Plot
fig = px.line_3d(sampled_df, x="loan_amount", y="rate_of_interest", z="age")
fig.show()

# 3D Scatter Plot
fig = px.scatter_3d(sampled_df, x="loan_amount", y="rate_of_interest", z="age", 
                    color='age', size='rate_of_interest', symbol='loan_amount')
fig.show()