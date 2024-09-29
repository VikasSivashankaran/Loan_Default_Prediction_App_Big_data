import os
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import subprocess

# Set the path to your Python executable for PySpark
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

# --------------------------------------------------------
# After the PySpark job completes, automatically start the Streamlit app
# --------------------------------------------------------

# Path to the Streamlit app (assuming it is in the same directory)
streamlit_app_path = os.path.join(os.getcwd(), 'app.py')

# Use subprocess to start Streamlit programmatically
subprocess.run(['streamlit', 'run', streamlit_app_path])
