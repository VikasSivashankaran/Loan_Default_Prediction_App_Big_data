import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import pandas as pd
import plotly.express as px
import atexit

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
# Data Loading and Display
# ---------------------------

@st.cache_data
def load_data_pandas(file):
    try:
        # Attempt to read the CSV file
        df = pd.read_csv(file)
        
        # Check if DataFrame is empty
        if df.empty:
            st.error("Uploaded CSV file is empty. Please upload a valid file.")
            st.stop()
        
        # Columns to impute
        columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
        
        # Verify if required columns exist
        missing_columns = [col for col in columns_to_impute if col not in df.columns]
        if missing_columns:
            st.error(f"The following required columns are missing in the uploaded file: {missing_columns}")
            st.stop()
        
        # Impute missing values with column mean
        imputer = SimpleImputer(strategy='mean')
        df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
        
        # Additional Validation: Check if 'Status' column exists
        if 'Status' not in df.columns:
            st.error("The 'Status' column is missing from the uploaded file. Please include it for prediction.")
            st.stop()
        
        return df
    except pd.errors.EmptyDataError:
        st.error("No columns to parse from the uploaded file. Please ensure it's a valid CSV with headers.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please ensure it's properly formatted.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the file: {e}")
        st.stop()

# File uploader for Loan_Default.csv
uploaded_file = st.sidebar.file_uploader("Upload Loan_Default.csv", type=["csv"])

if uploaded_file is not None:
    # Read uploaded file into pandas DataFrame
    df_pandas = load_data_pandas(uploaded_file)
    st.success("CSV file uploaded and validated successfully!")
    st.balloons()
else:
    st.warning("Please upload the `Loan_Default.csv` file to proceed.")
    st.stop()

st.header("Dataset Schema")
with st.expander("View Schema"):
    st.text(df_pandas.dtypes)

st.header("Sample Data")
st.dataframe(df_pandas.head())

# ---------------------------
# Data Preprocessing with Pandas
# ---------------------------

st.header("Data Preprocessing")

# Features and target
X = df_pandas[['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV']]
y = df_pandas['Status']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success("Data preprocessing completed successfully.")

# ---------------------------
# Train-Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

st.write(f"**Training Data Count:** {X_train.shape[0]}")
st.write(f"**Test Data Count:** {X_test.shape[0]}")

# ---------------------------
# Logistic Regression Model
# ---------------------------

st.header("Logistic Regression Model")

# Logistic Regression model
lr = LogisticRegression()

# Fit the model to training data
with st.spinner("Training Logistic Regression model..."):
    lr.fit(X_train, y_train)
st.success("Logistic Regression model trained successfully.")

# Make predictions on test data
predictions = lr.predict(X_test)
probs = lr.predict_proba(X_test)[:, 1]

# Evaluate the model using ROC-AUC
roc_auc = roc_auc_score(y_test, probs)

# Calculate prediction accuracy
accuracy = accuracy_score(y_test, predictions)

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
def train_pytorch_model(X, y):
    try:
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)
        
        # Create a DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Create and train the model
        input_dim = X.shape[1]  # Number of features
        pytorch_model = SimpleModel(input_dim)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
        
        st.write("Starting PyTorch model training...")
        st.snow()  # Trigger snow effect during training
        
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
    except Exception as e:
        st.error(f"An error occurred during PyTorch model training: {e}")
        st.stop()

# Train PyTorch model
with st.spinner("Training PyTorch model..."):
    pytorch_model = train_pytorch_model(X_train, y_train)
st.success("PyTorch model trained and saved successfully.")

# Load the pre-trained PyTorch model
def load_pytorch_model(filepath, input_dim):
    try:
        model = SimpleModel(input_dim)
        model.load_state_dict(torch.load(filepath))
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the PyTorch model: {e}")
        st.stop()

pytorch_model = load_pytorch_model('loan_prediction_model.pth', input_dim=6)

# ---------------------------
# Loan Eligibility Prediction and Model Logic
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

# ---------------------------
# Implementing Rule-Based Logic
# ---------------------------

st.subheader("Eligibility Criteria Analysis")

def eligibility_logic(input_data):
    """
    Implements rule-based logic for loan eligibility.
    """
    reasons = []
    eligible = True
    
    # Example criteria
    if input_data['Credit_Score'].values[0] < 600:
        eligible = False
        reasons.append("Credit Score below 600.")
    
    if input_data['income'].values[0] < 20000:
        eligible = False
        reasons.append("Income below $20,000.")
    
    if input_data['loan_amount'].values[0] > 5 * input_data['income'].values[0]:
        eligible = False
        reasons.append("Loan amount exceeds 5 times the income.")
    
    if input_data['LTV'].values[0] > 90:
        eligible = False
        reasons.append("Loan-to-Value (LTV) ratio above 90%.")
    
    return eligible, reasons

is_eligible, eligibility_reasons = eligibility_logic(input_df)

if is_eligible:
    st.success("Based on eligibility criteria, **your loan is eligible for approval**.")
else:
    st.error("Based on eligibility criteria, **your loan is not eligible for approval**.")
    for reason in eligibility_reasons:
        st.write(f"- {reason}")

# ---------------------------
# Make Prediction Using Models
# ---------------------------

if st.button("Make Prediction"):
    # Trigger a snow effect to indicate prediction is in progress
    st.snow()
    
    # ---------------------------
    # Prediction Using Logistic Regression Model
    # ---------------------------
    
    st.subheader("Prediction using Logistic Regression")
    
    try:
        # Preprocess the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        lr_pred = lr.predict(input_scaled)[0]
        
        # Interpret the prediction
        if lr_pred == 1:
            lr_prediction_text = "The loan is likely to be **sanctioned**."
        else:
            lr_prediction_text = "The loan is likely to be **rejected**."
        
        st.write(lr_prediction_text)
    except Exception as e:
        st.error(f"An error occurred during Logistic Regression prediction: {e}")
    
    # ---------------------------
    # Prediction Using PyTorch Model
    # ---------------------------
    
    st.subheader("Prediction using Neural Network (PyTorch)")
    
    try:
        # Convert user input into tensor for PyTorch
        user_input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        # Make prediction using PyTorch model
        with torch.no_grad():
            output = pytorch_model(user_input_tensor)
            pytorch_pred = torch.argmax(output, dim=1).item()  # Get the predicted class
        
        # Interpret the prediction
        if pytorch_pred == 1:
            pytorch_prediction_text = "The loan is likely to be **sanctioned**."
        else:
            pytorch_prediction_text = "The loan is likely to be **rejected**."
        
        st.write(pytorch_prediction_text)
    except Exception as e:
        st.error(f"An error occurred during PyTorch prediction: {e}")
    
    # ---------------------------
    # Combined Prediction Interpretation
    # ---------------------------
    
    st.subheader("Combined Prediction Analysis")
    
    try:
        if lr_pred == pytorch_pred:
            st.write(f"Both models agree: The loan is likely to be **{'sanctioned' if lr_pred == 1 else 'rejected'}**.")
        else:
            st.write("The models have differing predictions:")
            st.write(f"- Logistic Regression predicts: **{'Sanctioned' if lr_pred == 1 else 'Rejected'}**.")
            st.write(f"- Neural Network predicts: **{'Sanctioned' if pytorch_pred == 1 else 'Rejected'}**.")
    except NameError:
        st.error("Model predictions are not available.")
    
    # ---------------------------
    # Additional Information Based on Eligibility Logic
    # ---------------------------
    
    st.subheader("Eligibility Criteria Influence")
    
    if is_eligible:
        st.write("Eligibility criteria indicate that the loan is eligible. Model predictions further support this decision.")
    else:
        st.write("Despite model predictions, eligibility criteria indicate that the loan should be **rejected** based on the following reasons:")
        for reason in eligibility_reasons:
            st.write(f"- {reason}")

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

# No Spark to stop since we're using Pandas
