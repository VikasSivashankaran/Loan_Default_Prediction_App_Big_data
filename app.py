import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import plotly.express as px
import atexit
import os

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
# PyTorch Model Definition
# ---------------------------

class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Binary classification

    def forward(self, x):
        return self.fc(x)

# Load the pre-trained PyTorch model
def load_pytorch_model(filepath, input_dim):
    model = SimpleModel(input_dim)
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    return model

# ---------------------------
# User Input for Prediction
# ---------------------------

st.header("Loan Eligibility Prediction")

# Collect user input for prediction using Streamlit widgets
def user_input_features():
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, value=10000.0)
    rate_of_interest = st.sidebar.number_input("Rate of Interest", min_value=0.0, value=5.0)
    property_value = st.sidebar.number_input("Property Value", min_value=0.0, value=300000.0)
    income = st.sidebar.number_input("Annual Income", min_value=0.0, value=60000.0)
    credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=700)
    ltv = st.sidebar.number_input("LTV (Loan-To-Value Ratio)", min_value=0.0, max_value=1.0, value=0.8)
    return [loan_amount, rate_of_interest, property_value, income, credit_score, ltv]

user_input = user_input_features()
user_input_df = pd.DataFrame([user_input], columns=['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV'])

# Load the pre-trained PyTorch model
pytorch_model = load_pytorch_model('loan_prediction_model.pth', input_dim=6)

# Convert user input to tensor
input_tensor = torch.tensor(user_input_df.values, dtype=torch.float32)

# Make prediction with the PyTorch model
with torch.no_grad():
    prediction = pytorch_model(input_tensor)
    predicted_class = torch.argmax(prediction, dim=1).item()

# ---------------------------
# Loan Approval Logic
# ---------------------------

st.write("### Prediction Result:")
if predicted_class == 0:
    # Apply custom logic for loan approval
    if user_input[3] < 30000 and user_input[0] > 50000:  # If income < 30000 and loan amount > 50000
        st.error("Loan is likely to be denied due to insufficient income.")
    else:
        st.success("Loan is likely to be approved.")
else:
    st.error("Loan is likely to be denied.")

# ---------------------------
# Visualization of Predictions
# ---------------------------

st.header("Prediction Visualization")

# Load CSV file
uploaded_file = st.sidebar.file_uploader("Upload Loan_Default.csv", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    data_path = "Loan_Default.csv"
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load data for visualization
    df = pd.read_csv(data_path)

    # Visualization
    fig = px.scatter_3d(df, x='loan_amount', y='property_value', z='income', color='Status',
                         labels={'Status': 'Prediction', 'loan_amount': 'Loan Amount', 'property_value': 'Property Value', 'income': 'Annual Income'},
                         title='3D Visualization of Loan Data')
    st.plotly_chart(fig)

# ---------------------------
# Cleanup
# ---------------------------

def cleanup():
    if os.path.exists("Loan_Default.csv"):
        os.remove("Loan_Default.csv")  # Clean up the temporary file

atexit.register(cleanup)
