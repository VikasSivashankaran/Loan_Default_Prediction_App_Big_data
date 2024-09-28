import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Set the background image
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('image.jpeg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Data Loading and Display
# ---------------------------

@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    # Columns to impute
    columns_to_impute = ['rate_of_interest', 'property_value', 'income', 'LTV']
    # Impute missing values with column mean
    df[columns_to_impute] = df[columns_to_impute].fillna(df[columns_to_impute].mean())
    return df

# File uploader for Loan_Default.csv
uploaded_file = st.sidebar.file_uploader("Upload Loan_Default.csv", type=["csv"])

if uploaded_file is not None:
    data_path = "Loan_Default.csv"
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Show balloon effect after file upload
    st.balloons()  
else:
    st.warning("Please upload the `Loan_Default.csv` file to proceed.")
    st.stop()

# Load data using Pandas
df = load_data(data_path)

st.header("Dataset Schema")
st.write(df.dtypes)

st.header("Sample Data")
st.dataframe(df.head())

# ---------------------------
# Data Preprocessing
# ---------------------------

st.header("Data Preprocessing")

# Prepare features and labels
X = df[['loan_amount', 'rate_of_interest', 'property_value', 'income', 'Credit_Score', 'LTV']].values
y = df['Status'].values  # Assuming 'Status' is the column to predict

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.write(f"**Training Data Count:** {len(X_train)}")
st.write(f"**Test Data Count:** {len(X_test)}")

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

# Train PyTorch model
@st.cache_resource
def train_pytorch_model(X_train, y_train):
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    # Create a DataLoader
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Create and train the model
    input_dim = X_train.shape[1]  # Number of features
    pytorch_model = SimpleModel(input_dim)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

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

pytorch_model = train_pytorch_model(X_train, y_train)
st.success("PyTorch model trained and saved successfully.")

# Load the pre-trained PyTorch model
def load_pytorch_model(filepath):
    model = SimpleModel(input_dim=X_train.shape[1])
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    return model

pytorch_model = load_pytorch_model('loan_prediction_model.pth')

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

# Preprocess user input for prediction
user_input_array = scaler.transform(input_df)
user_input_tensor = torch.tensor(user_input_array, dtype=torch.float32)

# Make prediction using PyTorch model
with torch.no_grad():
    output = pytorch_model(user_input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()  # Get the predicted class

# Apply custom logic for prediction
low_income_threshold = 30000  # Example threshold
high_loan_threshold = 200000  # Example threshold
low_property_value_threshold = 100000  # Example threshold

if (input_df['income'].values[0] < low_income_threshold and 
    input_df['loan_amount'].values[0] > high_loan_threshold and 
    input_df['property_value'].values[0] < low_property_value_threshold):
    predicted_class = 0  # Reject loan
    st.warning("Based on the criteria, the loan is likely to be **rejected**.")
    st.snow()  # Add snow effect if the loan is rejected
else:
    if predicted_class == 1:
        prediction_text = "The loan is likely to be **sanctioned**."
        st.balloons()  # Add balloons effect if the loan is sanctioned
    else:
        prediction_text = "The loan is likely to be **rejected**."

st.subheader("Prediction")
st.write(prediction_text)

# ---------------------------
# 3D Visualizations
# ---------------------------

st.header("3D Visualizations")

# Sample 100 rows
sampled_df = df.sample(n=min(100, len(df)), random_state=42)

# Drop rows with NaN in 'loan_amount', 'rate_of_interest', or 'age'
sampled_df = sampled_df.dropna(subset=['loan_amount', 'rate_of_interest', 'age'])

# Ensure that 'rate_of_interest' has no negative or zero values if required
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



