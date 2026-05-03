import streamlit as st
import pickle
import numpy as np

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------
# TITLE
# -------------------------------
st.title("Customer Churn Prediction System")

st.write("Enter customer details:")

# -------------------------------
# INPUT FIELDS
# -------------------------------
age = st.number_input("Age", min_value=1, max_value=100, value=30)
income = st.number_input("Income", min_value=0.0, value=50000.0)
spending = st.number_input("Spending Score", min_value=0.0, max_value=100.0, value=50.0)
purchase = st.number_input("Purchase Amount", min_value=0.0, value=2000.0)
session = st.number_input("Session Time", min_value=0.0, value=100.0)
review = st.number_input("Review Score", min_value=0.0, max_value=5.0, value=3.0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    # Arrange in same order as training
    input_data = np.array([[age, income, spending, purchase, review, session]])

    # Scale input
    input_data = scaler.transform(input_data)

    # Get probability
    prob = model.predict_proba(input_data)[0][1]

    # Decision
    if prob > 0.7:
        st.error(f"Customer will churn ❌\nProbability: {prob:.2f}")
    else:
        st.success(f"Customer will stay ✅\nProbability: {1 - prob:.2f}")