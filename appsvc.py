import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Loan Approval Predictor (SVC)", layout="centered")

@st.cache_resource
def load_model():
    bundle = joblib.load("svc_model.pkl")
    return bundle["model"], bundle["scaler"], bundle["feature_names"], bundle["class_names"]

model, scaler, feature_names, class_names = load_model()

st.title("Loan Approval Predictor")
st.write("This app uses a **Support Vector Classifier (SVC)** to predict whether a loan "
         "application is likely to be **Approved** or **Rejected**.")

st.divider()
st.subheader("Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    income = st.number_input("Annual Income ($)", min_value=0, max_value=1_000_000, value=50000, step=1000)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650, step=1)

with col2:
    loan_amount = st.number_input("Loan Amount Requested", min_value=500, max_value=500000, value=15000, step=500)
    years_employed = st.number_input("Years Employed", min_value=0, max_value=50, value=5, step=1)

st.divider()

if st.button("Predict Loan Status", type="primary", use_container_width=True):
    input_data = np.array([[age, income, credit_score, loan_amount, years_employed]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    label = class_names[prediction]
    confidence = proba[prediction] * 100

    if label == "Approved":
        st.success(f"Loan {label}")
    else:
        st.error(f"Loan {label}")
