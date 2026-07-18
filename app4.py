import streamlit as st
import pickle
import pandas as pd

# Load model
with open("randomforestclassifier12.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Random Forest Loan Amount Predictor",
    layout="centered"
)

st.title("🌲 Random Forest Loan Amount Predictor")
st.write("Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=0,
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    [0, 1]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# Encode exactly like training
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

area = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}

property_area = area[property_area]

data = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
})

if st.button("Predict"):
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("🟢 High Loan Amount")
    else:
        st.error("🔴 Low Loan Amount")