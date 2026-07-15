import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("linear_regression_model.pkl")

st.set_page_config(
    page_title="Average Resting Blood Pressure Prediction",
    layout="centered"
)

st.title("Average Resting Blood Pressure Prediction")

st.write("Enter the patient Age")

Age = st.number_input(
    "Age (0 to 120)",
    min_value=0,
    max_value=120,
    value=50
)

if st.button("Predict"):

    data = pd.DataFrame(
        [[
            Age
        ]],
        columns=[
            "Age"
        ]
    )

    prediction = model.predict(data)

    st.success(f"Prediction : {prediction[0]}")
