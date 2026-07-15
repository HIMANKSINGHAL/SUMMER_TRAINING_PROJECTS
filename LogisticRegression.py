import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("logistic_regression_model1.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("Heart Disease Prediction")

st.write("Enter the patient Resting Blood Pressure")

RestingBP = st.number_input(
    "Resting Blood Pressure (0 to 200)",
    min_value=0.0,
    max_value=200.0,
    value=120.0
)
ST_Slope = st.number_input(
    "ST_Slope (0 or 2)",
    min_value=0.0,
    max_value=2.0,
    value=1.0
)

st.write("0:Down, 1:Flat, 2:Up")

if st.button("Predict"):

    data = pd.DataFrame(
        [[
            RestingBP,
            ST_Slope
        ]],
        columns=[
            "RestingBP",
            "ST_Slope"
        ]
    )

    prediction = model.predict(data)

    st.success(f"Prediction : {prediction[0]}")

