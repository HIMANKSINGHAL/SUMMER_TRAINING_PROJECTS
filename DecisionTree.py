import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn import tree

model = joblib.load("DTmodel.pkl")

st.set_page_config(
    page_title="House Quality Prediction",
    layout="centered"
)

st.title("House Quality Prediction")

st.write("Enter the house features")

Area = st.number_input(
    "Area (0 to 5000 sq ft)",
    min_value=0.0,
    max_value=5000.0,
    value=500.0
)

Bedrooms = st.number_input(
    "Number of Bedrooms (0 to 6)",
    min_value=0,
    max_value=6,
    value=2
)

Floors = st.number_input(
    "Number of Floors (0 to 5)",
    min_value=0,
    max_value=5,
    value=1
)

Bathrooms = st.number_input(
    "Number of Bathrooms (0 to 4)",
    min_value=0,
    max_value=4,
    value=1
)

st.write("0:Excellent, 1:Fair, 2:Good, 3:Poor")

if st.button("Predict"):

    data = pd.DataFrame(
        [[
            Area,
            Bedrooms,
            Floors,
            Bathrooms
        ]],
        columns=[
            "Area",
            "Bedrooms",
            "Floors",
            "Bathrooms"
        ]
    )

    prediction = model.predict(data)

    st.success(f"Prediction : {prediction[0]}")


if st.checkbox("Show Decision Tree"):

    fig, ax = plt.subplots(figsize=(20, 15))

    tree.plot_tree(
        model,                   # your loaded Decision Tree model
        filled=True,
        feature_names=["Area", "Bedrooms", "Floors", "Bathrooms"],
        rounded=True,
        fontsize=10
    )

    st.pyplot(fig)