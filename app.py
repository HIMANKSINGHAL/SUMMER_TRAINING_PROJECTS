import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor (SVR)", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    bundle = joblib.load("svr_model.pkl")
    return bundle["model"], bundle["scaler_X"], bundle["scaler_y"], bundle["feature_names"]

model, scaler_X, scaler_y, feature_names = load_model()

st.title("House Price Predictor")
st.write("This app uses a **Support Vector Regressor (SVR)** to estimate house prices "
         "based on area, rooms, age, and location quality.")

st.divider()
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    area_sqft = st.number_input("Area (sq ft)", min_value=200, max_value=10000, value=1500, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)

with col2:
    age_years = st.number_input("Age of Property (years)", min_value=0, max_value=100, value=10, step=1)
    location_score = st.slider("Location Score (1 = poor, 10 = prime)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)

st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    input_data = np.array([[area_sqft, bedrooms, bathrooms, age_years, location_score]])

    input_scaled = scaler_X.transform(input_data)

    pred_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]

    st.success(f"### Estimated Price: {prediction:.2f}")


