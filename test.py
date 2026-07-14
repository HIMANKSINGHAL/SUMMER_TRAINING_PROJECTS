import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Regression Demo", layout="centered")

st.title("📈 Linear / Logistic Regression Demo")

# ---------- Mode selection ----------
mode = st.radio("Choose model type", ["Linear Regression", "Logistic Regression"], horizontal=True)

st.write(
    "Upload your own CSV, or use generated sample data, to fit and visualize "
    f"a simple **{mode}** model."
)

# ---------- Data source ----------
data_source = st.radio("Choose data source", ["Generate sample data", "Upload CSV"])

if data_source == "Generate sample data":
    st.subheader("Sample Data Settings")
    n_points = st.slider("Number of points", 10, 500, 100)
    rng = np.random.default_rng(42)

    if mode == "Linear Regression":
        noise = st.slider("Noise level", 0.0, 50.0, 10.0)
        slope = st.slider("True slope", -10.0, 10.0, 3.0)
        intercept = st.slider("True intercept", -50.0, 50.0, 5.0)

        x = rng.uniform(0, 100, n_points)
        y = slope * x + intercept + rng.normal(0, noise, n_points)
        df = pd.DataFrame({"x": x, "y": y})

    else:  # Logistic Regression
        separation = st.slider("Class separation", 0.5, 10.0, 3.0)
        threshold = st.slider("True decision boundary (x value)", 0.0, 100.0, 50.0)

        x = rng.uniform(0, 100, n_points)
        # Probability of class 1 rises smoothly around the threshold
        prob = 1 / (1 + np.exp(-(x - threshold) / (100 / separation / 10)))
        y = rng.binomial(1, prob)
        df = pd.DataFrame({"x": x, "y": y})

else:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(cols) < 2:
            st.error("Need at least two numeric columns.")
            st.stop()

        x_col = st.selectbox("Select X column (feature)", cols, index=0)
        y_col = st.selectbox("Select Y column (target)", cols, index=1)
        df = df[[x_col, y_col]].dropna()
        df.columns = ["x", "y"]

        if mode == "Logistic Regression":
            unique_vals = sorted(df["y"].unique())
            if len(unique_vals) != 2:
                st.error(
                    f"Logistic Regression needs a binary target (found {len(unique_vals)} "
                    f"unique values: {unique_vals}). Pick a different column or use Linear Regression."
                )
                st.stop()
            # Map to 0/1 if not already
            df["y"] = (df["y"] == unique_vals[1]).astype(int)
            st.caption(f"Mapped '{unique_vals[0]}' → 0, '{unique_vals[1]}' → 1")
    else:
        st.info("Please upload a CSV file to continue.")
        st.stop()

# ---------- Fit model ----------
X = df[["x"]].values
y = df["y"].values

if mode == "Linear Regression":
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    st.subheader("Model Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Slope (coef)", f"{model.coef_[0]:.3f}")
    col2.metric("Intercept", f"{model.intercept_:.3f}")
    col3.metric("R² score", f"{r2:.3f}")
    st.caption(f"RMSE: {rmse:.3f}")

    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], alpha=0.6, label="Data")
    order = np.argsort(df["x"].values)
    ax.plot(df["x"].values[order], y_pred[order], color="red", linewidth=2, label="Fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Try a Prediction")
    x_input = st.number_input("Enter an x value", value=float(df["x"].mean()))
    prediction = model.predict([[x_input]])[0]
    st.write(f"Predicted y: **{prediction:.3f}**")

else:  # Logistic Regression
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    st.subheader("Model Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Coefficient", f"{model.coef_[0][0]:.3f}")
    col2.metric("Intercept", f"{model.intercept_[0]:.3f}")
    col3.metric("Accuracy", f"{acc:.3f}")

    st.text("Confusion matrix:")
    st.dataframe(
        pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    )

    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], alpha=0.6, c=df["y"], cmap="coolwarm", edgecolors="k", label="Data")

    x_range = np.linspace(df["x"].min(), df["x"].max(), 300).reshape(-1, 1)
    y_curve = model.predict_proba(x_range)[:, 1]

    ax.plot(x_range, y_curve, color="red", linewidth=2, label="Predicted probability")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision threshold (0.5)")
    ax.set_xlabel("x")
    ax.set_ylabel("y / Probability of class 1")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Try a Prediction")
    x_input = st.number_input("Enter an x value", value=float(df["x"].mean()))
    pred_class = model.predict([[x_input]])[0]
    pred_prob = model.predict_proba([[x_input]])[0][1]
    st.write(f"Predicted class: **{pred_class}**  |  Probability of class 1: **{pred_prob:.3f}**")

with st.expander("View raw data"):
    st.dataframe(df)
