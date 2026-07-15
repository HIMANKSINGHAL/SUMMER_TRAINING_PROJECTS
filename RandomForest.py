import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn import tree

model1 = joblib.load("RandomReg2.pkl")

st.set_page_config(
    page_title="Salary Prediction",
    layout="centered"
)

st.title("Salary Prediction")

print(type(model1))

st.write("Enter the Years of Experience")

YearsExperience = st.number_input(
    "Years of Experience (1.1 to 10.5)",
    min_value=1.1,
    max_value=10.5,
    value=5.0
)


if st.button("Predict"):

    data = pd.DataFrame(
        [[
            YearsExperience
        ]],
        columns=[
            "YearsExperience"
        ]
    )

    prediction = model1.predict(data)

    st.success(f"Prediction : {prediction[0]}")


# if st.checkbox("Show Decision Tree"):

#     fig, ax = plt.subplots(figsize=(20, 15))

#     tree.plot_tree(
#         model,                   # your loaded Decision Tree model
#         filled=True,
#         feature_names=["Area", "Bedrooms", "Floors", "Bathrooms"],
#         rounded=True,
#         fontsize=10
#     )

#     st.pyplot(fig)

