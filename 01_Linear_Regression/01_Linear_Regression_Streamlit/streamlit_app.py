import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="House Price Predictor")

st.title("🏠 House Price Predictor")
st.write("Powered by Linear Regression")

size = st.number_input(
    "Enter House Size (sq ft)",
    min_value=100,
    max_value=10000,
    value=1800
)

if st.button("Predict Price"):
    prediction = model.predict([[size]])[0]

    st.success(
        f"Predicted Price: ₹ {prediction:.2f} Lakh"
    )