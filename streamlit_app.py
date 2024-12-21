import streamlit as st
import requests

# Streamlit app title
st.title("Wine Quality Prediction App")

# Description
st.write("""
This Streamlit app interacts with the deployed FastAPI backend to predict wine quality based on input features.
""")

# Input fields
st.header("Input Wine Features")
fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
citric_acid = st.number_input("Citric Acid", value=0.0)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

# Prediction button
if st.button("Predict Quality"):
    # Prepare the input data for API
    input_data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }

    # API URL
    api_url = "https://wine-quality-project.onrender.com/predict"

    # Make a POST request to the FastAPI backend
    try:
        response = requests.post(api_url, json=input_data)
        if response.status_code == 200:
            prediction = response.json().get("prediction", "No prediction returned")
            st.success(f"Predicted Wine Quality: {prediction}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")