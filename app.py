import streamlit as st
import pandas as pd
import pickle

# Load the model
model = pickle.load(open("model.pkl", "rb"))

st.title("💻 Laptop Purchase Predictor")
st.write("Predict if a customer will buy a laptop based on their Age and Income.")

# Input fields
age = st.slider("Select Age", 18, 60, 25)
income = st.number_input("Enter Monthly Income (₹)", min_value=10000, max_value=100000, value=30000, step=1000)

# Predict
if st.button("Predict"):
    prediction = model.predict([[age, income]])
    result = "Yes ✅" if prediction[0] == 1 else "No ❌"
    st.success(f"Will the customer buy a laptop? — **{result}**")

# Show sample data
if st.checkbox("Show sample dataset"):
    df = pd.read_csv("laptop_data.csv")
    st.dataframe(df)
