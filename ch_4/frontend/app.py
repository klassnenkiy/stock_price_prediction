import streamlit as st

st.sidebar.title("Stock Price Prediction")
page = st.sidebar.radio("Navigate", ["EDA", "Model Training", "Predictions"])

if page == "EDA":
    st.write("## Exploratory Data Analysis")

elif page == "Model Training":
    st.write("## Train Your Model")

elif page == "Predictions":
    st.write("## Predict Stock Prices")

