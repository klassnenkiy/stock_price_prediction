import streamlit as st
import requests
import plotly.graph_objects as go

def show_model_training_page():
    st.header("Model Training")

    ticker = st.text_input("Enter Ticker (e.g., SBER)", "")
    window_size = st.number_input("Window Size", min_value=1, value=10)

    if st.button("Start Training"):
        if ticker:
            request_data = {"ticker": ticker, "window_size": window_size}
            response = requests.post("http://localhost:8000/fit/", json=request_data)
            if response.status_code == 200:
                st.success("Training started successfully.")
            else:
                st.error(f"Failed to start training: {response.text}")
        else:
            st.error("Please enter a ticker to train the model.")

    st.subheader("Training Curves")
    training_data_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
    if training_data_file:
        training_data = pd.read_csv(training_data_file)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=training_data['Epoch'], y=training_data['Train_Loss'], mode='lines', name='Train Loss'))
        fig.add_trace(go.Scatter(x=training_data['Epoch'], y=training_data['Val_Loss'], mode='lines', name='Validation Loss'))
        fig.update_layout(title="Training and Validation Loss", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig)
