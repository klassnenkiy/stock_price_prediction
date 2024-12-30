import streamlit as st
import requests
import plotly.graph_objects as go


def compare_models():
    st.header("Compare Models")
    response = requests.get("http://127.0.0.1:8000/api/models/")
    models = response.json()
    tickers = [model['ticker'] for model in models]
    selected_tickers = st.multiselect("Select models to compare", tickers)

    if selected_tickers:
        response = requests.get(f"http://127.0.0.1:8000/api/models/metrics", params={"tickers": selected_tickers})
        metrics = response.json()
        fig = go.Figure()

        for metric in metrics:
            ticker = metric['ticker']
            fig.add_trace(go.Scatter(
                x=list(range(len(metric['train_losses']))),
                y=metric['train_losses'],
                mode='lines',
                name=f"{ticker} - Train Loss"
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(metric['val_losses']))),
                y=metric['val_losses'],
                mode='lines',
                name=f"{ticker} - Validation Loss"
            ))

        fig.update_layout(
            title="Comparison of Training and Validation Losses",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend_title="Model"
        )

        st.plotly_chart(fig)
