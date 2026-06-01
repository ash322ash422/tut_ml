import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import streamlit as st

from components.layout import (
    render_header,
    render_footer
)

from components.charts import show_price_chart
from components.metrics import show_stock_metrics

from backend.services.stock_service import StockService
from backend.services.prediction_service import PredictionService


def show():

    render_header(
        "📊 Stock Analysis",
        "Analyze real stock market data"
    )

    ticker = st.text_input( "Enter Stock Ticker", value="AAPL")

    if st.button("Analyze"):

        try:
            # Fetch Stock Data
            history = StockService.get_stock_data(ticker)

            info = StockService.get_stock_info(ticker)

            # Calculate Predictions
            history = PredictionService.calculate_moving_averages( history )
            predicted_price = PredictionService.forecast_next_day(history)
            signal = PredictionService.generate_trend_signal(history)
            

            # Success Message
            st.success(f"Loaded data for {ticker}")

            # Metrics Section
            st.subheader("Key Metrics" )

            show_stock_metrics(info)

            # Prediction Section
            st.subheader("Prediction")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted Next Close", round(predicted_price, 2))

            with col2:
                st.metric("Trend Signal",  signal)

            # Chart Section
            st.subheader("Historical Price Chart")

            show_price_chart( history, ticker)

            # Recent Data
            st.subheader( "Recent Data")

            st.dataframe(history.tail(10))

        except Exception as e:

            st.error( f"Error: {e}")

    render_footer()
    