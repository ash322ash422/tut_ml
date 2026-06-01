import streamlit as st

def show_stock_metrics(info):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Price",
            info.get("currentPrice", "N/A")
        )

    with col2:
        st.metric(
            "Previous Close",
            info.get("previousClose", "N/A")
        )

    with col3:
        st.metric(
            "Volume",
            info.get("volume", "N/A")
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "52 Week High",
            info.get("fiftyTwoWeekHigh", "N/A")
        )

    with col5:
        st.metric(
            "52 Week Low",
            info.get("fiftyTwoWeekLow", "N/A")
        )

    with col6:
        st.metric(
            "Market Cap",
            info.get("marketCap", "N/A")
        )