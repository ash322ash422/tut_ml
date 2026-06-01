import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import streamlit as st
from components.layout import (
    render_header,
    render_footer
)

from backend.services.stock_service import (
    StockService
)

from backend.services.news_service import (
    NewsService
)


def show():

    render_header(
        "🏠 Dashboard",
        "Overview of stock analytics platform"
    )

    # DEFAULT STOCK
    default_ticker = "AAPL"

    try:
        info = StockService.get_stock_info( default_ticker)
        current_price = info.get( "currentPrice", "N/A")

    except Exception:
        current_price = "N/A"

    # TOP METRICS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric( "Current Stock", default_ticker)

    with col2:
        st.metric("Current Price", current_price)

    with col3:
        st.metric( "Prediction", "Bullish")

    # WATCHLIST
    st.subheader("📈 Watchlist")

    watchlist = [
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "GOOG"
    ]

    watchlist_data = []

    for ticker in watchlist:

        try:
            stock_info = StockService.get_stock_info(ticker)
        
            watchlist_data.append(
                {
                  "Ticker": ticker,
                  "Price": stock_info.get("currentPrice","N/A")
                }
            )

        except Exception:
            watchlist_data.append(
                {
                  "Ticker": ticker,
                  "Price": "N/A"
                }
            )

    st.dataframe( watchlist_data,use_container_width=True)

    # PRICE CHART
    st.subheader("📊 AAPL Price Overview" )

    try:
        history = StockService.get_stock_data(
                default_ticker,
                period="6mo"
            )
        
        st.line_chart( history["Close"])

    except Exception:
        st.warning( "Unable to load chart.")

    # NEWS
    
    st.subheader( "📰 Latest Headlines" )

    try:
        news = NewsService.get_news(  default_ticker)

        for article in news[:5]:
            st.markdown(f"• {article['title']}")

    except Exception:
        st.warning("Unable to load news." )

    render_footer()
    
    