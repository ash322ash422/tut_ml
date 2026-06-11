# app.py

import streamlit as st
import pandas as pd

from data import get_dashboard_data as get_dashboard_data
# from data import get_dashboard_data_test as get_dashboard_data # Use this for testing without API calls

# =====================================================
# SAFE DISPLAY FUNCTIONS
# =====================================================

def safe_text(value):
    """
    Return value as text or N/A
    """
    if value is None or value == "":
        return "N/A"
    return str(value)


def safe_price(value):
    """
    Format currency safely
    """
    try:
        return f"${float(value):,.2f}"
    except:
        return "N/A"


def safe_market_cap(value):
    """
    Convert market cap to trillions safely
    """
    try:
        return f"${float(value)/1e12:.2f} T"
    except:
        return "N/A"




# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size:40px;
        font-weight:bold;
        text-align:center;
        color:#00BFFF;
        margin-bottom:20px;
    }

    .sub-title {
        text-align:center;
        color:gray;
        margin-bottom:30px;
    }

    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #444;
        padding: 15px;
        border-radius: 12px;
        text-align:center;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<div class='main-title'>📈 Financial Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Powered by Alpha Vantage</div>",
    unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Configuration")

api_key = st.sidebar.text_input(
    "Alpha Vantage API Key",
    type="password"
)

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="AAPL"
)

load_button = st.sidebar.button(
    "Load Data"
)


# =====================================================
# MAIN
# =====================================================

if load_button:

    if not api_key:

        st.warning("Please enter an API key.")
        st.stop()

    with st.spinner("Fetching data..."):

        try:

            data = get_dashboard_data(
                api_key=api_key,
                ticker=ticker.upper()
            )

            st.success("Data Loaded Successfully")

            # ======================================
            # ROW 1
            # ======================================

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "💲 Current Price",
                    safe_price(data.get("current_price"))
                )

            with col2:
                st.metric(
                    "📊 P/E Ratio",
                    safe_text(data.get("pe_ratio"))
                )

            with col3:
                st.metric(
                    "💰 EPS",
                    safe_text(data.get("eps"))
                )

            st.markdown("---")

            # ======================================
            # ROW 2
            # ======================================

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🏦 Market Cap",
                    safe_market_cap(data.get("market_cap"))
                )
                            
            
            with col2:
                st.metric(
                    "📈 52 Week High",
                    safe_text(data.get("52_week_high"))
                )

            with col3:
                st.metric(
                    "📉 52 Week Low",
                    safe_text(data.get("52_week_low"))
                )

            st.markdown("---")

            # ======================================
            # ROW 3
            # ======================================

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🎁 Dividend Yield",
                    safe_text(data.get("dividend_yield"))
                )

            with col2:
                st.metric(
                    "🇮🇳 USD → INR",
                    safe_text(data.get("usd_inr"))
                )

            with col3:
                st.metric(
                    "🇯🇴 USD → JOD",
                    safe_text(data.get("usd_jod"))
                )

            st.markdown("---")

            # ======================================
            # SUMMARY TABLE
            # ======================================

            st.subheader("Company Snapshot")

            summary_df = pd.DataFrame(
                {
                    "Metric": [
                        "Ticker",
                        "Current Price",
                        "Market Cap",
                        "PE Ratio",
                        "EPS",
                        "Dividend Yield",
                        "52 Week High",
                        "52 Week Low",
                        "USD -> INR",
                        "USD -> JOD",
                    ],
                    "Value": [
                        data["ticker"],
                        data["current_price"],
                        data["market_cap"],
                        data["pe_ratio"],
                        data["eps"],
                        data["dividend_yield"],
                        data["52_week_high"],
                        data["52_week_low"],
                        data["usd_inr"],
                        data["usd_jod"],
                    ]
                }
            )

            st.dataframe(
                summary_df,
                use_container_width=True
            )

            # ======================================
            # DASHBOARD FOOTER
            # ======================================

            st.markdown("---")

            st.info(
                f"""
                Ticker: {ticker.upper()}

                Dashboard generated using Alpha Vantage APIs.

                Demonstrates:
                - Financial Fundamentals
                - Market Data
                - Currency Exchange Rates
                - Streamlit KPI Cards
                """
            )

        except Exception as e:

            st.error(f"Error: {e}")

else:

    st.info(
        "Enter your Alpha Vantage API key and a ticker symbol in the sidebar, then click 'Load Data'."
    )