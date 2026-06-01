import streamlit as st

from components.layout import render_sidebar

from pages import dashboard
from pages import stock_analysis
from pages import ai_insights


st.set_page_config(
    page_title="AI Stock Insight",
    page_icon="📈",
    layout="wide"
)

selected_page = render_sidebar()


if selected_page == "Dashboard":
    dashboard.show()

elif selected_page == "Stock Analysis":
    stock_analysis.show()

elif selected_page == "AI Insights":
    ai_insights.show()

