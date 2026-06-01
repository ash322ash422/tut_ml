import streamlit as st

def render_sidebar():
    st.sidebar.title("📈 AI Stock Insight")

    selected_page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Stock Analysis",
            "AI Insights"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("Version 1.0")

    return selected_page


def render_header(title, description):
    st.title(title)
    st.caption(description)
    st.markdown("---")
    

def render_footer():
    st.markdown("---")
    st.caption("AI Stock Insight App | Demo Version")