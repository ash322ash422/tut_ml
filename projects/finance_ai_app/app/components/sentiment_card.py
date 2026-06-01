import streamlit as st

def show_sentiment(sentiment):
    if sentiment == "Positive":
        st.success(f"Market Sentiment: {sentiment}")

    elif sentiment == "Negative":
        st.error(f"Market Sentiment: {sentiment}")

    else:
        st.warning( f"Market Sentiment: {sentiment}")