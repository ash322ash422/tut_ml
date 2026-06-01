import streamlit as st
from backend.ai.sentiment_analysis import  SentimentAnalysis

def show_news_cards(news):

    for article in news:

        sentiment = (
            SentimentAnalysis.analyze_text(
                article["title"]
            )
        )

        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"**{article['title']}**" )

            st.caption(article["publisher"] )

        with col2:
            if sentiment == "Positive":
                st.success(sentiment)

            elif sentiment == "Negative":
                st.error(sentiment)

            else:
                st.warning(sentiment)

        st.markdown("---")