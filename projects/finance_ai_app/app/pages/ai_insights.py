import streamlit as st
from components.layout import render_header, render_footer
from components.news_cards import show_news_cards
from components.sentiment_card import  show_sentiment
from backend.services.news_service import NewsService
from backend.ai.sentiment_analysis import SentimentAnalysis

def show():
    render_header(
        "🧠 AI Insights",
        "News Analysis and Sentiment Intelligence"
    )

    ticker = st.text_input( "Ticker",value="AAPL" )

    if st.button( "Generate Insights"):
        news = NewsService.get_news( ticker )
        # st.write("DEBUG: " + str(news)) # DEBUG: Check news data structure
        
        if not news:
            st.error( "No news found."  )
            return

        st.subheader( "Latest News" )

        show_news_cards( news )

        news_text = "\n".join(
            [
                article["title"]
                for article in news
            ]
        )

        sentiment =  SentimentAnalysis.analyze_text(news_text)

        score = SentimentAnalysis.score_text(news_text)

        summary = SentimentAnalysis.generate_summary(news)
        
        st.subheader("Sentiment Analysis")

        show_sentiment(sentiment)

        st.metric("Sentiment Score",round(score, 3))

        st.subheader("Summary")

        st.info(summary )

        st.subheader( "News Used" )

        st.text_area("Headlines", news_text,height=250)

    render_footer()