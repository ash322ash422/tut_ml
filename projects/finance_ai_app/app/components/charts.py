import plotly.graph_objects as go
import streamlit as st

def show_price_chart(df, ticker):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter( x=df.index, y=df["Close"], name="Close" )
    )

    fig.add_trace(
        go.Scatter( x=df.index, y=df["MA20"],  name="MA20" )
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA50"], name="MA50" )
    )

    fig.update_layout(title=f"{ticker} Price Trend" )

    st.plotly_chart(fig, use_container_width=True)