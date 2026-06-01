import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class PredictionService:

    @staticmethod
    def calculate_moving_averages(df):
        df = df.copy()
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()

        return df
    
    @staticmethod
    def generate_trend_signal(df):
        latest = df.dropna().iloc[-1]

        if latest["MA20"] > latest["MA50"]:
            return "Bullish"

        return "Bearish"

    @staticmethod
    def forecast_next_day(df, window=10):

        closes = df["Close"].values

        if len(closes) < window + 1:
            raise ValueError(
                f"Need at least {window+1} days of data"
            )

        X = []
        y = []
        for i in range(len(closes) - window):
            X.append(closes[i:i+window])
            y.append(closes[i+window])

        X = np.array(X)
        y = np.array(y)

        model = LinearRegression()
        model.fit(X, y)

        last_window = closes[-window:]
        prediction = model.predict([last_window])

        return float(prediction[0])    
    
