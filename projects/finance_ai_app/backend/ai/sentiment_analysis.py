from textblob import TextBlob

class SentimentAnalysis:

    @staticmethod
    def analyze_text(text):
        polarity = TextBlob(text).sentiment.polarity

        if polarity > 0.1:
            return "Positive"

        elif polarity < -0.1:
            return "Negative"

        return "Neutral"

    @staticmethod
    def score_text(text):
        return TextBlob(text).sentiment.polarity
    
    @staticmethod
    def generate_summary(news):
        count = len(news)

        return (
            f"Analyzed {count} recent news articles. "
            f"Sentiment is determined from "
            f"headline polarity scores."
        )
        
        