import yfinance as yf

class NewsService:
    @staticmethod
    def get_news(ticker):
        stock = yf.Ticker(ticker)
        try:
            news = stock.news
        except Exception:
            return []

        articles = []
        for item in news[:10]:
            content = item.get("content", {})
            title = content.get("title", "")
            
            # This is your current extraction logic
            provider_info = content.get("provider", {})
            publisher = provider_info.get("displayName", "")

            articles.append(
                {
                    "title": title,
                    "publisher": publisher
                }
            )
        
        return articles

# For testing purposes, you can run this file directly to see the output of the get_news method.
if __name__ == "__main__":
    # Test with Apple Inc.
    news = NewsService.get_news("AAPL")
    print(news)