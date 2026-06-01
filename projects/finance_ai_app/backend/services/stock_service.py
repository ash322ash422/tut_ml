import yfinance as yf

class StockService:

    @staticmethod
    def get_stock_data(ticker, period="1y"):
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)

        return history

    @staticmethod
    def get_stock_info(ticker):
        stock = yf.Ticker(ticker)
        
        return stock.info
    