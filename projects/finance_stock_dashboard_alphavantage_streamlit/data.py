# data.py

import requests

def get_dashboard_data(api_key: str, ticker: str) -> dict:
    """
    Returns all dashboard data as a dictionary.
    """

    result = {}

    # ==========================================
    # COMPANY OVERVIEW
    # ==========================================

    overview_url = (
        "https://www.alphavantage.co/query"
        f"?function=OVERVIEW"
        f"&symbol={ticker}"
        f"&apikey={api_key}"
    )

    overview = requests.get(overview_url).json()

    # ==========================================
    # GLOBAL QUOTE
    # ==========================================

    quote_url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={ticker}"
        f"&apikey={api_key}"
    )

    quote = requests.get(quote_url).json()

    quote_data = quote.get("Global Quote", {})

    # ==========================================
    # USD -> INR
    # ==========================================

    inr_url = (
        "https://www.alphavantage.co/query"
        "?function=CURRENCY_EXCHANGE_RATE"
        "&from_currency=USD"
        "&to_currency=INR"
        f"&apikey={api_key}"
    )

    inr_data = requests.get(inr_url).json()

    # ==========================================
    # USD -> JOD
    # ==========================================

    jod_url = (
        "https://www.alphavantage.co/query"
        "?function=CURRENCY_EXCHANGE_RATE"
        "&from_currency=USD"
        "&to_currency=JOD"
        f"&apikey={api_key}"
    )

    jod_data = requests.get(jod_url).json()

    # ==========================================
    # BUILD DICTIONARY
    # ==========================================

    result = {
        "ticker": ticker,

        "current_price":
            quote_data.get("05. price"),

        "market_cap":
            overview.get("MarketCapitalization"),

        "pe_ratio":
            overview.get("PERatio"),

        "eps":
            overview.get("EPS"),

        "dividend_yield":
            overview.get("DividendYield"),

        "52_week_high":
            overview.get("52WeekHigh"),

        "52_week_low":
            overview.get("52WeekLow"),

        "usd_inr":
                inr_data.get(
                    "Realtime Currency Exchange Rate",
                    {}
                ).get("5. Exchange Rate"),

        "usd_jod":
                jod_data.get(
                    "Realtime Currency Exchange Rate",
                    {}
                ).get("5. Exchange Rate"),
    }

    return result

def get_dashboard_data_test(api_key: str, ticker: str) -> dict:
    data = {'ticker': 'AAPL',
            'current_price': None,
            'market_cap': '4629454651000',
            'pe_ratio': '38.16',
            'eps': '8.26',
            'dividend_yield': '0.0034',
            '52_week_high': '315.45',
            '52_week_low': '194.3',
            'usd_inr': '95.69577565',
            'usd_jod': None
    }
    
    return data
    
    
if __name__ == "__main__":
    import os

    # api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    api_key = ""
    ticker = "AAPL"
    data = get_dashboard_data(api_key, ticker)
    print(data)
    
""" # sample output
{'ticker': 'AAPL', 'current_price': None, 'market_cap': '4629454651000', 'pe_ratio': '38.16', 'eps': '8.26', 'dividend_yield': '0.0034', '52_week_high': '315.45', '52_week_low': '194.3', 'usd_inr': '95.69577565', 'usd_jod': None}
"""