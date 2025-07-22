# src/config.py
import os

# Reverted to a local SQLite database file
DATABASE_URL = "sqlite:///financial_data.db"

RISK_FREE_RATE = 0.02

# --- Manually Curated Ticker Lists for Stability ---
US_TICKERS = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc. (Class A)",
    "AMZN": "Amazon.com, Inc.", "NVDA": "NVIDIA Corp.", "TSLA": "Tesla, Inc.",
    "JPM": "JPMorgan Chase & Co.", "V": "Visa Inc.", "JNJ": "Johnson & Johnson",
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ Trust", "GLD": "SPDR Gold Shares"
}

INDIAN_TICKERS = {
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services", "HDFCBANK": "HDFC Bank",
    "INFY": "Infosys", "ICICIBANK": "ICICI Bank", "HINDUNILVR": "Hindustan Unilever",
    "ITC": "ITC Limited", "SBIN": "State Bank of India",
    "NIFTYBEES": "Nippon India ETF Nifty 50 BeES", "BANKBEES": "Nippon India ETF Bank BeES"
}

INDIAN_STOCK_SYMBOLS = set(INDIAN_TICKERS.keys())

INDEX_TICKERS_MAP = {
    "^GSPC": "S&P 500", "^IXIC": "NASDAQ Composite", "^DJI": "Dow Jones Industrial",
    "^NSEI": "NIFTY 50", "^BSESN": "SENSEX", "^NSEBANK": "NIFTY BANK"
}