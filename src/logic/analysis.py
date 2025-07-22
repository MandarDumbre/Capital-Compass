# src/logic/analysis.py
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import re

from groq import Groq
from .metrics import calculate_advanced_metrics, clean_ticker_name, calculate_cagr
from .. import config

@st.cache_data(ttl="1h")
def get_usd_inr_rate():
    try:
        rate = yf.Ticker("INR=X").history(period="1d")['Close'].iloc[-1]
        return rate
    except Exception:
        st.warning("Could not fetch live exchange rate. Using a fallback value of 83.50.", icon="⚠️")
        return 83.50

@st.cache_data(ttl="6h")
def get_financial_metrics(ticker):
    """
    Fetches key financial metrics for a single ticker.
    """
    stock = yf.Ticker(ticker)
    try:
        financials = stock.financials
        if financials.empty:
            return {}
        financials = financials.transpose()
        financials.index = pd.to_datetime(financials.index)
        required_cols = ['Total Revenue', 'Net Income']
        if not all(col in financials.columns for col in required_cols):
            return {}
        financials = financials.sort_index().dropna(subset=required_cols)
        if len(financials) < 2: return {}

        latest, previous = financials.iloc[-1], financials.iloc[-2]
        rev_yoy = (latest['Total Revenue'] / previous['Total Revenue']) - 1 if previous['Total Revenue'] > 0 else None
        profit_yoy = (latest['Net Income'] / previous['Net Income']) - 1 if previous['Net Income'] > 0 else None

        rev_cagr_3y = calculate_cagr(financials['Total Revenue'].iloc[-1], financials['Total Revenue'].iloc[-4], 3) if len(financials) >= 4 else None
        profit_cagr_3y = calculate_cagr(financials['Net Income'].iloc[-1], financials['Net Income'].iloc[-4], 3) if len(financials) >= 4 else None

        rev_cagr_10y = calculate_cagr(financials['Total Revenue'].iloc[-1], financials['Total Revenue'].iloc[-11], 10) if len(financials) >= 11 else None
        profit_cagr_10y = calculate_cagr(financials['Net Income'].iloc[-1], financials['Net Income'].iloc[-11], 10) if len(financials) >= 11 else None

        return {
            "Revenue Growth (YoY)": rev_yoy, "Revenue CAGR (3Y)": rev_cagr_3y, "Revenue CAGR (10Y)": rev_cagr_10y,
            "Profit Growth (YoY)": profit_yoy, "Profit CAGR (3Y)": profit_cagr_3y, "Profit CAGR (10Y)": profit_cagr_10y,
        }
    except Exception:
        return {}

@st.cache_data(ttl="1h")
def get_benchmark_performance(index_tickers: tuple, _data_tier, usd_inr_rate: float):
    index_prices = _data_tier.get_price_history_for_tickers(index_tickers)
    if index_prices.empty: return {}

    benchmark_results = {}
    indian_indices = {"^NSEI", "^BSESN", "^NSEBANK"}

    for ticker in index_prices.columns:
        if not isinstance(ticker, str) or not re.match(r"^[\^A-Z0-9\.-]+$", ticker):
            st.warning(f"Invalid ticker format: {ticker}")
            continue

        price_series = index_prices[ticker].dropna()
        if price_series.empty: continue

        price_series_usd = price_series / usd_inr_rate if ticker in indian_indices else price_series
        daily_returns = price_series_usd.pct_change().dropna()
        if daily_returns.empty: continue

        mu = daily_returns.mean() * 252
        sigma = daily_returns.std() * np.sqrt(252)
        sharpe = (mu - config.RISK_FREE_RATE) / sigma if sigma > 0 else 0

        advanced_metrics = calculate_advanced_metrics(daily_returns, config.RISK_FREE_RATE)

        benchmark_results[ticker] = {
            "name": config.INDEX_TICKERS_MAP.get(ticker, ticker),
            "performance": [mu, sigma, sharpe],
            "advanced_metrics": advanced_metrics
        }
    return benchmark_results

def get_ai_comparison(client: Groq, all_results: dict):
    if not client: return "Groq client not initialized. AI features are disabled."
    prompt_data = ""
    for profile, data in all_results.items():
        if profile in ["Low", "Medium", "High"] and data:
            perf = data['performance']
            adv_metrics = data['advanced_metrics']
            top_holdings_clean = {clean_ticker_name(k): f'{v:.1%}' for k, v in sorted(data['weights'].items(), key=lambda item: item[1], reverse=True)[:3]}
            prompt_data += f"""
### {profile} Risk Profile
- **Objective**: {'Minimize Volatility' if profile == 'Low' else 'Maximize Sharpe Ratio' if profile == 'Medium' else 'Target High Return'}
- **Expected Annual Return**: {perf[0]:.2%}
- **Annual Volatility**: {perf[1]:.2%}
- **Sharpe Ratio**: {perf[2]:.2f}
- **Sortino Ratio**: {adv_metrics['sortino']:.2f}
- **Top Holdings**: {top_holdings_clean}
"""
    prompt = f"""
    You are a professional investment analyst. Compare three model portfolios based on the data provided. Use clear, simple terms and Markdown formatting.

    **Data:**
    {prompt_data}

    **Task:**
    1.  Start with a title: `### AI Portfolio Comparison`.
    2.  Briefly explain the risk-return trade-off using these portfolios as examples.
    3.  Describe the characteristics of each portfolio (Low, Medium, High).
    4.  Analyze how the asset allocation shifts across the profiles to achieve their objectives.
    5.  Briefly touch on why the Sortino Ratio might differ from the Sharpe Ratio for these portfolios.
    6.  Conclude with a standard disclaimer that this is not financial advice.
    """
    try:
        with st.spinner("🤖 Groq AI is comparing the portfolios..."):
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], model="llama3-8b-8192",
            )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the Groq API: {e}"