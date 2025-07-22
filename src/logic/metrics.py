# src/logic/metrics.py
import pandas as pd
import numpy as np
from .. import config

def clean_ticker_name(ticker: str) -> str:
    """Removes .NS suffix for display purposes."""
    return ticker.replace(".NS", "")

def calculate_cagr(end_value, start_value, years):
    """Calculates Compound Annual Growth Rate."""
    if pd.isna(end_value) or pd.isna(start_value) or start_value <= 0 or years <= 0:
        return None
    return ((end_value / start_value) ** (1 / years)) - 1

def calculate_advanced_metrics(daily_returns: pd.Series, risk_free_rate=config.RISK_FREE_RATE):
    """Calculates Sortino, Calmar, Max Drawdown, and VaR for a portfolio."""
    if daily_returns.empty:
        return {"sortino": 0, "calmar": 0, "max_drawdown": 0, "var_95": 0}

    # Annualized mean return
    mean_return = daily_returns.mean() * 252

    # Sortino Ratio
    target_return = 0
    downside_returns = daily_returns[daily_returns < target_return]
    downside_std = downside_returns.std()
    sortino_ratio = (mean_return - risk_free_rate) / (downside_std * np.sqrt(252)) if downside_std != 0 else 0

    # Max Drawdown and Calmar Ratio
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Value at Risk (VaR)
    var_95 = daily_returns.quantile(0.05)

    return {
        "sortino": sortino_ratio,
        "calmar": calmar_ratio,
        "max_drawdown": max_drawdown,
        "var_95": var_95
    }