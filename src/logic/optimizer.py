# src/logic/optimizer.py
import pandas as pd
import numpy as np
import streamlit as st
from pypfopt import expected_returns, risk_models, EfficientFrontier, DiscreteAllocation
from .metrics import calculate_advanced_metrics

def run_all_optimizations(price_history: pd.DataFrame, total_portfolio_value_usd: float, usd_inr_rate: float, budget_currency: str):
    """
    Runs portfolio optimization for Low, Medium, and High risk profiles.
    """
    prices = price_history.ffill().dropna()

    if prices.empty or len(prices.columns) < 2 or len(prices) < 2:
        st.error("Insufficient historical data for optimization. Ensure at least two assets have overlapping history.", icon="📊")
        return None

    prices_in_usd = prices.copy()
    for ticker in prices_in_usd.columns:
        if '.NS' in ticker:
            prices_in_usd[ticker] /= usd_inr_rate

    try:
        mu = expected_returns.mean_historical_return(prices_in_usd)
        S = risk_models.sample_cov(prices_in_usd)
    except Exception as e:
        st.error(f"Could not process price data for modeling: {e}", icon="⚙️")
        return None

    results = {}
    profiles = {"Low": "🛡️", "Medium": "⚖️", "High": "🚀"}

    for profile, icon in profiles.items():
        st.info(f"Calculating {profile} Risk portfolio... {icon}", icon=icon)
        ef = EfficientFrontier(mu, S)
        try:
            weights = None
            if profile == "Low":
                weights = ef.min_volatility()
            elif profile == "Medium":
                weights = ef.max_sharpe()
            elif profile == "High":
                try:
                    # Target a return slightly below the max possible to avoid infeasibility
                    weights = ef.efficient_return(target_return=mu.max() * 0.95)
                except ValueError:
                    st.warning("High-risk target was infeasible. Defaulting to Max Sharpe.", icon="⚠️")
                    weights = ef.max_sharpe()

            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False)
            portfolio_daily_returns = prices_in_usd.pct_change().dropna().dot(pd.Series(cleaned_weights))
            advanced_metrics = calculate_advanced_metrics(portfolio_daily_returns)
            latest_prices_usd = prices_in_usd.iloc[-1].dropna()
            da = DiscreteAllocation(cleaned_weights, latest_prices_usd, total_portfolio_value=total_portfolio_value_usd)
            allocation, leftover_usd = da.greedy_portfolio()

            # Convert leftover cash back to the user's selected currency
            leftover_final = leftover_usd * usd_inr_rate if budget_currency == 'INR' else leftover_usd

            results[profile] = {
                "weights": cleaned_weights, "performance": list(performance), "advanced_metrics": advanced_metrics,
                "discrete_allocation": allocation, "leftover_cash": leftover_final,
            }
        except Exception as e:
            st.warning(f"Could not calculate '{profile}' risk portfolio. Error: {e}", icon="⚠️")
            results[profile] = None

    return results

def run_monte_carlo_simulation(portfolio_performance, initial_value, forecast_years=10, num_simulations=500):
    """
    Runs a Monte Carlo simulation for a given portfolio's performance.
    """
    mu, sigma = portfolio_performance[0], portfolio_performance[1]
    num_trading_days, dt = 252, 1/252

    z = np.random.standard_normal(size=(num_trading_days * forecast_years, num_simulations))
    daily_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = initial_value
    for t in range(1, price_paths.shape[0]):
        price_paths[t] = price_paths[t-1] * daily_returns[t]

    date_index = pd.date_range(start=pd.to_datetime('today'), periods=price_paths.shape[0])
    return pd.DataFrame(price_paths, index=date_index)