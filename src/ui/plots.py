# src/ui/plots.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models
from ..logic.metrics import clean_ticker_name

def plot_normalized_history(price_df: pd.DataFrame):
    """Plots the 10-year normalized price history of selected assets."""
    df_cleaned = price_df.rename(columns=lambda c: clean_ticker_name(c))
    end_date = df_cleaned.index.max()
    start_date = end_date - pd.DateOffset(years=10)
    recent_prices = df_cleaned[df_cleaned.index >= start_date].dropna(how='all')

    if recent_prices.empty or len(recent_prices) < 2:
        st.warning("Not enough data in the last 10 years to plot normalized history.")
        return

    # Process the DataFrame once to fill and drop NaNs
    processed_prices = recent_prices.ffill().dropna()
    if processed_prices.empty:
        return
    normalized_df = processed_prices / processed_prices.iloc[0] * 100
    fig = px.line(normalized_df)
    fig.update_layout(
        title_text="Normalized Price History (Last 10 Years)",
        xaxis_title="Date", yaxis_title="Normalized Price (Starts at 100)",
        legend_title_text='Ticker'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_efficient_frontier(price_history_usd, results):
    """Plots the efficient frontier and the optimized portfolios."""
    mu = expected_returns.mean_historical_return(price_history_usd)
    S = risk_models.sample_cov(price_history_usd)

    fig = go.Figure()
    n_samples = 500
    portfolios = np.zeros((3, n_samples))
    for i in range(n_samples):
        weights = np.random.random(len(mu)); weights /= np.sum(weights)
        p_return = np.sum(weights * mu)
        p_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
        p_sharpe = (p_return - 0.02) / p_volatility
        portfolios[:, i] = [p_return, p_volatility, p_sharpe]

    fig.add_trace(go.Scatter(
        x=portfolios[1,:], y=portfolios[0,:], mode='markers',
        marker=dict(size=7, color=portfolios[2,:], colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Sharpe Ratio")),
        text=[f"Sharpe: {s:.2f}" for s in portfolios[2,:]], name="Random Portfolios"
    ))

    markers = {"Low": 'diamond', "Medium": 'star', "High": 'triangle-up'}
    colors = {"Low": 'green', "Medium": 'orange', "High": 'red'}
    names = {"Low": 'Min Volatility', "Medium": 'Max Sharpe Ratio', "High": 'High Return'}
    for profile, marker in markers.items():
        if results.get(profile):
            perf = results[profile]['performance']
            fig.add_trace(go.Scatter(x=[perf[1]], y=[perf[0]], mode='markers',
                                     marker=dict(symbol=marker, color=colors[profile], size=15),
                                     name=names[profile]))

    fig.update_layout(title="Efficient Frontier Simulation",
                      xaxis_title="Annual Volatility (Risk)", yaxis_title="Expected Annual Return",
                      yaxis_tickformat=".0%", xaxis_tickformat=".0%",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

def plot_monte_carlo(sim_df: pd.DataFrame, title: str, currency_symbol: str):
    """Plots the results of a Monte Carlo simulation."""
    fig = go.Figure()
    for i in range(min(sim_df.shape[1], 100)):
        fig.add_trace(go.Scatter(x=sim_df.index, y=sim_df.iloc[:, i], line=dict(width=0.5, color='grey'), showlegend=False))

    p10, p50, p90 = sim_df.quantile(0.10, axis=1), sim_df.quantile(0.50, axis=1), sim_df.quantile(0.90, axis=1)
    fig.add_trace(go.Scatter(x=p10.index, y=p10, line=dict(color='red', width=2, dash='dash'), name='10th Percentile'))
    fig.add_trace(go.Scatter(x=p50.index, y=p50, line=dict(color='blue', width=3), name='Median Outcome'))
    fig.add_trace(go.Scatter(x=p90.index, y=p90, line=dict(color='green', width=2, dash='dash'), name='90th Percentile'))

    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Portfolio Value",
        legend_title="Outcome", yaxis_tickprefix=currency_symbol, yaxis_tickformat=',.0f'
    )
    st.plotly_chart(fig, use_container_width=True)