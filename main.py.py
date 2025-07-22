# app.py
import os
import time
import streamlit as st
import re
import yfinance as yf
import pandas as pd
import asyncio
from sqlalchemy import create_engine
from groq import Groq
from dotenv import load_dotenv

# Import from our new structure
from src import config
from src.data.database import DataTier
from src.logic import metrics, optimizer, analysis
from src.ui import plots

st.set_page_config(page_title="Capital Compass", layout="wide", page_icon="🗭")

load_dotenv()

@st.cache_resource
def get_db_engine():
    return create_engine(config.DATABASE_URL)

@st.cache_resource
def get_groq_client():
    """Creates and caches the Groq client using environment variables."""
    try:
        # Reverted to use os.environ.get for Groq API Key
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key:
            return Groq(api_key=groq_api_key)
        st.warning("GROQ_API_KEY not found in environment. AI features will be disabled.", icon="⚠️")
        return None
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}. Please check your API key.", icon="🔥")
        return None
        
# (The rest of the app.py file remains the same as the last complete version you have)
# --- Asynchronous Data Fetching, UI, Processing Logic, and Results Display ---
async def fetch_and_store_data_async(data_tier, ticker, name_override=None):
    try:
        if not isinstance(ticker, str) or not re.match(r"^[\^A-Z0-9\.-]+$", ticker, re.IGNORECASE):
            return False, f"Invalid ticker format: '{ticker}'"

        stock = yf.Ticker(ticker)
        df = await asyncio.to_thread(stock.history, period="10y", auto_adjust=False)

        if df.empty:
            return False, f"Could not fetch data for {metrics.clean_ticker_name(ticker)}"

        try:
            stock_info = await asyncio.to_thread(getattr, stock, 'info')
            company_name = name_override or stock_info.get('longName', metrics.clean_ticker_name(ticker))
        except Exception:
            company_name = name_override or metrics.clean_ticker_name(ticker)

    except Exception as e:
        return False, f"Error fetching {metrics.clean_ticker_name(ticker)}: {e}"

    df = df.rename(columns={
        'Open': 'open_price', 'High': 'high_price', 'Low': 'low_price',
        'Close': 'close_price', 'Adj Close': 'adjusted_close_price', 'Volume': 'volume'
    }).drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
    df.index.name = 'date'
    df = df[['open_price', 'high_price', 'low_price', 'close_price', 'adjusted_close_price', 'volume']]
    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    data_tier.insert_securities([{'ticker': ticker, 'name': company_name}])
    data_tier.insert_daily_prices(ticker, df)
    return True, company_name

async def run_async_fetch(tickers_to_fetch, data_tier, progress_bar):
    tasks = []
    for ticker in tickers_to_fetch:
        name = None
        if ticker in config.INDEX_TICKERS_MAP: name = config.INDEX_TICKERS_MAP[ticker]
        elif ticker.replace(".NS", "") in config.INDIAN_TICKERS: name = config.INDIAN_TICKERS[ticker.replace(".NS", "")]
        elif ticker in config.US_TICKERS: name = config.US_TICKERS[ticker]
        tasks.append(fetch_and_store_data_async(data_tier, ticker, name_override=name))

    completed_count = 0
    for future in asyncio.as_completed(tasks):
        completed_count += 1
        success, resolved_name = await future
        if success:
            progress_text = f"Updating data for {resolved_name}..."
        else:
            progress_text = f"Failed: {resolved_name}"

        progress_bar.progress(completed_count / len(tasks), text=progress_text)

# --- Initialize Global Objects ---
engine = get_db_engine()
client = get_groq_client()
data_tier = DataTier(engine)

# --- BRANDING ---
st.title("Capital Compass 🧭")
st.markdown("##### *Navigating Your Investment Strategy*")
st.markdown("---")

# --- UI CONTROLS ---
with st.container(border=True):
    st.header("⚙️ Portfolio Configuration")
    with st.form("portfolio_form"):
        st.subheader("Step 1: Select Your Assets")
        col1, col2 = st.columns(2)
        with col1:
            us_stock_options = list(config.US_TICKERS.keys())
            desired_us_defaults = ["SPY", "AAPL", "NVDA", "JPM", "TSLA"]
            valid_us_defaults = [ticker for ticker in desired_us_defaults if ticker in us_stock_options]
            selected_us_stocks = st.multiselect("🇺🇸 Select US Stocks", options=us_stock_options, format_func=lambda x: f"{x} ({config.US_TICKERS.get(x, 'N/A')})", default=valid_us_defaults)
        with col2:
            in_stock_options = list(config.INDIAN_TICKERS.keys())
            desired_in_defaults = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "NIFTYBEES"]
            valid_in_defaults = [ticker for ticker in desired_in_defaults if ticker in in_stock_options]
            selected_in_stocks = st.multiselect("🇮🇳 Select Indian Stocks", options=in_stock_options, format_func=lambda x: f"{x} ({config.INDIAN_TICKERS.get(x, 'N/A')})", default=valid_in_defaults)

        all_selected_tickers = selected_us_stocks + selected_in_stocks

        st.subheader("Step 2: Set Your Budget")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            currency_select = st.selectbox("Select Budget Currency", ("₹ INR", "$ USD"))
            currency_symbol, currency_code = currency_select.split(" ")
        with b_col2:
            default_val = 100000 if currency_code == "INR" else 1000
            step_val = 10000 if currency_code == "INR" else 100
            total_value = st.number_input(f"Portfolio Value ({currency_symbol})", min_value=1000, value=default_val, step=step_val)

        submitted = st.form_submit_button("🚀 Generate & Compare Portfolios", type="primary", use_container_width=True, disabled=len(all_selected_tickers) < 2)
        if len(all_selected_tickers) < 2:
            st.caption(":warning: Please select at least two assets.")

# --- PROCESSING LOGIC ---
if submitted:
    if 'results' in st.session_state:
        del st.session_state.results

    progress_bar = st.progress(0, text="Initializing parallel data fetch...")

    backend_tickers = [f"{t}.NS" if t in config.INDIAN_STOCK_SYMBOLS else t for t in all_selected_tickers]
    all_tickers_to_fetch = backend_tickers + list(config.INDEX_TICKERS_MAP.keys())

    asyncio.run(run_async_fetch(all_tickers_to_fetch, data_tier, progress_bar))

    progress_bar.progress(1.0, text="Data fetch complete! Running optimizations...")

    with st.spinner("Running optimizations and simulations..."):
        st.cache_data.clear()
        price_history = data_tier.get_price_history_for_tickers(tuple(sorted(backend_tickers)))

        if price_history.empty or len(price_history.columns) < 2:
            st.error("Could not fetch data for enough assets. Please check warnings and try again.", icon="❌")
            st.session_state.results = None
        else:
            usd_inr_rate = analysis.get_usd_inr_rate()
            total_value_usd = total_value / usd_inr_rate if currency_code == 'INR' else total_value

            results = optimizer.run_all_optimizations(price_history, total_value_usd, usd_inr_rate, currency_code)

            if results:
                st.session_state.results = results
                st.session_state.results["price_history"] = price_history
                st.session_state.benchmark_results = analysis.get_benchmark_performance(tuple(config.INDEX_TICKERS_MAP.keys()), data_tier, usd_inr_rate)
                st.session_state.ai_narrative = analysis.get_ai_comparison(client, results)
                st.session_state.usd_inr_rate = usd_inr_rate
                st.session_state.currency_symbol = currency_symbol
                st.session_state.total_value = total_value
            else:
                st.session_state.results = None

    progress_bar.empty()

# --- RESULTS DISPLAY ---
if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results

    st.info(f"📈 Using conversion rate: **1 USD = {st.session_state.usd_inr_rate:.2f} INR**", icon="💹")

    st.header("📊 Portfolio Comparison Results")
    cols = st.columns(3)
    profiles = ["Low", "Medium", "High"]
    icons = {"Low": "🛡️", "Medium": "⚖️", "High": "🚀"}

    for i, profile in enumerate(profiles):
        if results.get(profile):
            with cols[i]:
                st.subheader(f"{icons[profile]} {profile} Risk")
                perf = results[profile]['performance']
                st.metric("Expected Return", f"{perf[0]:.2%}", help="The anticipated yearly return based on historical data.")
                st.metric("Volatility (Risk)", f"{perf[1]:.2%}", help="The annualized standard deviation; a measure of how much the portfolio's returns fluctuate (higher means riskier).")
                st.metric("Sharpe Ratio", f"{perf[2]:.2f}", help="Measures return per unit of total risk. A higher value (typically > 1) is better.")

                alloc_df = pd.DataFrame(results[profile]['discrete_allocation'].items(), columns=['Ticker', 'Shares'])
                alloc_df['Ticker'] = alloc_df['Ticker'].apply(metrics.clean_ticker_name)
                st.write("**Ideal Allocation**")
                st.dataframe(alloc_df, use_container_width=True, hide_index=True)
                st.markdown(f"**Leftover Cash:** `{st.session_state.currency_symbol}{results[profile]['leftover_cash']:.2f}`")

    st.markdown("---")
    st.header("Visualizations & Analysis")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📜 Price History", "🧭 Efficient Frontier", "🆚 Index Benchmark", "📈 Asset Details", "🔮 Monte Carlo", "🤖 AI Insights"])

    with tab1:
        st.markdown("**Insight**: This chart shows the **relative performance** of your chosen assets over the last 10 years, with every asset's starting price normalized to 100.")
        plots.plot_normalized_history(results['price_history'])

    with tab2:
        st.markdown("**Insight**: The Efficient Frontier shows all *optimal* portfolios offering the highest return for a given level of risk. Your portfolios are marked on this frontier.")
        prices_usd = results['price_history'].copy()
        for ticker in prices_usd.columns:
            if '.NS' in ticker: prices_usd[ticker] /= st.session_state.usd_inr_rate
        plots.plot_efficient_frontier(prices_usd, results)

    with tab3:
        st.subheader("Comparison with Market Indices")
        st.markdown("**Insight**: This table compares key performance metrics of your portfolios against prominent indices. All figures are annualized and converted to USD for a fair comparison.")
        if 'benchmark_results' in st.session_state and st.session_state.benchmark_results:
            data = {}
            for profile, p_data in results.items():
                if profile in profiles and p_data:
                    all_metrics = p_data['performance'] + list(p_data['advanced_metrics'].values())
                    data[f"{icons[profile]} {profile} Risk"] = all_metrics
            for b_ticker, b_data in st.session_state.benchmark_results.items():
                all_metrics = b_data['performance'] + list(b_data['advanced_metrics'].values())
                data[b_data['name']] = all_metrics

            summary_df = pd.DataFrame(data, index=["Return", "Volatility", "Sharpe", "Sortino", "Calmar", "Max Drawdown", "VaR (95%)"]).T
            st.dataframe(summary_df.style.format("{:.2%}", subset=["Return", "Volatility", "Max Drawdown", "VaR (95%)"]).format("{:.2f}", subset=["Sharpe", "Sortino", "Calmar"]), use_container_width=True)

            with st.expander("What do these metrics mean?"):
                st.markdown("""
                - **Return**: The expected annualized growth rate.
                - **Volatility**: A measure of price fluctuation (risk).
                - **Sharpe Ratio**: Measures risk-adjusted return. Higher is better.
                - **Sortino Ratio**: Similar to Sharpe, but only penalizes for downside risk.
                - **Calmar Ratio**: Return relative to the maximum drawdown.
                - **Max Drawdown**: The largest peak-to-trough drop.
                - **VaR (95%)**: Value at Risk; estimated max daily loss (95% confidence).
                """)

    with tab4:
        st.subheader("Understanding the Components")
        st.markdown("**Insight**: This section provides a look at each asset's **10-year price history and fundamental growth metrics**.")
        price_history_df = results['price_history']
        selected_tickers_backend = price_history_df.columns.tolist()
        for i in range(0, len(selected_tickers_backend), 2):
            plot_cols = st.columns(2)
            for j in range(2):
                if i + j < len(selected_tickers_backend):
                    ticker = selected_tickers_backend[i+j]
                    with plot_cols[j]:
                        st.subheader(f"Analysis for {metrics.clean_ticker_name(ticker)}")
                        end_date = price_history_df.index.max()
                        start_date = end_date - pd.DateOffset(years=10)
                        recent_price_history = price_history_df[price_history_df.index >= start_date]

                        fig_df = recent_price_history[[ticker]].rename(columns={ticker: metrics.clean_ticker_name(ticker)})
                        if not fig_df.dropna().empty:
                            fig = plots.px.line(fig_df, title=f"Price History for {metrics.clean_ticker_name(ticker)}")
                            fig.update_layout(showlegend=False, yaxis_title="Price", xaxis_title="Date", title_x=0.5)
                            st.plotly_chart(fig, use_container_width=True)

                        with st.spinner(f"Fetching metrics for {metrics.clean_ticker_name(ticker)}..."):
                            fin_metrics = analysis.get_financial_metrics(ticker)
                        if fin_metrics:
                            metrics_df = pd.DataFrame({"Metric": list(fin_metrics.keys()), "Value": [f"{v:.2%}" if v is not None and isinstance(v, (int, float)) else "N/A" for v in fin_metrics.values()]})
                            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                        else: st.write("Financial metrics not available.")

    with tab5:
        st.subheader("Forecasting Future Possibilities")
        st.markdown("**Insight**: A Monte Carlo simulation projects future portfolio performance by running thousands of random trials based on historical performance.")
        for profile in profiles:
            if results.get(profile):
                st.subheader(f"Forecast for {profile} Risk Portfolio")
                with st.spinner(f"Running simulation for {profile} risk..."):
                    sim_df = optimizer.run_monte_carlo_simulation(
                        portfolio_performance=results[profile]['performance'],
                        initial_value=st.session_state.total_value
                    )
                plots.plot_monte_carlo(sim_df, f"10-Year Projection: {profile} Risk", st.session_state.currency_symbol)

                final_values = sim_df.iloc[-1]
                metric_cols = st.columns(2)
                median_value = f"{st.session_state.currency_symbol}{final_values.median():,.0f}"
                range_value = f"{st.session_state.currency_symbol}{final_values.quantile(0.1):,.0f} - {st.session_state.currency_symbol}{final_values.quantile(0.9):,.0f}"

                with metric_cols[0]:
                    st.metric("Median Projected Value (10Y)", median_value)
                with metric_cols[1]:
                    st.metric("Potential Range (10th-90th)", range_value)

    with tab6:
        st.subheader("Qualitative Summary")
        st.markdown("**Insight**: This analysis is generated by a Large Language Model (LLM) to provide a human-readable interpretation of the quantitative data.")
        if 'ai_narrative' in st.session_state:
            st.markdown(st.session_state.ai_narrative)

# --- FOOTER & ADMIN ---
with st.expander("Admin & Database Management"):
    if st.button("Initialize/Reset Local Database", key="reset_db"):
        with st.spinner("Resetting database..."):
            data_tier.create_schema()
            st.cache_data.clear()
            st.cache_resource.clear()
        st.success("Database has been reset.")
        st.rerun()

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Built with ❤️ Capital Compass</div>", unsafe_allow_html=True)