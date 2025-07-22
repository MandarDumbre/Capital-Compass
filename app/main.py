# main.py
#
# This script is the main entry point for the AI-Powered Financial Analysis Tool.
# It contains all application logic in a single file for simplicity and to
# ensure all components are synchronized.
#
# To Run This Application:
# 1. Make sure all packages from requirements.txt are installed.
# 2. Run this script from your terminal using:
#    python main.py
#
# 3. Follow the on-screen menu to interact with the application.
#    If you encounter schema errors, run Option 1 to reset the database.

import os
import sys
import time
import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from groq import Groq # Use Groq instead of OpenAI
from dotenv import load_dotenv
from pypfopt import expected_returns, risk_models, EfficientFrontier, DiscreteAllocation

# --- Configuration & Setup ---

# Load environment variables from .env file for secure API key management
load_dotenv()

# Use a local SQLite database file. This persists data across runs.
DATABASE_URL = "sqlite:///financial_data.db"
engine = create_engine(DATABASE_URL)

# Securely initialize the Groq client
try:
    # Note: Ensure GROQ_API_KEY is set in your .env file
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        client = Groq(api_key=groq_api_key)
    else:
        client = None
        print("Warning: GROQ_API_KEY not found in environment variables.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    print("Please ensure your GROQ_API_KEY is set correctly in the .env file.")
    client = None # Will be handled gracefully if key is missing

# --- Data Tier: Database Interaction ---

class DataTier:
    """
    Handles all direct interactions with the SQL database.
    Corresponds to the Data Tier in the 3-tier architecture.
    """
    def __init__(self, engine):
        self.engine = engine

    def create_schema(self):
        """
        Creates the database tables based on the defined schema.
        This version now DROPS existing tables to ensure a fresh schema,
        preventing errors from outdated database files.
        """
        print("Initializing database schema...")
        with self.engine.connect() as connection:
            with connection.begin(): # Use a transaction for DDL statements
                # Drop tables first to ensure a clean slate
                connection.execute(text("DROP TABLE IF EXISTS daily_prices"))
                connection.execute(text("DROP TABLE IF EXISTS securities"))
                
                print("Old tables dropped. Creating new schema...")
                
                connection.execute(text("""
                CREATE TABLE securities (
                    id INTEGER PRIMARY KEY,
                    ticker VARCHAR(20) UNIQUE NOT NULL,
                    name VARCHAR(255)
                )
                """))
                connection.execute(text("""
                CREATE TABLE daily_prices (
                    id INTEGER PRIMARY KEY,
                    security_id INTEGER,
                    date TEXT NOT NULL,
                    open_price DECIMAL(19, 4) NOT NULL,
                    high_price DECIMAL(19, 4) NOT NULL,
                    low_price DECIMAL(19, 4) NOT NULL,
                    close_price DECIMAL(19, 4) NOT NULL,
                    adjusted_close_price DECIMAL(19, 4) NOT NULL,
                    volume BIGINT NOT NULL,
                    UNIQUE(security_id, date),
                    FOREIGN KEY (security_id) REFERENCES securities (id)
                )
                """))
            print("Schema reset and initialized successfully.")

    def get_security_id(self, ticker):
        """Retrieves the primary key for a given ticker, or None if not found."""
        with self.engine.connect() as connection:
            result = connection.execute(
                text("SELECT id FROM securities WHERE ticker = :ticker"),
                {'ticker': ticker}
            ).fetchone()
            return result[0] if result else None

    def insert_securities(self, securities_data):
        """Inserts a list of securities into the database."""
        with self.engine.connect() as connection:
            with connection.begin():
                for security in securities_data:
                    if not self.get_security_id(security['ticker']):
                        connection.execute(
                            text("INSERT INTO securities (ticker, name) VALUES (:ticker, :name)"),
                            security
                        )

    def insert_daily_prices(self, ticker, prices_df):
        """
        Inserts a DataFrame of daily prices for a given ticker.
        Handles conflicts to avoid inserting duplicate data.
        """
        security_id = self.get_security_id(ticker)
        if not security_id:
            print(f"Warning: Security for ticker {ticker} not found. Skipping price insertion.")
            return

        prices_df['security_id'] = security_id
        
        with self.engine.connect() as connection:
            with connection.begin():
                prices_df.to_sql('temp_daily_prices', connection, if_exists='replace', index=False)
                
                insert_sql = """
                    INSERT OR IGNORE INTO daily_prices (security_id, date, open_price, high_price, low_price, close_price, adjusted_close_price, volume)
                    SELECT security_id, date, open_price, high_price, low_price, close_price, adjusted_close_price, volume 
                    FROM temp_daily_prices;
                """
                connection.execute(text(insert_sql))

    def get_price_history_for_tickers(self, tickers):
        """
        Retrieves historical adjusted close price data for a list of tickers 
        and pivots it into a format suitable for portfolio optimization.
        """
        if not tickers:
            return pd.DataFrame()

        # Create the placeholders and parameters properly for SQLAlchemy
        placeholders = ', '.join([f':ticker_{i}' for i in range(len(tickers))])
        params = {f'ticker_{i}': ticker for i, ticker in enumerate(tickers)}
        
        sql_query = f"""
            SELECT s.ticker, dp.date, dp.adjusted_close_price
            FROM daily_prices dp
            JOIN securities s ON dp.security_id = s.id
            WHERE s.ticker IN ({placeholders})
            ORDER BY dp.date
        """
        
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(sql_query), connection, params=params)
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()
            
        try:
            price_pivot = df.pivot(index='date', columns='ticker', values='adjusted_close_price')
            # Convert date index to datetime for better handling
            price_pivot.index = pd.to_datetime(price_pivot.index)
            return price_pivot
        except Exception as e:
            print(f"Error pivoting data: {e}")
            return pd.DataFrame()


# --- Application Tier: Business Logic ---

class ApplicationTier:
    """
    Contains the core business logic, financial modeling, and AI integration.
    """
    def __init__(self, data_tier):
        self.data_tier = data_tier

    def fetch_and_store_data(self, ticker):
        """
        ETL Process: Extracts full daily data from Yahoo Finance, transforms it,
        and loads it into the SQL database.
        """
        print(f"\n--- ETL Process for {ticker} ---")
        print(f"1. EXTRACT: Fetching data for {ticker} from Yahoo Finance...")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="max", auto_adjust=False)

            if df.empty:
                print(f"Could not fetch data for {ticker}. It may be an invalid ticker.")
                return False

        except Exception as e:
            print(f"An unexpected error occurred during data fetching for {ticker}: {e}")
            return False

        print("2. TRANSFORM: Cleaning and structuring data with pandas...")
        
        df = df.rename(columns={
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Adj Close': 'adjusted_close_price',
            'Volume': 'volume'
        })
        
        df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')
        df.index.name = 'date'
        
        columns_to_keep = ['open_price', 'high_price', 'low_price', 'close_price', 'adjusted_close_price', 'volume']
        df = df[columns_to_keep]
        for col in columns_to_keep:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with NaN values
        df = df.dropna()
        
        df = df.reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        print(f"3. LOAD: Storing data for {ticker} in the SQL database...")
        try:
            company_name = stock.info.get('longName', f"{ticker} Inc.")
        except:
            company_name = f"{ticker} Inc."
            
        self.data_tier.insert_securities([{'ticker': ticker, 'name': company_name}])
        self.data_tier.insert_daily_prices(ticker, df)
        print(f"ETL process for {ticker} completed successfully.")
        return True

    def run_portfolio_optimization(self, tickers, total_portfolio_value=10000):
        """
        Performs portfolio optimization and discrete allocation.
        """
        print("\n--- Running Portfolio Optimization ---")
        prices = self.data_tier.get_price_history_for_tickers(tickers)
        
        if prices.empty:
            print("Could not retrieve any price data from the database.")
            print("Please ensure you have fetched data for all tickers first.")
            return None
            
        # Check if we have data for all requested tickers
        missing_tickers = set(tickers) - set(prices.columns)
        if missing_tickers:
            print(f"Missing data for tickers: {missing_tickers}")
            print("Please fetch data for these tickers first.")
            return None
            
        # Remove any columns with all NaN values
        prices = prices.dropna(axis=1, how='all')
        
        # Forward fill and backward fill to handle missing values
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        
        # Drop any remaining rows with NaN values
        prices = prices.dropna()
        
        if prices.empty or len(prices) < 2:
            print("Insufficient price data for optimization (need at least 2 data points).")
            return None
            
        if len(prices.columns) < 2:
            print("Need at least 2 securities for portfolio optimization.")
            return None

        try:
            mu = expected_returns.mean_historical_return(prices)
            S = risk_models.sample_cov(prices)

            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False)
            
            latest_prices = prices.iloc[-1]
            da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
            allocation, leftover = da.greedy_portfolio()

            results = {
                "weights": cleaned_weights,
                "performance": {
                    "Expected annual return": performance[0],
                    "Annual volatility": performance[1],
                    "Sharpe Ratio": performance[2]
                },
                "discrete_allocation": allocation,
                "leftover_cash": leftover
            }
            print("Optimization complete.")
            return results
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            return None

    def get_ai_narrative(self, portfolio_results):
        """
        Uses the Groq API to generate a human-readable summary of the results.
        """
        if not client:
            return "Groq client not initialized. Please check your GROQ_API_KEY in the .env file."

        print("\n--- Generating AI-Powered Narrative via Groq ---")
        prompt = f"""
        You are a professional and cautious investment advisor.
        Your task is to explain a model portfolio to a client in simple, clear terms.
        The underlying data was sourced from Yahoo Finance.
        Do not give financial advice, but explain the characteristics of the portfolio.
        
        Here is the data for the optimized portfolio:
        - Objective: Maximize the Sharpe Ratio (best risk-adjusted return).
        - Optimal Portfolio Weights: {portfolio_results['weights']}
        - Expected Annual Return: {portfolio_results['performance']['Expected annual return']:.2%}
        - Expected Annual Volatility: {portfolio_results['performance']['Annual volatility']:.2%}
        - Sharpe Ratio: {portfolio_results['performance']['Sharpe Ratio']:.2f}
        - Suggested Allocation for a $10,000 Portfolio: {portfolio_results['discrete_allocation']}
        - Leftover Cash: ${portfolio_results['leftover_cash']:.2f}

        Based on this data, please generate a brief, easy-to-understand summary.
        - Start with a clear header: "Portfolio Analysis Summary".
        - Explain what the portfolio is designed to do (maximize risk-adjusted return).
        - List the top holdings based on the weights.
        - Briefly touch on the expected risk and return characteristics.
        - Explain the discrete allocation in simple terms (e.g., how many shares of each stock to buy).
        - End with a standard financial disclaimer about this not being financial advice and the importance of consulting a professional.
        """

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful investment analyst."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192", # Using a Llama3 model via Groq
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"An error occurred while contacting the Groq API: {e}"


# --- Presentation Tier: Command-Line Interface ---

class PresentationTier:
    """
    Handles user interaction via a simple command-line interface.
    """
    def __init__(self, application_tier):
        self.app = application_tier
        self.last_ai_call_time = 0

    def display_menu(self):
        """Prints the main menu and returns the user's choice."""
        print("\n===== AI Financial Analysis Tool =====")
        print("1. Initialize/Reset Database Schema")
        print("2. Fetch Stock Data from Yahoo Finance (ETL)")
        print("3. Run Portfolio Optimization")
        print("4. Exit")
        return input("Choose an option: ")

    def run(self):
        """The main loop for the command-line interface."""
        while True:
            try:
                choice = self.display_menu()
                if choice == '1':
                    self.app.data_tier.create_schema()
                elif choice == '2':
                    tickers_str = input("Enter stock tickers to fetch (comma-separated, e.g., AAPL,MSFT,GOOG): ")
                    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
                    if not tickers:
                        print("No valid tickers entered. Please try again.")
                        continue
                        
                    for i, ticker in enumerate(tickers):
                        success = self.app.fetch_and_store_data(ticker)
                        if not success:
                            print(f"Stopping fetch process due to error with ticker {ticker}.")
                            break
                        if len(tickers) > 1 and i < len(tickers) - 1:
                            print("\nWaiting 1 second before next request...")
                            time.sleep(1)
                elif choice == '3':
                    tickers_str = input("Enter tickers for portfolio (comma-separated, e.g., AAPL,MSFT,GOOG): ")
                    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
                    if not tickers or tickers == ['']:
                        print("No tickers entered. Please try again.")
                        continue
                    
                    results = self.app.run_portfolio_optimization(tickers)
                    if results:
                        print("\n--- Quantitative Results ---")
                        print("Optimal Weights:")
                        for ticker, weight in results['weights'].items():
                            if weight > 0.001:  # Only show significant weights
                                print(f"  {ticker}: {weight:.2%}")
                        
                        print("\nPortfolio Performance:")
                        for key, value in results['performance'].items():
                            if "Ratio" in key:
                                print(f"  {key}: {value:.2f}")
                            else:
                                print(f"  {key}: {value:.2%}")

                        print("\nSuggested Discrete Allocation (for $10,000 portfolio):")
                        for ticker, num_shares in results['discrete_allocation'].items():
                            print(f"  Buy {num_shares} shares of {ticker}")
                        print(f"  Leftover Cash: ${results['leftover_cash']:.2f}")
                        
                        # Simple rate limiting for the AI API call
                        time_since_last_call = time.time() - self.last_ai_call_time
                        if time_since_last_call < 2: # Wait at least 2 seconds between calls
                            wait_time = 2 - time_since_last_call
                            print(f"\nWaiting {wait_time:.1f} seconds to respect AI API rate limit...")
                            time.sleep(wait_time)

                        narrative = self.app.get_ai_narrative(results)
                        self.last_ai_call_time = time.time()

                        print("\n" + "="*50)
                        print(narrative)
                        print("="*50)
                elif choice == '4':
                    print("Exiting.")
                    break
                else:
                    print("Invalid option, please try again.")
            except KeyboardInterrupt:
                print("\n\nExiting due to user interruption.")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                print("Please try again or contact support if the issue persists.")


if __name__ == "__main__":
    # This block runs when the script is executed directly.
    
    try:
        # 1. Instantiate the Data Tier using the shared engine
        data_tier = DataTier(engine)
        
        # 2. Instantiate the Application Tier, passing the data_tier to it
        app_tier = ApplicationTier(data_tier)
        
        # 3. Instantiate the Presentation Tier (the CLI)
        cli = PresentationTier(app_tier)

        # 4. Start the application's command-line interface
        cli.run()
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)