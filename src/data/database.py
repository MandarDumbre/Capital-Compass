# src/data/database.py
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

class DataTier:
    """Handles all database interactions."""
    def __init__(self, engine):
        self.engine = engine

    def create_schema(self):
        """Initializes or resets the database schema for SQLite."""
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(text("DROP TABLE IF EXISTS daily_prices"))
                connection.execute(text("DROP TABLE IF EXISTS securities"))
                connection.execute(text("""
                CREATE TABLE securities (
                    id INTEGER PRIMARY KEY,
                    ticker VARCHAR(20) UNIQUE NOT NULL,
                    name VARCHAR(255)
                )"""))
                connection.execute(text("""
                CREATE TABLE daily_prices (
                    id INTEGER PRIMARY KEY,
                    security_id INTEGER,
                    date TEXT NOT NULL,
                    open_price DECIMAL(19, 4),
                    high_price DECIMAL(19, 4),
                    low_price DECIMAL(19, 4),
                    close_price DECIMAL(19, 4),
                    adjusted_close_price DECIMAL(19, 4),
                    volume BIGINT,
                    UNIQUE(security_id, date),
                    FOREIGN KEY (security_id) REFERENCES securities (id)
                )"""))
                connection.execute(text("""
                CREATE INDEX idx_security_date ON daily_prices (security_id, date)
                """))

    def get_security_id(self, ticker):
        """Retrieves the primary key for a given ticker."""
        with self.engine.connect() as connection:
            result = connection.execute(text("SELECT id FROM securities WHERE ticker = :ticker"), {'ticker': ticker}).fetchone()
            return result[0] if result else None

    def insert_securities(self, securities_data):
        """Inserts new securities into the database, ignoring duplicates."""
        with self.engine.connect() as connection:
            with connection.begin():
                # Use INSERT OR IGNORE for SQLite, which is more efficient than a
                # separate SELECT check for each row.
                stmt = text("INSERT OR IGNORE INTO securities (ticker, name) VALUES (:ticker, :name)")
                connection.execute(stmt, securities_data)

    def insert_daily_prices(self, ticker, prices_df):
        """Bulk inserts daily price data for a given ticker using UPSERT for SQLite."""
        security_id = self.get_security_id(ticker)
        if not security_id or prices_df.empty:
            return

        prices_df['security_id'] = security_id
        with self.engine.connect() as connection:
            with connection.begin():
                prices_df.to_sql('temp_daily_prices', connection, if_exists='replace', index=False)
                # Use the more compatible REPLACE INTO syntax for SQLite
                insert_sql = """
                    REPLACE INTO daily_prices (security_id, date, open_price, high_price, low_price, close_price, adjusted_close_price, volume)
                    SELECT security_id, date, open_price, high_price, low_price, close_price, adjusted_close_price, volume FROM temp_daily_prices;
                """
                connection.execute(text(insert_sql))

    @st.cache_data(ttl="1h")
    def get_price_history_for_tickers(_self, tickers: tuple):
        """Fetches and pivots historical price data for a tuple of tickers."""
        if not tickers:
            return pd.DataFrame()

        placeholders = ', '.join([f':ticker_{i}' for i in range(len(tickers))])
        params = {f'ticker_{i}': ticker for i, ticker in enumerate(tickers)}
        sql_query = f"""
            SELECT s.ticker, dp.date, dp.adjusted_close_price
            FROM daily_prices dp JOIN securities s ON dp.security_id = s.id
            WHERE s.ticker IN ({placeholders}) ORDER BY dp.date
        """
        with _self.engine.connect() as connection:
            df = pd.read_sql(text(sql_query), connection, params=params)

        if df.empty:
            return pd.DataFrame()

        price_pivot = df.pivot(index='date', columns='ticker', values='adjusted_close_price')
        price_pivot.index = pd.to_datetime(price_pivot.index, errors='coerce').normalize()
        return price_pivot