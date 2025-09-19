#!/usr/bin/env python3
"""
Fetch top 50 US stocks by market cap and download 10-year price/volume history
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TopStocksFetcher:
    """Fetch top 50 US stocks by market cap"""

    def __init__(self):
        self.top_50_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS', 'NFLX', 'ADBE', 'CRM', 'BAC',
            'XOM', 'T', 'CMCSA', 'ABT', 'COST', 'AVGO', 'KO', 'PEP', 'NFLX', 'CSCO',
            'VZ', 'NKE', 'WFC', 'IBM', 'INTC', 'TXN', 'MRK', 'GE', 'BA', 'MCD',
            'BA', 'CAT', 'HON', 'UPS', 'MDT', 'UNP', 'AMGN', 'DD', 'MDLZ', 'PLTR'
        ]

        # Remove duplicates
        self.top_50_tickers = list(set(self.top_50_tickers))

        # Add some major stocks that might be missing
        additional_stocks = ['BRK-B', 'WMT', 'GS', 'MS', 'BLK', 'TSM', 'ASML', 'SAP', 'BABA']
        self.top_50_tickers.extend(additional_stocks)
        self.top_50_tickers = self.top_50_tickers[:50]  # Keep top 50

        self.output_dir = "market_data"
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_stock_data(self, symbol: str, period: str = "10y", interval: "1d" = "1d") -> pd.DataFrame:
        """Fetch historical data for a single stock"""
        try:
            logger.info(f"Fetching data for {symbol}...")

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Clean the data
            data = data.dropna()
            data = data[data['volume'] > 0]

            # Reset index to make date a column
            data = data.reset_index()

            # Standardize column names
            data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']

            # Convert date to datetime
            data['date'] = pd.to_datetime(data['date'])

            # Set date as index
            data = data.set_index('date')

            # Add symbol column
            data['symbol'] = symbol

            logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all top 50 stocks"""
        all_data = {}

        for symbol in self.top_50_tickers:
            data = self.fetch_stock_data(symbol)
            if not data.empty:
                all_data[symbol] = data

        logger.info(f"Successfully fetched data for {len(all_data)} out of {len(self.top_50_tickers)} stocks")
        return all_data

    def save_to_parquet(self, data: Dict[str, pd.DataFrame]) -> None:
        """Save all stock data to parquet files"""
        for symbol, df in data.items():
            filename = os.path.join(self.output_dir, f"{symbol}_10year.parquet")
            df.to_parquet(filename)
            logger.info(f"Saved {symbol} data to {filename}")

    def create_combined_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a combined dataset with all stocks"""
        all_dataframes = []

        for symbol, df in data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            all_dataframes.append(df_copy)

        combined_data = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Combined dataset created with {len(combined_data)} total rows")
        return combined_data

    def get_stock_info(self) -> pd.DataFrame:
        """Get basic information for all stocks"""
        stock_info = []

        for symbol in self.top_50_tickers:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                stock_info.append({
                    'symbol': symbol,
                    'name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', 0),
                    'current_price': info.get('currentPrice', 0),
                    'currency': info.get('currency', 'USD')
                })
            except Exception as e:
                logger.warning(f"Could not get info for {symbol}: {e}")
                stock_info.append({
                    'symbol': symbol,
                    'name': '',
                    'sector': '',
                    'industry': '',
                    'market_cap': 0,
                    'current_price': 0,
                    'currency': 'USD'
                })

        return pd.DataFrame(stock_info)

    def run(self):
        """Main execution function"""
        logger.info("Starting top 50 stocks data fetch...")

        # Get stock information
        logger.info("Fetching stock information...")
        stock_info = self.get_stock_info()

        # Sort by market cap to get true top 50
        stock_info = stock_info.sort_values('market_cap', ascending=False).head(50)
        self.top_50_tickers = stock_info['symbol'].tolist()

        logger.info(f"Top 50 stocks by market cap: {self.top_50_tickers}")

        # Save stock info
        stock_info.to_csv(os.path.join(self.output_dir, "top_50_stocks_info.csv"), index=False)

        # Fetch historical data
        logger.info("Fetching 10-year historical data...")
        all_data = self.fetch_all_stocks()

        # Save to parquet files
        self.save_to_parquet(all_data)

        # Create combined dataset
        combined_data = self.create_combined_dataset(all_data)
        combined_data.to_parquet(os.path.join(self.output_dir, "top_50_stocks_combined.parquet"))

        # Save tickers list
        with open(os.path.join(self.output_dir, "top_50_tickers.pkl"), 'wb') as f:
            pickle.dump(self.top_50_tickers, f)

        logger.info("Data fetch completed successfully!")

        return {
            'stock_info': stock_info,
            'all_data': all_data,
            'combined_data': combined_data,
            'top_tickers': self.top_50_tickers
        }

if __name__ == "__main__":
    fetcher = TopStocksFetcher()
    results = fetcher.run()