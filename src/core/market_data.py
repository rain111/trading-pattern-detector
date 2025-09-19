"""
Market Data Ingestion Layer

Automatically fetches market data from various sources including:
- Yahoo Finance (yfinance)
- Alpha Vantage
- Polygon.io
- Local CSV files

Provides standardized interfaces for pattern detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import yfinance as yf
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Base class for market data fetching"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch market data for a given symbol"""
        raise NotImplementedError

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate fetched data structure"""
        required_columns = ["open", "high", "low", "close", "volume"]
        return all(col in data.columns for col in required_columns)


class YahooFinanceFetcher(MarketDataFetcher):
    """Yahoo Finance data fetcher using yfinance"""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.logger.info("Initialized Yahoo Finance fetcher")

    def fetch_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance

        Args:
            symbol: Stock symbol (e.g., "AAPL", "GOOGL")
            period: Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")

        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Fetching {symbol} data from Yahoo Finance...")

            # Create ticker object
            ticker = yf.Ticker(symbol)

            # Fetch data
            if period == "max":
                data = ticker.history(period="max", interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)

            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Reset index and rename columns
            data.reset_index(inplace=True)
            data.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Dividends': 'dividends',
                'Stock Splits': 'stock_splits'
            }, inplace=True)

            # Ensure timestamp is datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Set timestamp as index
            data.set_index('timestamp', inplace=True)

            # Filter out weekends (for daily data)
            if interval == "1d":
                data = data[data.index.dayofweek < 5]

            # Clean data
            data = data.dropna()
            data = data[data['volume'] > 0]

            # Ensure numeric data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # Remove any rows with NaN values
            data = data.dropna()

            self.logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
            return data

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise


class AlphaVantageFetcher(MarketDataFetcher):
    """Alpha Vantage data fetcher"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://www.alphavantage.co/query"
        self.logger.info("Initialized Alpha Vantage fetcher")

    def fetch_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage

        Args:
            symbol: Stock symbol
            period: Time period (limited by API)
            interval: Data interval

        Returns:
            DataFrame with OHLCV data
        """
        try:
            self.logger.info(f"Fetching {symbol} data from Alpha Vantage...")

            # Map interval to Alpha Vantage function
            if interval == "1d":
                function = "TIME_SERIES_DAILY"
                outputsize = "full" if period == "max" else "compact"
            elif interval == "1wk":
                function = "TIME_SERIES_WEEKLY"
                outputsize = "full"
            else:
                raise ValueError(f"Interval {interval} not supported by Alpha Vantage")

            # Make API request
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": outputsize
            }

            # Simulate API call (in real implementation, use requests)
            # For now, return empty DataFrame
            self.logger.warning("Alpha Vantage API call not implemented - returning empty DataFrame")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            raise


class MarketDataIngestor:
    """Main market data ingestion class"""

    def __init__(self, fetcher: Optional[MarketDataFetcher] = None):
        self.fetcher = fetcher or YahooFinanceFetcher()
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data using the configured fetcher

        Args:
            symbol: Stock symbol
            period: Time period
            interval: Data interval

        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = self.fetcher.fetch_data(symbol, period, interval)

            if data.empty:
                raise ValueError(f"No data returned for symbol {symbol}")

            # Validate data
            if not self.fetcher.validate_data(data):
                raise ValueError(f"Invalid data structure for symbol {symbol}")

            self.logger.info(f"Successfully fetched data for {symbol}: {len(data)} rows")
            return data

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def fetch_multiple_stocks(self, symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks

        Args:
            symbols: List of stock symbols
            period: Time period
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}

        for symbol in symbols:
            try:
                data = self.fetch_stock_data(symbol, period, interval)
                results[symbol] = data
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed requests

        return results

    def save_data(self, data: pd.DataFrame, filename: str, format: str = "csv") -> None:
        """
        Save data to file

        Args:
            data: DataFrame to save
            filename: Output filename
            format: File format ("csv", "json", "parquet")
        """
        try:
            if format.lower() == "csv":
                data.to_csv(filename)
            elif format.lower() == "json":
                data.reset_index().to_json(filename, orient="records", date_format="iso")
            elif format.lower() == "parquet":
                data.to_parquet(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Data saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving data to {filename}: {str(e)}")
            raise

    def load_data(self, filename: str, format: str = "csv") -> pd.DataFrame:
        """
        Load data from file

        Args:
            filename: Input filename
            format: File format

        Returns:
            Loaded DataFrame
        """
        try:
            if format.lower() == "csv":
                data = pd.read_csv(filename, index_col=0, parse_dates=True)
            elif format.lower() == "json":
                data = pd.read_json(filename, orient="records")
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            elif format.lower() == "parquet":
                data = pd.read_parquet(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"Data loaded from {filename}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading data from {filename}: {str(e)}")
            raise

    def get_available_intervals(self) -> List[str]:
        """Get available data intervals"""
        return ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

    def get_available_periods(self) -> List[str]:
        """Get available time periods"""
        return ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]


class DataValidator:
    """Enhanced data validation for market data"""

    @staticmethod
    def validate_market_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate market data structure and quality

        Args:
            data: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        if data.empty:
            errors.append("DataFrame is empty")
            return False, errors

        # Check data types
        for col in required_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    errors.append(f"Column '{col}' is not numeric")

        # Check for NaN values
        for col in required_columns:
            if col in data.columns and data[col].isna().any():
                errors.append(f"Column '{col}' contains NaN values")

        # Check price logic consistency
        if all(col in data.columns for col in ["high", "low", "open", "close"]):
            if (data["high"] < data["low"]).any():
                errors.append("High prices cannot be lower than low prices")

            if (data["high"] < data["open"]).any() or (data["high"] < data["close"]).any():
                errors.append("High prices must be >= open and close prices")

            if (data["low"] > data["open"]).any() or (data["low"] > data["close"]).any():
                errors.append("Low prices must be <= open and close prices")

        # Check volume
        if "volume" in data.columns and (data["volume"] <= 0).any():
            errors.append("Volume must be positive")

        # Check index
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("DataFrame index must be datetime")

        # Check minimum data length
        if len(data) < 20:
            errors.append("Insufficient data: minimum 20 rows required")

        return len(errors) == 0, errors

    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess market data

        Args:
            data: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        cleaned = data.copy()

        # Drop rows with NaN values
        cleaned = cleaned.dropna()

        # Remove rows with zero or negative volume
        if "volume" in cleaned.columns:
            cleaned = cleaned[cleaned["volume"] > 0]

        # Sort by index
        cleaned = cleaned.sort_index()

        # Remove duplicates
        cleaned = cleaned[~cleaned.index.duplicated(keep='first')]

        return cleaned


# Convenience functions
def fetch_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Convenience function to fetch stock data"""
    ingestor = MarketDataIngestor()
    return ingestor.fetch_stock_data(symbol, period, interval)


def fetch_multiple_stocks(symbols: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Convenience function to fetch multiple stocks"""
    ingestor = MarketDataIngestor()
    return ingestor.fetch_multiple_stocks(symbols, period, interval)