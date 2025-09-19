import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging


class DataPreprocessor:
    """Data preprocessing utilities for pattern detection"""

    def __init__(self):
        self.price_columns = ["open", "high", "low", "close", "volume"]
        self.logger = logging.getLogger(self.__class__.__name__)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data"""
        try:
            # Remove duplicate timestamps
            data = data[~data.index.duplicated(keep="first")]

            # Sort by timestamp
            data = data.sort_index()

            # Handle missing values
            data = self._handle_missing_values(data)

            # Remove outliers
            data = self._remove_outliers(data)

            return data
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in market data"""
        try:
            # For price data, forward fill then interpolate
            price_data = data[["open", "high", "low", "close"]]
            price_data = price_data.fillna(method="ffill").interpolate()

            # For volume, fill with median
            volume_data = data["volume"].fillna(data["volume"].median())

            # Other columns forward fill
            other_columns = data.drop(
                columns=["open", "high", "low", "close", "volume"]
            )
            other_columns = other_columns.fillna(method="ffill")

            return pd.concat([price_data, volume_data, other_columns], axis=1)
        except Exception as e:
            self.logger.error(f"Error handling missing values: {e}")
            return data

    def _remove_outliers(
        self, data: pd.DataFrame, threshold: float = 3.0
    ) -> pd.DataFrame:
        """Remove outliers from market data"""
        try:
            cleaned_data = data.copy()

            for col in ["open", "high", "low", "close"]:
                if col in cleaned_data.columns:
                    z_scores = np.abs(
                        (cleaned_data[col] - cleaned_data[col].mean())
                        / cleaned_data[col].std()
                    )
                    cleaned_data = cleaned_data[z_scores < threshold]

            return cleaned_data
        except Exception as e:
            self.logger.error(f"Error removing outliers: {e}")
            return data

    def resample_data(self, data: pd.DataFrame, timeframe: str = "1D") -> pd.DataFrame:
        """Resample data to different timeframe"""
        try:
            # OHLCV aggregation functions
            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }

            # Handle other columns
            for col in data.columns:
                if col not in agg_dict and col != "volume":
                    agg_dict[col] = "last"

            return data.resample(timeframe).agg(agg_dict).dropna()
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            return data

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data"""
        try:
            data = data.copy()

            # Moving averages
            data["sma_20"] = data["close"].rolling(window=20).mean()
            data["sma_50"] = data["close"].rolling(window=50).mean()
            data["ema_12"] = data["close"].ewm(span=12).mean()
            data["ema_26"] = data["close"].ewm(span=26).mean()

            # RSI
            data["rsi"] = self._calculate_rsi(data["close"], period=14)

            # MACD
            macd_data = self._calculate_macd(data["close"])
            data = pd.concat([data, macd_data], axis=1)

            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(data["close"])
            data = pd.concat([data, bb_data], axis=1)

            # ATR
            data["atr"] = self._calculate_atr(data)

            # Volume indicators
            data["volume_sma"] = data["volume"].rolling(window=20).mean()
            data["volume_ratio"] = data["volume"] / data["volume_sma"]

            return data
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data

    def normalize_data(
        self, data: pd.DataFrame, method: str = "minmax"
    ) -> pd.DataFrame:
        """Normalize data for analysis"""
        try:
            normalized_data = data.copy()

            if method == "minmax":
                for col in ["open", "high", "low", "close"]:
                    if col in normalized_data.columns:
                        min_val = normalized_data[col].min()
                        max_val = normalized_data[col].max()
                        normalized_data[col] = (normalized_data[col] - min_val) / (
                            max_val - min_val
                        )

            elif method == "zscore":
                for col in ["open", "high", "low", "close"]:
                    if col in normalized_data.columns:
                        mean_val = normalized_data[col].mean()
                        std_val = normalized_data[col].std()
                        normalized_data[col] = (
                            normalized_data[col] - mean_val
                        ) / std_val

            return normalized_data
        except Exception as e:
            self.logger.error(f"Error normalizing data: {e}")
            return data

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics"""
        try:
            data = data.copy()

            # Price returns
            data["returns"] = data["close"].pct_change()
            data["log_returns"] = np.log(data["close"] / data["close"].shift())

            # Cumulative returns
            data["cumulative_returns"] = (1 + data["returns"]).cumprod() - 1

            # Volatility
            data["volatility_20"] = data["returns"].rolling(window=20).std() * np.sqrt(
                252
            )
            data["volatility_50"] = data["returns"].rolling(window=50).std() * np.sqrt(
                252
            )

            return data
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return data

    def align_data_timestamps(
        self, data: pd.DataFrame, timezone: str = "UTC"
    ) -> pd.DataFrame:
        """Align data timestamps"""
        try:
            # Convert timezone if needed
            if hasattr(data.index, "tz") and data.index.tz is not None:
                data = data.tz_convert(timezone)
            else:
                data = data.tz_localize(timezone)

            return data
        except Exception as e:
            self.logger.error(f"Error aligning timestamps: {e}")
            return data

    def validate_clean_data(self, data: pd.DataFrame) -> bool:
        """Validate that data is properly cleaned"""
        try:
            # Check for required columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]

            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check for NaN values
            for col in required_columns:
                if data[col].isna().any():
                    self.logger.error(f"Column '{col}' contains NaN values")
                    return False

            # Check price consistency
            if (data["high"] < data["low"]).any():
                self.logger.error("High prices cannot be lower than low prices")
                return False

            # Check for sufficient data points
            if len(data) < 20:
                self.logger.error("Insufficient data points for analysis")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error validating clean data: {e}")
            return False

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series()

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line

            return pd.DataFrame(
                {"macd": macd, "macd_signal": signal_line, "macd_histogram": histogram}
            )
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return pd.DataFrame()

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            return pd.DataFrame(
                {
                    "bb_upper": upper_band,
                    "bb_middle": sma,
                    "bb_lower": lower_band,
                    "bb_width": (upper_band - lower_band) / sma,
                }
            )
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return pd.DataFrame()

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high, low, close = data["high"], data["low"], data["close"]

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()

            return atr
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series()

    def create_synthetic_market_data(
        self, start_date: str, end_date: str, symbol: str = "SYNTHETIC"
    ) -> pd.DataFrame:
        """Create synthetic market data for testing"""
        try:
            # Generate date range
            dates = pd.date_range(start=start_date, end=end_date, freq="D")

            # Generate synthetic price data with realistic characteristics
            np.random.seed(42)  # For reproducible results

            # Starting price
            start_price = 100.0

            # Generate price path with random walk and trend
            price_changes = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
            prices = start_price * (1 + pd.Series(price_changes).cumprod())

            # Generate OHLCV data
            data = pd.DataFrame(index=dates)
            data["open"] = prices.shift(1).fillna(start_price)
            data["close"] = prices

            # Generate high and low with intraday volatility
            intraday_volatility = 0.01  # 1% intraday volatility
            daily_ranges = np.random.normal(0, intraday_volatility, len(dates))

            data["high"] = data[["open", "close"]].max(axis=1) * (
                1 + daily_ranges.abs() / 2
            )
            data["low"] = data[["open", "close"]].min(axis=1) * (
                1 - daily_ranges.abs() / 2
            )

            # Generate volume with some autocorrelation
            base_volume = 1000000
            volume_noise = np.random.normal(0, 0.3, len(dates))
            volume = base_volume * (1 + volume_noise).cumprod()
            data["volume"] = volume.abs()

            # Add some patterns for testing
            data = self._add_test_patterns(data)

            return data.dropna()
        except Exception as e:
            self.logger.error(f"Error creating synthetic data: {e}")
            return pd.DataFrame()

    def _add_test_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add test patterns to synthetic data"""
        try:
            # Add some volume spikes
            spike_indices = np.random.choice(data.index, size=5, replace=False)
            for idx in spike_indices:
                data.loc[idx, "volume"] *= 2

            # Add some price anomalies
            anomaly_indices = np.random.choice(data.index, size=3, replace=False)
            for idx in anomaly_indices:
                data.loc[idx, "high"] *= 1.05
                data.loc[idx, "low"] *= 0.95

            return data
        except Exception as e:
            self.logger.error(f"Error adding test patterns: {e}")
            return data
