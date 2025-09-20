import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.core.interfaces import PatternConfig, PatternSignal, PatternType
from src.utils.data_preprocessor import DataPreprocessor


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Create sample market data for testing"""
    np.random.seed(42)

    # Generate date range
    start_date = datetime.now() - timedelta(days=200)
    dates = pd.date_range(start=start_date, periods=200, freq="D")

    # Generate price data with trend and volatility
    base_price = 100.0
    trend = 0.001  # Slight upward trend
    volatility = 0.02

    prices = []
    current_price = base_price

    for i, date in enumerate(dates):
        # Random walk with trend
        change = np.random.normal(trend, volatility)
        current_price = current_price * (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data["close"] = prices
    data["open"] = prices  # Simplified for testing
    data["high"] = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    data["low"] = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    data["volume"] = np.random.lognormal(15, 1, 200)  # Realistic volume distribution

    return data


@pytest.fixture
def sample_market_data_with_patterns() -> pd.DataFrame:
    """Create sample market data with specific patterns for testing detectors"""
    np.random.seed(42)

    # Generate date range
    start_date = datetime.now() - timedelta(days=150)
    dates = pd.date_range(start=start_date, periods=150, freq="D")

    # Create a VCP pattern scenario
    prices = []

    # Stage 1: Initial decline
    decline_period = 20
    current_price = 100.0
    for i in range(decline_period):
        change = np.random.normal(-0.02, 0.03)  # Declining trend
        current_price = current_price * (1 + change)
        prices.append(current_price)

    # Stage 2: Volatility contraction
    contraction_period = 30
    for i in range(contraction_period):
        change = np.random.normal(0, 0.01)  # Low volatility
        current_price = current_price * (1 + change)
        prices.append(current_price)

    # Stage 3: Consolidation
    consolidation_period = 40
    base_price = current_price
    for i in range(consolidation_period):
        change = np.random.normal(0, 0.005)  # Very low volatility
        current_price = (
            base_price + (current_price - base_price) * 0.9 + change * base_price
        )
        prices.append(current_price)

    # Stage 4: Breakout
    breakout_period = 60
    for i in range(breakout_period):
        change = np.random.normal(0.01, 0.02)  # Upward breakout
        current_price = current_price * (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data["close"] = prices
    data["open"] = prices  # Simplified for testing
    data["high"] = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    data["low"] = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]

    # Add volume with pattern
    volume = []
    for i, price in enumerate(prices):
        if i < decline_period:
            # High volume during decline
            volume.append(np.random.lognormal(16, 0.5))
        elif i < decline_period + contraction_period:
            # Declining volume during contraction
            volume.append(np.random.lognormal(15, 0.5))
        elif i < decline_period + contraction_period + consolidation_period:
            # Low volume during consolidation
            volume.append(np.random.lognormal(14, 0.5))
        else:
            # High volume during breakout
            volume.append(np.random.lognormal(16, 0.5))

    data["volume"] = volume

    return data


@pytest.fixture
def pattern_config() -> PatternConfig:
    """Create a default pattern configuration for testing"""
    return PatternConfig(
        min_confidence=0.6,
        max_lookback=100,
        timeframe="1d",
        volume_threshold=1000000.0,
        volatility_threshold=0.001,
        reward_ratio=2.0,
    )


@pytest.fixture
def data_preprocessor() -> DataPreprocessor:
    """Create a data preprocessor instance for testing"""
    return DataPreprocessor()


@pytest.fixture
def sample_signal_data() -> List[Dict[str, Any]]:
    """Create sample signal data for testing"""
    return [
        {
            "symbol": "AAPL",
            "pattern_type": PatternType.VCP_BREAKOUT,
            "confidence": 0.85,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "target_price": 170.0,
            "timeframe": "1d",
            "timestamp": datetime.now(),
            "metadata": {
                "breakout_strength": 0.03,
                "volume_spike": True,
                "consolidation_range": 2.5,
            },
        },
        {
            "symbol": "GOOGL",
            "pattern_type": PatternType.FLAG_PATTERN,
            "confidence": 0.75,
            "entry_price": 2800.0,
            "stop_loss": 2750.0,
            "target_price": 3100.0,
            "timeframe": "1d",
            "timestamp": datetime.now() - timedelta(days=1),
            "metadata": {
                "flagpole_height": 0.08,
                "volume_decrease": True,
                "breakout_strength": 0.02,
            },
        },
        {
            "symbol": "MSFT",
            "pattern_type": PatternType.ASCENDING_TRIANGLE,
            "confidence": 0.65,
            "entry_price": 300.0,
            "stop_loss": 295.0,
            "target_price": 330.0,
            "timeframe": "1d",
            "timestamp": datetime.now() - timedelta(days=2),
            "metadata": {
                "triangle_type": "ascending",
                "breakout_strength": 0.015,
                "volume_spike": False,
            },
        },
    ]


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame for testing"""
    return pd.DataFrame()


@pytest.fixture
def invalid_dataframe() -> pd.DataFrame:
    """Create an invalid DataFrame for testing"""
    # Missing required columns
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10),
            "price": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        }
    )


@pytest.fixture
def corrupted_dataframe() -> pd.DataFrame:
    """Create a corrupted DataFrame with inconsistent data"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=50)

    data = pd.DataFrame(index=dates)
    data["close"] = np.random.normal(100, 10, 50)
    data["open"] = data["close"] + np.random.normal(0, 1, 50)
    data["high"] = data["close"] + np.random.normal(0, 2, 50)
    data["low"] = data["close"] - np.random.normal(0, 2, 50)
    data["volume"] = np.random.lognormal(15, 1, 50)

    # Add some corrupted data
    data.loc[dates[10], "high"] = 50  # Impossible high
    data.loc[dates[20], "low"] = 200  # Impossible low
    data.loc[dates[30], "close"] = np.nan  # NaN value

    return data
