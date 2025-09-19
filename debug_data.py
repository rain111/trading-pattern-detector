#!/usr/bin/env python3
"""
Debug data generation and validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.interfaces import DataValidator

def create_valid_test_data():
    """Create test data that passes validation"""
    dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends

    # Create prices
    prices = np.linspace(100, 150, len(dates)) + np.random.normal(0, 2, len(dates))
    prices = np.maximum(prices, 50)

    # Create valid OHLC data
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 5, len(prices)),
        'low': prices - np.random.uniform(0, 5, len(prices)),
        'close': prices,
        'volume': np.random.uniform(5000000, 50000000, len(prices)).astype(int),
    }, index=dates)

    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

    return data

def main():
    print("🔍 Debug Data Generation and Validation")
    print("=" * 50)

    validator = DataValidator()

    # Create test data
    data = create_valid_test_data()

    print(f"Generated {len(data)} rows of data")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Check validation
    print("\n🔍 Data validation check:")
    try:
        validator.validate_price_data(data)
        print("✅ Data validation passed!")
    except Exception as e:
        print(f"❌ Data validation failed: {e}")

    # Check sample data
    print("\n📊 Sample data:")
    print(data.head())

    # Check data quality
    print("\n🔍 Data quality:")
    print(f"Missing values: {data.isnull().sum().sum()}")
    print(f"High < Low: {(data['high'] < data['low']).sum()}")
    print(f"High < Open: {(data['high'] < data['open']).sum()}")
    print(f"High < Close: {(data['high'] < data['close']).sum()}")
    print(f"Low > Open: {(data['low'] > data['open']).sum()}")
    print(f"Low > Close: {(data['low'] > data['close']).sum()}")

if __name__ == "__main__":
    main()