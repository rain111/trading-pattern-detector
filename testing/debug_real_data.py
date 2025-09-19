#!/usr/bin/env python3
"""
Debug real data validation issues
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.market_data import MarketDataIngestor
from core.interfaces import DataValidator

def main():
    print("🔍 Debug Real Data Validation")
    print("=" * 50)

    # Fetch real AAPL data
    ingestor = MarketDataIngestor()
    data = ingestor.fetch_stock_data("AAPL", period="2y", interval="1d")

    print(f"Fetched {len(data)} rows")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Data types:\n{data.dtypes}")

    # Check for OHLC violations
    print("\n🔍 OHLC Data Quality Check:")
    print(f"High < Low: {(data['high'] < data['low']).sum()}")
    print(f"High < Open: {(data['high'] < data['open']).sum()}")
    print(f"High < Close: {(data['high'] < data['close']).sum()}")
    print(f"Low > Open: {(data['low'] > data['open']).sum()}")
    print(f"Low > Close: {(data['low'] > data['close']).sum()}")

    # Show sample data
    print("\n📊 Sample Data (first 5 rows):")
    print(data[['open', 'high', 'low', 'close', 'volume']].head())

    # Check specific violations
    violations = data[(data['high'] < data['low']) | (data['high'] < data['open']) | (data['high'] < data['close']) | (data['low'] > data['open']) | (data['low'] > data['close'])]
    if len(violations) > 0:
        print(f"\n❌ Found {len(violations)} rows with OHLC violations")
        print("Sample violations:")
        print(violations[['open', 'high', 'low', 'close']].head())
    else:
        print("\n✅ No OHLC violations found")

    # Try validation
    print("\n🔍 Running Data Validation:")
    from src.core.interfaces import DataValidator
    validator = DataValidator()

    # Clean the data first
    print("\n🧹 Cleaning OHLC data...")
    cleaned_data = DataValidator.clean_ohlc_data(data)
    print(f"   Cleaned data shape: {cleaned_data.shape}")

    # Now validate the cleaned data
    try:
        validator.validate_price_data(cleaned_data)
        print("✅ Data validation passed!")
    except ValueError as e:
        print("❌ Data validation failed:")
        print(f"   - {e}")

if __name__ == "__main__":
    main()