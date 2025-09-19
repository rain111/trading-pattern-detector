#!/usr/bin/env python3
"""
Quick backtest for pattern detection testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from core.interfaces import PatternConfig, PatternEngine, DataValidator
from core.market_data import MarketDataIngestor
from detectors import (
    HeadAndShouldersDetector,
    DoubleBottomDetector,
)

def quick_pattern_test():
    """Test pattern detection on a single stock with limited data"""
    print("🔍 Quick Pattern Detection Test")
    print("=" * 40)

    # Test with single stock and limited time period
    ingestor = MarketDataIngestor()
    data = ingestor.fetch_stock_data("AAPL", period="1y", interval="1d")

    if data.empty:
        print("❌ No data fetched")
        return

    print(f"📊 Fetched {len(data)} rows of AAPL data")
    print(f"📅 Date range: {data.index.min()} to {data.index.max()}")

    # Clean data
    cleaned_data = DataValidator.clean_ohlc_data(data)
    print(f"🧹 Cleaned data: {len(cleaned_data)} rows")

    # Run pattern detection with fewer detectors for speed
    config = PatternConfig(min_confidence=0.5)
    detectors = [
        HeadAndShouldersDetector(config),
        DoubleBottomDetector(config),
    ]

    engine = PatternEngine(detectors)
    signals = engine.detect_patterns(cleaned_data, "AAPL")

    print(f"🎯 Found {len(signals)} patterns")

    # Display signals
    for i, signal in enumerate(signals[:3], 1):  # Show first 3 signals
        print(f"\n📊 Signal #{i}: {signal.pattern_type.value}")
        print(f"   🎯 Confidence: {signal.confidence:.2f}")
        print(f"   💰 Entry: ${signal.entry_price:.2f}")
        print(f"   🛑 Stop: ${signal.stop_loss:.2f}")
        print(f"   🎯 Target: ${signal.target_price:.2f}")
        print(f"   🕐 Detected: {signal.timestamp}")

    if len(signals) == 0:
        print("🤔 No patterns detected in the data")
        print("This could be due to:")
        print("  1. Data not showing clear patterns")
        print("  2. Pattern sensitivity settings")
        print("  3. Insufficient data length")

    return signals

if __name__ == "__main__":
    signals = quick_pattern_test()