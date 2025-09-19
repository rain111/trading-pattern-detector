#!/usr/bin/env python3
"""
Simple AAPL pattern detection test
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.interfaces import PatternConfig, PatternEngine
from detectors import (
    VCPBreakoutDetector,
    FlagPatternDetector,
    CupHandleDetector,
    DoubleBottomDetector,
    HeadAndShouldersDetector,
    RoundingBottomDetector,
    AscendingTriangleDetector,
    DescendingTriangleDetector,
    RisingWedgeDetector,
    FallingWedgeDetector,
)

def create_simple_test_data():
    """Create simple test data with clear patterns"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends

    # Create a simple ascending trend
    base_price = 100
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.normal(0, 1, len(dates))
    prices = base_price + trend + noise

    # Ensure all prices are positive and realistic
    prices = np.maximum(prices, 50)

    # Create OHLC data with proper price relationships
    data = pd.DataFrame({
        'open': prices,
        'high': prices * np.random.uniform(1.001, 1.020, len(prices)),
        'low': prices * np.random.uniform(0.980, 0.999, len(prices)),
        'close': prices,
        'volume': np.random.uniform(5000000, 50000000, len(prices)).astype(int),
    }, index=dates)

    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

    return data

def create_simple_flag_pattern(data):
    """Add a simple flag pattern to the data"""
    data_copy = data.copy()

    # Find middle section to create flag
    start_idx = len(data_copy) // 3
    end_idx = start_idx + 20

    # Create flag pattern (small pullback with consolidation)
    for i in range(start_idx, min(end_idx, len(data_copy))):
        if i < len(data_copy):
            # Small decline and consolidation
            data_copy.iloc[i, 3] *= 0.98  # Slight decline
            data_copy.iloc[i, 4] = int(data_copy.iloc[i, 4] * 0.7)  # Lower volume

    # Breakout above flag
    if end_idx < len(data_copy):
        data_copy.iloc[end_idx, 3] *= 1.03  # 3% breakout
        data_copy.iloc[end_idx, 4] = int(data_copy.iloc[end_idx, 4] * 1.8)  # Volume spike

    return data_copy

def main():
    print("🔍 Simple AAPL Pattern Detection Test")
    print("=" * 50)

    # Create simple test data
    print("📊 Creating simple test data...")
    raw_data = create_simple_test_data()
    pattern_data = create_simple_flag_pattern(raw_data)

    print(f"📈 Generated {len(pattern_data)} days of data")
    print(f"Price range: ${pattern_data['close'].min():.2f} - ${pattern_data['close'].max():.2f}")
    print()

    # Check data validation
    from core.interfaces import DataValidator
    validator = DataValidator()

    print("🔍 Data validation:")
    print(f"  Data shape: {pattern_data.shape}")
    print(f"  Sample data:\n{pattern_data.head()}")

    try:
        validator.validate_price_data(pattern_data)
        print("  ✅ Data validation passed")
    except Exception as e:
        print(f"  ❌ Data validation failed: {e}")
    print()

    # Setup detectors with low confidence threshold
    config = PatternConfig(min_confidence=0.3)

    # Create a subset of detectors for testing
    detectors = [
        VCPBreakoutDetector(config),
        FlagPatternDetector(config),
        CupHandleDetector(config),
        DoubleBottomDetector(config),
    ]

    # Create pattern engine
    engine = PatternEngine(detectors)

    # Run pattern detection
    print("🔍 Running pattern detection...")
    signals = engine.detect_patterns(pattern_data, "AAPL")

    print(f"📋 Found {len(signals)} trading patterns")
    print()

    # Display results
    for i, signal in enumerate(signals, 1):
        print(f"🎯 Pattern #{i}: {signal.pattern_type.value.upper()}")
        print(f"   📊 Confidence: {signal.confidence:.2f}")
        print(f"   💰 Entry Price: ${signal.entry_price:.2f}")
        print(f"   🛑 Stop Loss: ${signal.stop_loss:.2f}")
        print(f"   🎯 Target Price: ${signal.target_price:.2f}")
        print(f"   ⏱️  Expected Duration: {signal.expected_duration}")
        print(f"   🎲 Probability Target: {signal.probability_target:.1%}")
        print(f"   ⚖️  Risk Level: {signal.risk_level}")
        print(f"   📊 Signal Strength: {signal.signal_strength:.2f}")
        print(f"   🕐 Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print()

    if not signals:
        print("🤔 No patterns detected. This could be due to:")
        print("   1. Insufficient pattern strength in test data")
        print("   2. Pattern detector sensitivity settings")
        print("   3. Data validation issues")
        print()

        # Try with raw data (no patterns added)
        print("🔍 Testing with raw data (no patterns)...")
        raw_signals = engine.detect_patterns(raw_data, "AAPL")
        print(f"Raw data patterns found: {len(raw_signals)}")

    print("✅ Test complete!")

if __name__ == "__main__":
    main()