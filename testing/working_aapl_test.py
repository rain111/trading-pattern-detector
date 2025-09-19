#!/usr/bin/env python3
"""
Working AAPL pattern detection test
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.interfaces import PatternConfig, PatternEngine, DataValidator
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

def create_valid_aapl_data():
    """Create valid AAPL-style test data"""
    dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends

    # Create AAPL-style price movement
    base_price = 170
    trend = np.linspace(0, 30, len(dates))
    noise = np.random.normal(0, 3, len(dates))
    prices = base_price + trend + noise

    # Ensure all prices are positive and realistic
    prices = np.maximum(prices, 100)

    # Create valid OHLC data
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 8, len(prices)),
        'low': prices - np.random.uniform(0, 8, len(prices)),
        'close': prices,
        'volume': np.random.uniform(50000000, 150000000, len(prices)).astype(int),
    }, index=dates)

    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

    return data

def add_flag_pattern(data):
    """Add a flag pattern to the data"""
    data_copy = data.copy()

    # Add flag pattern starting at day 60
    flag_start = 60
    flag_end = flag_start + 15

    # Create flag (small pullback with consolidation)
    for i in range(flag_start, min(flag_end, len(data_copy))):
        if i < len(data_copy):
            # Small decline and consolidation
            data_copy.iloc[i, 3] *= 0.995  # Slight decline
            data_copy.iloc[i, 4] = int(data_copy.iloc[i, 4] * 0.8)  # Lower volume

    # Breakout above flag
    if flag_end < len(data_copy):
        data_copy.iloc[flag_end, 3] *= 1.025  # 2.5% breakout
        data_copy.iloc[flag_end, 4] = int(data_copy.iloc[flag_end, 4] * 1.5)  # Volume spike

    return data_copy

def add_vcp_pattern(data):
    """Add a VCP pattern to the data"""
    data_copy = data.copy()

    # Add VCP pattern starting at day 90
    vcp_start = 90
    vcp_end = vcp_start + 25

    # Create VCP (volatility contraction then breakout)
    for i in range(vcp_start, min(vcp_end, len(data_copy))):
        if i < len(data_copy):
            # Volatility contraction
            data_copy.iloc[i, 4] = int(data_copy.iloc[i, 4] * 0.7)  # Decreasing volume
            if i < vcp_end - 5:
                data_copy.iloc[i, 3] *= 0.998  # Slight decline
            else:
                # Breakout phase
                data_copy.iloc[i, 3] *= 1.002  # Gradual rise

    # Breakout
    if vcp_end < len(data_copy):
        data_copy.iloc[vcp_end, 3] *= 1.03  # 3% breakout
        data_copy.iloc[vcp_end, 4] = int(data_copy.iloc[vcp_end, 4] * 2.0)  # Volume spike

    return data_copy

def main():
    print("🔍 AAPL Trading Pattern Detection")
    print("=" * 50)

    # Create valid AAPL data
    print("📊 Creating valid AAPL test data...")
    raw_data = create_valid_aapl_data()

    # Add patterns
    data_with_patterns = add_flag_pattern(raw_data)
    data_with_patterns = add_vcp_pattern(data_with_patterns)

    print(f"📈 Generated {len(data_with_patterns)} days of data")
    print(f"Price range: ${data_with_patterns['close'].min():.2f} - ${data_with_patterns['close'].max():.2f}")
    print()

    # Validate data
    validator = DataValidator()
    print("🔍 Data validation:")
    try:
        validator.validate_price_data(data_with_patterns)
        print("✅ Data validation passed!")
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return

    # Setup detectors
    config = PatternConfig(min_confidence=0.4)

    # Create all detectors
    detectors = [
        VCPBreakoutDetector(config),
        FlagPatternDetector(config),
        CupHandleDetector(config),
        DoubleBottomDetector(config),
        HeadAndShouldersDetector(config),
        RoundingBottomDetector(config),
        AscendingTriangleDetector(config),
        DescendingTriangleDetector(config),
        RisingWedgeDetector(config),
        FallingWedgeDetector(config),
    ]

    # Create pattern engine
    engine = PatternEngine(detectors)

    # Run pattern detection
    print("🔍 Running pattern detection...")
    signals = engine.detect_patterns(data_with_patterns, "AAPL")

    print(f"📋 Found {len(signals)} trading patterns")
    print()

    # Display results
    if signals:
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

            # Display metadata if available
            if signal.metadata:
                print("   📋 Pattern Metadata:")
                for key, value in signal.metadata.items():
                    if isinstance(value, (int, float)):
                        print(f"      {key}: {value:.3f}")
                    else:
                        print(f"      {key}: {value}")
                print()
    else:
        print("🤔 No patterns detected. The test data might need:")
        print("   1. Stronger pattern formation")
        print("   2. Longer time series")
        print("   3. More pronounced volume changes")
        print("   4. Different pattern configurations")

    # Summary statistics
    print("📊 Summary Statistics:")
    print("-" * 30)

    if signals:
        confidence_scores = [s.confidence for s in signals]
        print(f"Average Confidence: {np.mean(confidence_scores):.2f}")
        print(f"High Confidence Signals (>0.7): {sum(1 for s in signals if s.confidence > 0.7)}")
        print(f"Medium Confidence Signals (0.5-0.7): {sum(1 for s in signals if 0.5 <= s.confidence <= 0.7)}")
        print(f"Low Confidence Signals (<0.5): {sum(1 for s in signals if s.confidence < 0.5)}")

        # Risk level distribution
        risk_levels = [s.risk_level for s in signals]
        print(f"Risk Levels - High: {risk_levels.count('high')}, Medium: {risk_levels.count('medium')}, Low: {risk_levels.count('low')}")
    else:
        print("No patterns detected in the generated data.")

    # Save results
    output_file = "aapl_working_results.csv"
    data_with_patterns.to_csv(output_file)
    print(f"\n💾 Data saved to: {output_file}")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()