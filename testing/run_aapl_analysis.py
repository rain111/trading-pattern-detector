#!/usr/bin/env python3
"""
Sample data generation and pattern detection for AAPL
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.interfaces import PatternConfig, PatternEngine, PatternType
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

def generate_sample_aapl_data(days=252):
    """Generate realistic AAPL sample data"""
    np.random.seed(42)  # For reproducible results

    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends

    # Generate base price with trend
    base_price = 170
    trend = np.linspace(0, 20, len(dates))

    # Add noise and volatility
    volatility = np.random.normal(0, 2, len(dates))

    # Create price series
    prices = base_price + trend + volatility

    # Add some realistic patterns
    for i in range(len(prices)):
        # Add some volatility clustering
        if i > 10:
            prices[i] += prices[i-1] * 0.1 * np.random.normal(0, 0.1)

        # Ensure positive prices
        prices[i] = max(prices[i], 50)

    # Generate OHLC data
    data = pd.DataFrame({
        'open': prices * np.random.uniform(0.998, 1.002, len(prices)),
        'high': prices * np.random.uniform(1.001, 1.015, len(prices)),
        'low': prices * np.random.uniform(0.985, 0.999, len(prices)),
        'close': prices,
        'volume': np.random.uniform(50000000, 150000000, len(prices)).astype(int),
    }, index=dates)

    return data

def add_patterns_to_data(data):
    """Add some artificial patterns for testing"""
    data_copy = data.copy()

    # Add a small VCP pattern around day 50
    if len(data_copy) > 60:
        vcp_start = 50
        vcp_end = vcp_start + 30

        # Create volatility contraction
        for i in range(vcp_start, min(vcp_end, len(data_copy))):
            if i < len(data_copy):
                data_copy.iloc[i, 4] = int(data_copy.iloc[i, 4] * 0.8)  # Reduce volume
                data_copy.iloc[i, 3] *= 0.995  # Slight price decline

        # Add breakout
        if vcp_end < len(data_copy):
            data_copy.iloc[vcp_end, 3] *= 1.05  # 5% breakout
            data_copy.iloc[vcp_end, 4] = int(data_copy.iloc[vcp_end, 4] * 2.0)   # Volume spike

    # Add a rounding bottom pattern around day 120
    if len(data_copy) > 140:
        bottom_start = 120
        bottom_end = bottom_start + 40

        for i in range(bottom_start, min(bottom_end, len(data_copy))):
            if i < len(data_copy):
                # Create U-shape
                progress = (i - bottom_start) / (bottom_end - bottom_start)
                price_adjustment = -5 + 10 * progress  # From -5% to +5%
                data_copy.iloc[i, 3] *= (1 + price_adjustment * 0.01)

    return data_copy

def main():
    print("🔍 AAPL Trading Pattern Analysis")
    print("=" * 50)

    # Generate sample data
    print("📊 Generating sample AAPL data...")
    raw_data = generate_sample_aapl_data(252)
    data_with_patterns = add_patterns_to_data(raw_data)

    print(f"📈 Generated {len(data_with_patterns)} days of data")
    print(f"Price range: ${data_with_patterns['close'].min():.2f} - ${data_with_patterns['close'].max():.2f}")
    print()

    # Setup detectors
    config = PatternConfig(min_confidence=0.4)

    # Create all 12 detectors
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
                print(f"      {key}: {value}")
            print()

    # Display summary statistics
    print("📊 Summary Statistics:")
    print("-" * 30)

    if signals:
        confidence_scores = [s.confidence for s in signals]
        print(f"Average Confidence: {np.mean(confidence_scores):.2f}")
        print(f"High Confidence Signals (>0.8): {sum(1 for s in signals if s.confidence > 0.8)}")
        print(f"Medium Confidence Signals (0.6-0.8): {sum(1 for s in signals if 0.6 <= s.confidence <= 0.8)}")
        print(f"Low Confidence Signals (<0.6): {sum(1 for s in signals if s.confidence < 0.6)}")

        # Risk level distribution
        risk_levels = [s.risk_level for s in signals]
        print(f"Risk Levels - High: {risk_levels.count('high')}, Medium: {risk_levels.count('medium')}, Low: {risk_levels.count('low')}")
    else:
        print("No patterns detected in the generated data.")

    # Save results
    output_file = "aapl_analysis_results.csv"
    data_with_patterns.to_csv(output_file)
    print(f"\n💾 Data saved to: {output_file}")

    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()