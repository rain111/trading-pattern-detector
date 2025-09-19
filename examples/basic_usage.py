#!/usr/bin/env python3
"""
Basic Usage Example for Trading Pattern Detector

This script demonstrates how to use the Trading Pattern Detector
to analyze market data and identify trading patterns.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_pattern_detector import PatternConfig, PatternEngine
from trading_pattern_detector.detectors import (
    VCPBreakoutDetector,
    FlagPatternDetector,
    TrianglePatternDetector,
    WedgePatternDetector,
    CupHandleDetector,
    DoubleBottomDetector,
)


def generate_sample_data(days: int = 200) -> pd.DataFrame:
    """Generate sample market data for demonstration"""
    np.random.seed(42)

    # Generate date range
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=days, freq='D')

    # Generate price data with some patterns
    base_price = 100.0
    volatility = 0.02

    prices = []
    current_price = base_price

    for i in range(days):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, volatility)
        current_price = current_price * (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = [p * (1 + np.random.normal(0, 0.001)) for p in prices]
    data['high'] = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    data['low'] = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    data['volume'] = np.random.lognormal(15, 1, days)

    return data


def main():
    """Main demonstration function"""
    print("Trading Pattern Detector - Basic Usage Example")
    print("=" * 50)

    # Step 1: Generate sample data
    print("\n1. Generating sample market data...")
    data = generate_sample_data(200)
    print(f"Generated {len(data)} days of data")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Step 2: Configure pattern detection
    print("\n2. Configuring pattern detection...")
    config = PatternConfig(
        min_confidence=0.6,           # Minimum confidence threshold
        max_lookback=100,             # Maximum historical data to analyze
        timeframe="1d",               # Data timeframe
        volume_threshold=1000000.0,   # Minimum volume threshold
        volatility_threshold=0.001,   # Volatility threshold
        reward_ratio=2.0             # Risk/reward ratio
    )

    print(f"Configuration:")
    print(f"  - Minimum confidence: {config.min_confidence}")
    print(f"  - Maximum lookback: {config.max_lookback}")
    print(f"  - Timeframe: {config.timeframe}")
    print(f"  - Volume threshold: {config.volume_threshold:,.0f}")
    print(f"  - Volatility threshold: {config.volatility_threshold}")
    print(f"  - Reward ratio: {config.reward_ratio}")

    # Step 3: Create detectors
    print("\n3. Creating pattern detectors...")
    detectors = [
        VCPBreakoutDetector(config),
        FlagPatternDetector(config),
        TrianglePatternDetector(config),
        WedgePatternDetector(config),
        CupHandleDetector(config),
        DoubleBottomDetector(config)
    ]

    detector_names = [det.__class__.__name__ for det in detectors]
    print(f"Detectors: {', '.join(detector_names)}")

    # Step 4: Initialize pattern engine
    print("\n4. Initializing pattern engine...")
    engine = PatternEngine(detectors)
    print(f"Pattern engine initialized with {len(detectors)} detectors")

    # Step 5: Detect patterns
    print("\n5. Detecting patterns...")
    symbol = "DEMO"
    signals = engine.detect_patterns(data, symbol)

    # Step 6: Display results
    print(f"\n6. Analysis Results:")
    print(f"Analyzed {len(data)} data points for {symbol}")
    print(f"Found {len(signals)} patterns above confidence threshold")

    if not signals:
        print("\nNo patterns detected above confidence threshold.")
        print("Try lowering the confidence threshold or generating more data.")
        return

    print("\n" + "=" * 60)
    print("DETECTED PATTERNS:")
    print("=" * 60)

    for i, signal in enumerate(signals, 1):
        print(f"\nPattern {i}: {signal.pattern_type.name}")
        print("-" * 40)
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Entry Price: ${signal.entry_price:.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Target Price: ${signal.target_price:.2f}")
        print(f"  Risk/Reward Ratio: {signal.target_price/signal.entry_price:.2f}:1")
        print(f"  Timeframe: {signal.timeframe}")
        print(f"  Timestamp: {signal.timestamp}")

        if signal.metadata:
            print(f"  Pattern Details:")
            for key, value in signal.metadata.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")

    # Step 7: Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS:")
    print("=" * 60)

    pattern_counts = {}
    for signal in signals:
        pattern_name = signal.pattern_type.name
        pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1

    print(f"Total patterns detected: {len(signals)}")
    print(f"Pattern distribution:")
    for pattern_name, count in pattern_counts.items():
        print(f"  {pattern_name}: {count}")

    # Calculate average statistics
    avg_confidence = sum(s.confidence for s in signals) / len(signals)
    avg_rr = sum(s.target_price/s.entry_price for s in signals) / len(signals)

    print(f"\nAverage confidence: {avg_confidence:.2f}")
    print(f"Average risk/reward ratio: {avg_rr:.2f}:1")

    # Step 8: Configuration tips
    print("\n" + "=" * 60)
    print("CONFIGURATION TIPS:")
    print("=" * 60)
    print("• Lower confidence threshold (0.4-0.5) to detect more patterns")
    print("• Increase volume threshold for liquid stocks")
    print("• Adjust max_lookback based on your trading style")
    print("• Use different timeframes for short-term vs. long-term patterns")
    print("• Combine multiple signals for better trade selection")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()