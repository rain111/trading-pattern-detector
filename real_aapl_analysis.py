#!/usr/bin/env python3
"""
Real AAPL Market Data Analysis with Pattern Detection

Uses yfinance to fetch real market data and apply all pattern detectors.
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
from core.market_data import MarketDataIngestor
from core.interfaces import DataValidator

def setup_analysis():
    """Setup analysis configuration"""
    print("🔍 Real AAPL Market Pattern Analysis")
    print("=" * 60)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def fetch_real_aapl_data():
    """Fetch real AAPL market data"""
    print("📊 Fetching real AAPL market data...")

    try:
        # Initialize data fetcher
        ingestor = MarketDataIngestor()

        # Fetch data for the last 2 years with daily intervals
        data = ingestor.fetch_stock_data("AAPL", period="2y", interval="1d")

        print(f"✅ Successfully fetched {len(data)} rows of AAPL data")
        print(f"📅 Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        print(f"💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"📊 Average volume: {data['volume'].mean():,.0f}")
        print()

        return data

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("🔄 Falling back to sample data generation...")
        return generate_fallback_data()

def generate_fallback_data():
    """Generate realistic fallback data if API fails"""
    print("🔄 Generating realistic AAPL sample data...")

    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends

    # Generate realistic AAPL price movements
    np.random.seed(42)  # For reproducible results
    base_price = 130

    # Create realistic price movement with trends and volatility
    trend = np.linspace(0, 40, len(dates))  # Overall upward trend
    noise = np.random.normal(0, 3, len(dates))  # Daily volatility
    volatility_cluster = np.random.exponential(2, len(dates)) * np.sin(np.arange(len(dates)) * 0.1)

    prices = base_price + trend + noise + volatility_cluster
    prices = np.maximum(prices, 50)  # Ensure positive prices

    # Create OHLC data with proper relationships
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 10, len(prices)),
        'low': prices - np.random.uniform(0, 10, len(prices)),
        'close': prices,
        'volume': np.random.uniform(50000000, 150000000, len(prices)).astype(int),
    }, index=dates)

    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))

    # Clean the data
    data = data.dropna()
    data = data[data['volume'] > 0]

    print(f"✅ Generated {len(data)} rows of realistic AAPL data")
    print(f"📅 Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
    print(f"💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print()

    return data

def validate_and_clean_data(data):
    """Validate and clean the market data"""
    print("🔍 Validating and cleaning data...")

    validator = DataValidator()

    # Clean OHLC data first
    print("🧹 Cleaning OHLC data...")
    cleaned_data = DataValidator.clean_ohlc_data(data)

    # Now validate the cleaned data
    try:
        validator.validate_price_data(cleaned_data)
        print("✅ Data validation passed")
        print(f"📊 Cleaned data: {len(cleaned_data)} rows")

        # Check if any rows were modified
        modified_rows = len(data) - len(cleaned_data)
        if modified_rows > 0:
            print(f"🔧 Fixed {modified_rows} OHLC inconsistencies")
        print()

        return cleaned_data
    except ValueError as e:
        print("❌ Data validation failed:")
        print(f"   - {e}")
        return None

def run_pattern_detection(data):
    """Run pattern detection on the data"""
    print("🔍 Running pattern detection...")

    # Setup detectors with moderate sensitivity
    config = PatternConfig(min_confidence=0.5)

    # Create all available detectors
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
    signals = engine.detect_patterns(data, "AAPL")

    print(f"📋 Pattern detection complete. Found {len(signals)} patterns.")
    print()

    return signals

def display_results(signals, data):
    """Display pattern detection results"""
    if not signals:
        print("🤔 No patterns detected in the data.")
        print("📊 This could be due to:")
        print("   1. Current market conditions not showing clear patterns")
        print("   2. Pattern detector sensitivity settings")
        print("   3. Insufficient data length or quality")
        print("   4. Market being in a trend/momentum phase vs. consolidation")
        return

    print("🎯 Detected Trading Patterns:")
    print("=" * 60)

    for i, signal in enumerate(signals, 1):
        print(f"\n📊 Pattern #{i}: {signal.pattern_type.value.upper()}")
        print(f"   🎯 Confidence: {signal.confidence:.2f}")
        print(f"   💰 Entry Price: ${signal.entry_price:.2f}")
        print(f"   🛑 Stop Loss: ${signal.stop_loss:.2f}")
        print(f"   🎯 Target Price: ${signal.target_price:.2f}")
        print(f"   📈 Risk/Reward: {signal.target_price/signal.stop_loss:.2f}:1")
        print(f"   ⏱️  Expected Duration: {signal.expected_duration}")
        print(f"   🎲 Probability Target: {signal.probability_target * 100:.1f}%" if signal.probability_target else "   🎲 Probability Target: N/A")
        print(f"   ⚖️  Risk Level: {signal.risk_level}")
        print(f"   📊 Signal Strength: {signal.signal_strength:.2f}")
        print(f"   🕐 Detected: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}")

        # Display pattern-specific metadata
        if signal.metadata:
            print(f"   📋 Pattern Details:")
            for key, value in signal.metadata.items():
                if isinstance(value, (int, float)):
                    print(f"      {key}: {value:.3f}")
                else:
                    print(f"      {key}: {value}")

        # Calculate potential returns
        potential_return = (signal.target_price - signal.entry_price) / signal.entry_price
        risk_distance = signal.entry_price - signal.stop_loss
        reward_distance = signal.target_price - signal.entry_price

        print(f"   📈 Potential Return: +{potential_return:.1%}")
        print(f"   📏 Risk Distance: ${risk_distance:.2f}")
        print(f"   📏 Reward Distance: ${reward_distance:.2f}")

    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print("-" * 40)
    confidence_scores = [s.confidence for s in signals]
    print(f"Average Confidence: {np.mean(confidence_scores):.2f}")
    print(f"High Confidence Signals (>0.7): {sum(1 for s in signals if s.confidence > 0.7)}")
    print(f"Medium Confidence Signals (0.5-0.7): {sum(1 for s in signals if 0.5 <= s.confidence <= 0.7)}")

    # Risk distribution
    risk_levels = [s.risk_level for s in signals]
    print(f"Risk Levels - High: {risk_levels.count('high')}, Medium: {risk_levels.count('medium')}, Low: {risk_levels.count('low')}")

    # Pattern type distribution
    pattern_types = [s.pattern_type.value for s in signals]
    print(f"Pattern Types Detected: {', '.join(set(pattern_types))}")

    # Expected returns
    expected_returns = [(s.target_price - s.entry_price) / s.entry_price for s in signals]
    if expected_returns:
        print(f"Average Expected Return: {np.mean(expected_returns):.1%}")
        print(f"Maximum Expected Return: {max(expected_returns):.1%}")
        print(f"Minimum Expected Return: {min(expected_returns):.1%}")

def save_results(signals, data):
    """Save analysis results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save data
    data_file = f"aapl_data_{timestamp}.csv"
    data.to_csv(data_file)
    print(f"\n💾 Market data saved to: {data_file}")

    # Save signals if any found
    if signals:
        signals_data = []
        for signal in signals:
            signal_data = {
                'symbol': signal.symbol,
                'pattern_type': signal.pattern_type.value,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'target_price': signal.target_price,
                'risk_level': signal.risk_level,
                'expected_duration': signal.expected_duration,
                'probability_target': signal.probability_target,
                'timestamp': signal.timestamp.strftime('%Y-%m-%d %H:%M'),
                'potential_return': (signal.target_price - signal.entry_price) / signal.entry_price
            }
            signals_data.append(signal_data)

        signals_df = pd.DataFrame(signals_data)
        signals_file = f"aapl_signals_{timestamp}.csv"
        signals_df.to_csv(signals_file, index=False)
        print(f"📊 Trading signals saved to: {signals_file}")

def main():
    """Main analysis function"""
    setup_analysis()

    # Fetch real market data
    data = fetch_real_aapl_data()

    # Validate and clean data
    cleaned_data = validate_and_clean_data(data)
    if cleaned_data is None:
        print("❌ Cannot proceed with analysis due to data validation issues.")
        return

    # Run pattern detection
    signals = run_pattern_detection(cleaned_data)

    # Display results
    display_results(signals, cleaned_data)

    # Save results
    save_results(signals, cleaned_data)

    # Final summary
    print(f"\n✅ Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if signals:
        print(f"🎯 Found {len(signals)} trading opportunities for AAPL")
        print("📊 Review the signals above for potential trading opportunities")
    else:
        print("📊 No clear patterns detected in current AAPL data")
        print("🔍 Consider different timeframes or market conditions")

if __name__ == "__main__":
    main()