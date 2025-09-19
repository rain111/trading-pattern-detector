#!/usr/bin/env python3
"""
Advanced Usage Example for Trading Pattern Detector

This script demonstrates advanced features including:
- Custom detector configuration
- Multi-symbol analysis
- Signal filtering and ranking
- Performance simulation
- Visualization
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_pattern_detector import PatternConfig, PatternEngine, PatternSignal
from trading_pattern_detector.detectors import (
    VCPBreakoutDetector,
    FlagPatternDetector,
    TrianglePatternDetector,
    WedgePatternDetector,
    CupHandleDetector,
    DoubleBottomDetector,
)


def generate_multi_symbol_data(symbols: List[str], days: int = 100) -> Dict[str, pd.DataFrame]:
    """Generate sample data for multiple symbols with different characteristics"""
    data = {}

    for symbol in symbols:
        np.random.seed(hash(symbol) % 1000)  # Different seed for each symbol

        # Generate different price characteristics for each symbol
        if symbol == "TECH":
            base_price = 150.0
            volatility = 0.03  # High volatility tech stock
            trend = 0.002
        elif symbol == "FINANCE":
            base_price = 50.0
            volatility = 0.015  # Medium volatility financial stock
            trend = 0.001
        elif symbol == "ENERGY":
            base_price = 75.0
            volatility = 0.025  # High volatility energy stock
            trend = -0.001
        else:  # "CONSUMER"
            base_price = 80.0
            volatility = 0.012  # Lower volatility consumer stock
            trend = 0.0015

        # Generate date range
        start_date = datetime.now() - timedelta(days=days)
        dates = pd.date_range(start=start_date, periods=days, freq='D')

        # Generate price data
        prices = []
        current_price = base_price

        for i in range(days):
            change = np.random.normal(trend, volatility)
            current_price = current_price * (1 + change)
            prices.append(current_price)

        # Create OHLCV data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = [p * (1 + np.random.normal(0, 0.001)) for p in prices]
        df['high'] = [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices]
        df['low'] = [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices]
        df['volume'] = np.random.lognormal(14 + np.random.uniform(-2, 2), 0.8, days)

        data[symbol] = df

    return data


def create_optimized_configs() -> Dict[str, PatternConfig]:
    """Create optimized configurations for different pattern types"""
    configs = {}

    # Conservative configuration for long-term patterns
    configs['conservative'] = PatternConfig(
        min_confidence=0.75,
        max_lookback=150,
        timeframe="1d",
        volume_threshold=2000000.0,
        volatility_threshold=0.0008,
        reward_ratio=3.0
    )

    # Aggressive configuration for short-term patterns
    configs['aggressive'] = PatternConfig(
        min_confidence=0.5,
        max_lookback=50,
        timeframe="1h",
        volume_threshold=500000.0,
        volatility_threshold=0.002,
        reward_ratio=1.5
    )

    # Balanced configuration for general use
    configs['balanced'] = PatternConfig(
        min_confidence=0.6,
        max_lookback=100,
        timeframe="1d",
        volume_threshold=1000000.0,
        volatility_threshold=0.001,
        reward_ratio=2.0
    )

    return configs


def filter_signals(signals: List[PatternSignal], min_rr: float = 1.5) -> List[PatternSignal]:
    """Filter signals based on risk/reward ratio and other criteria"""
    filtered = []

    for signal in signals:
        risk_reward = signal.target_price / signal.entry_price

        # Filter by risk/reward ratio
        if risk_reward < min_rr:
            continue

        # Filter by stop loss distance (too wide stops)
        stop_distance = (signal.entry_price - signal.stop_loss) / signal.entry_price
        if stop_distance > 0.1:  # More than 10% stop loss
            continue

        # Calculate quality score
        quality_score = (
            signal.confidence * 0.4 +
            min(risk_reward / 3.0, 1.0) * 0.4 +
            (1.0 - stop_distance / 0.1) * 0.2
        )

        # Add quality score to metadata
        signal.metadata['quality_score'] = quality_score

        filtered.append(signal)

    return sorted(filtered, key=lambda x: x.metadata['quality_score'], reverse=True)


def simulate_performance(signals: List[PatternSignal], data: pd.DataFrame,
                        hold_period: int = 20) -> Dict[str, Any]:
    """Simulate trading performance based on detected signals"""
    results = []

    for signal in signals:
        # Find signal entry point in data
        entry_date = signal.timestamp
        if entry_date not in data.index:
            continue

        # Get trade data
        trade_data = data.loc[entry_date:]

        # Check if stop loss or target was hit within hold period
        if len(trade_data) < 2:
            continue

        trade_prices = trade_data['close']
        trade_high = trade_data['high']
        trade_low = trade_data['low']

        # Initialize trade result
        result = {
            'symbol': signal.symbol,
            'pattern_type': signal.pattern_type.name,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'target_price': signal.target_price,
            'entry_date': entry_date,
            'confidence': signal.confidence,
            'result': 'open',
            'exit_price': None,
            'exit_date': None,
            'return_pct': None,
            'max_drawdown': None,
            'holding_days': 0
        }

        # Check for stop loss hit
        stop_loss_hit = (trade_low <= signal.stop_loss).any()

        # Check for target hit
        target_hit = (trade_high >= signal.target_price).any()

        # Find exit point
        if stop_loss_hit:
            stop_loss_idx = (trade_low <= signal.stop_loss).idxmax()
            result['exit_price'] = signal.stop_loss
            result['exit_date'] = stop_loss_idx
            result['result'] = 'stop_loss'
            result['return_pct'] = (signal.stop_loss - signal.entry_price) / signal.entry_price * 100
            result['holding_days'] = (stop_loss_idx - entry_date).days

        elif target_hit:
            target_idx = (trade_high >= signal.target_price).idxmax()
            result['exit_price'] = signal.target_price
            result['exit_date'] = target_idx
            result['result'] = 'target_hit'
            result['return_pct'] = (signal.target_price - signal.entry_price) / signal.entry_price * 100
            result['holding_days'] = (target_idx - entry_date).days

        else:
            # Exit at hold period
            hold_end_idx = min(hold_period, len(trade_data) - 1)
            result['exit_price'] = trade_prices.iloc[hold_end_idx]
            result['exit_date'] = trade_prices.index[hold_end_idx]
            result['result'] = 'time_exit'
            result['return_pct'] = (trade_prices.iloc[hold_end_idx] - signal.entry_price) / signal.entry_price * 100
            result['holding_days'] = hold_period

        # Calculate max drawdown
        if len(trade_data) > 1:
            price_series = trade_data['close']
            cumulative_returns = (price_series / signal.entry_price - 1) * 100
            result['max_drawdown'] = cumulative_returns.min()

        results.append(result)

    return results


def analyze_performance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trading performance statistics"""
    if not results:
        return {}

    total_trades = len(results)
    winning_trades = len([r for r in results if r['return_pct'] > 0])
    losing_trades = len([r for r in results if r['return_pct'] <= 0])

    total_return = sum(r['return_pct'] for r in results) / len(results)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Returns by result type
    stop_loss_returns = [r['return_pct'] for r in results if r['result'] == 'stop_loss']
    target_returns = [r['return_pct'] for r in results if r['result'] == 'target_hit']
    time_returns = [r['return_pct'] for r in results if r['result'] == 'time_exit']

    # Average holding days
    avg_holding_days = sum(r['holding_days'] for r in results) / len(results)

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_return_pct': total_return,
        'stop_loss_avg_return': np.mean(stop_loss_returns) if stop_loss_returns else 0,
        'target_avg_return': np.mean(target_returns) if target_returns else 0,
        'time_exit_avg_return': np.mean(time_returns) if time_returns else 0,
        'avg_holding_days': avg_holding_days,
        'best_trade': max(results, key=lambda x: x['return_pct']) if results else None,
        'worst_trade': min(results, key=lambda x: x['return_pct']) if results else None
    }


def main():
    """Main advanced demonstration function"""
    print("Trading Pattern Detector - Advanced Usage Example")
    print("=" * 60)

    # Step 1: Generate multi-symbol data
    print("\n1. Generating multi-symbol market data...")
    symbols = ["TECH", "FINANCE", "ENERGY", "CONSUMER"]
    data = generate_multi_symbol_data(symbols, 100)

    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} days, Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Step 2: Create optimized configurations
    print("\n2. Creating optimized configurations...")
    configs = create_optimized_configs()
    for name, config in configs.items():
        print(f"  {name}: min_conf={config.min_confidence}, lookback={config.max_lookback}")

    # Step 3: Analyze each symbol with different configurations
    print("\n3. Analyzing symbols with multiple configurations...")
    all_results = {}

    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")

        # Use balanced configuration for this example
        config = configs['balanced']
        detectors = [
            VCPBreakoutDetector(config),
            FlagPatternDetector(config),
            TrianglePatternDetector(config),
            WedgePatternDetector(config),
            CupHandleDetector(config),
            DoubleBottomDetector(config)
        ]

        engine = PatternEngine(detectors)
        signals = engine.detect_patterns(data[symbol], symbol)

        # Filter and rank signals
        filtered_signals = filter_signals(signals, min_rr=1.5)

        print(f"  Raw signals: {len(signals)}")
        print(f"  Filtered signals: {len(filtered_signals)}")

        # Simulate performance
        performance_results = simulate_performance(filtered_signals, data[symbol])
        performance_stats = analyze_performance(performance_results)

        all_results[symbol] = {
            'signals': filtered_signals,
            'performance': performance_results,
            'stats': performance_stats
        }

        # Display summary for this symbol
        if performance_stats:
            print(f"  Performance: {performance_stats['total_trades']} trades, "
                  f"{performance_stats['win_rate']:.1%} win rate, "
                  f"{performance_stats['total_return_pct']:.1f}% avg return")

    # Step 4: Cross-symbol analysis
    print("\n4. Cross-symbol Analysis Summary:")
    print("=" * 50)

    total_signals = sum(len(results['signals']) for results in all_results.values())
    total_trades = sum(results['stats']['total_trades'] for results in all_results.values() if results['stats'])
    overall_win_rate = sum(results['stats']['win_rate'] for results in all_results.values() if results['stats']) / len(all_results)

    print(f"Total signals detected: {total_signals}")
    print(f"Total trades simulated: {total_trades}")
    print(f"Overall win rate: {overall_win_rate:.1%}")

    # Step 5: Pattern type distribution
    print("\n5. Pattern Type Distribution:")
    pattern_counts = {}
    for results in all_results.values():
        for signal in results['signals']:
            pattern_name = signal.pattern_type.name
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1

    for pattern_name, count in sorted(pattern_counts.items()):
        print(f"  {pattern_name}: {count}")

    # Step 6: Best performing pattern
    print("\n6. Best Performing Pattern:")
    best_pattern = max(all_results.items(),
                      key=lambda x: x[1]['stats']['total_return_pct'] if x[1]['stats'] else -100)

    if best_pattern[1]['stats']:
        symbol = best_pattern[0]
        stats = best_pattern[1]['stats']
        print(f"  Symbol: {symbol}")
        print(f"  Total Return: {stats['total_return_pct']:.1f}%")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Avg Holding Days: {stats['avg_holding_days']:.1f}")

    # Step 7: Recommendations
    print("\n7. Trading Recommendations:")
    print("=" * 50)

    if total_trades > 0:
        avg_return = sum(results['stats']['total_return_pct'] for results in all_results.values() if results['stats']) / len(all_results)

        if avg_return > 5:
            print("✓ Good average returns detected - consider trading signals")
        elif avg_return > 0:
            print("⚠ Moderate returns - use strict risk management")
        else:
            print("✗ Low returns detected - consider optimizing parameters")

        if overall_win_rate > 0.6:
            print("✓ High win rate - good signal quality")
        elif overall_win_rate > 0.5:
            print("⚠ Moderate win rate - acceptable risk profile")
        else:
            print("✗ Low win rate - consider refining detection logic")

    print("\nAdvanced analysis completed!")


if __name__ == "__main__":
    main()