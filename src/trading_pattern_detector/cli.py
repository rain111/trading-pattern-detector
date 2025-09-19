#!/usr/bin/env python3
"""
Command Line Interface for Trading Pattern Detector

Provides easy-to-use CLI commands for pattern detection analysis.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from .core.interfaces import PatternConfig, PatternEngine
from .detectors import (
    VCPBreakoutDetector,
    FlagPatternDetector,
    TrianglePatternDetector,
    WedgePatternDetector,
    CupHandleDetector,
    DoubleBottomDetector,
)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def create_detectors(config: PatternConfig) -> List:
    """Create all available pattern detectors"""
    return [
        VCPBreakoutDetector(config),
        FlagPatternDetector(config),
        TrianglePatternDetector(config),
        WedgePatternDetector(config),
        CupHandleDetector(config),
        DoubleBottomDetector(config),
    ]


def load_data(file_path: str) -> pd.DataFrame:
    """Load market data from CSV file"""
    try:
        data = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)


def analyze_patterns(args: argparse.Namespace) -> None:
    """Analyze patterns in market data"""
    logging.info(f"Starting pattern analysis for {args.symbol}")

    # Load configuration
    config = PatternConfig(
        min_confidence=args.min_confidence,
        max_lookback=args.max_lookback,
        timeframe=args.timeframe,
        volume_threshold=args.volume_threshold,
        volatility_threshold=args.volatility_threshold,
        reward_ratio=args.reward_ratio,
    )

    # Load data
    data = load_data(args.input)

    # Create detectors and engine
    detectors = create_detectors(config)
    engine = PatternEngine(detectors)

    # Detect patterns
    signals = engine.detect_patterns(data, args.symbol)

    if args.output:
        # Save results to file
        results = []
        for signal in signals:
            result = {
                'pattern_type': signal.pattern_type.name,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'target_price': signal.target_price,
                'timeframe': signal.timeframe,
                'timestamp': signal.timestamp.isoformat(),
                'metadata': signal.metadata,
            }
            results.append(result)

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to {args.output}")
    else:
        # Print results to console
        if not signals:
            print("No patterns detected above confidence threshold.")
            return

        print(f"\nPattern Analysis Results for {args.symbol}")
        print("=" * 50)

        for i, signal in enumerate(signals, 1):
            print(f"\nPattern {i}: {signal.pattern_type.name}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Entry Price: ${signal.entry_price:.2f}")
            print(f"  Stop Loss: ${signal.stop_loss:.2f}")
            print(f"  Target Price: ${signal.target_price:.2f}")
            print(f"  Risk/Reward Ratio: {signal.target_price/signal.entry_price:.2f}:1")
            print(f"  Timeframe: {signal.timeframe}")
            print(f"  Timestamp: {signal.timestamp}")

            if signal.metadata:
                print(f"  Metadata: {json.dumps(signal.metadata, indent=4)}")


def list_patterns(args: argparse.Namespace) -> None:
    """List available pattern types"""
    patterns = [
        'VCP_BREAKOUT - Volatility Contraction Pattern Breakout',
        'FLAG_PATTERN - Flag/Pennant Continuation Pattern',
        'CUP_HANDLE - Cup and Handle Pattern',
        'DOUBLE_BOTTOM - Double Bottom Reversal Pattern',
        'ASCENDING_TRIANGLE - Ascending Triangle Pattern',
        'WEDGE_PATTERN - Wedge Pattern',
    ]

    print("Available Trading Patterns:")
    print("=" * 30)
    for pattern in patterns:
        print(f"  {pattern}")


def generate_sample_data(args: argparse.Namespace) -> None:
    """Generate sample market data for testing"""
    import numpy as np
    from datetime import datetime, timedelta

    logging.info("Generating sample market data...")

    # Set random seed for reproducible results
    np.random.seed(42)

    # Generate date range
    start_date = datetime.now() - timedelta(days=args.days)
    dates = pd.date_range(start=start_date, periods=args.days, freq='D')

    # Generate price data with some patterns
    base_price = args.base_price
    volatility = args.volatility

    prices = []
    current_price = base_price

    for i in range(args.days):
        # Random walk with trend
        change = np.random.normal(0.001, volatility)
        current_price = current_price * (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = [p * (1 + np.random.normal(0, 0.001)) for p in prices]
    data['high'] = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    data['low'] = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    data['volume'] = np.random.lognormal(15, 1, args.days)

    # Save to CSV
    data.to_csv(args.output)
    logging.info(f"Sample data saved to {args.output}")
    print(f"Generated {args.days} days of sample data starting from {start_date.date()}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description='Trading Pattern Detector - Analyze market data for trading patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  trading-pattern-detector analyze data.csv --symbol AAPL --min-confidence 0.7
  trading-pattern-detector analyze data.csv --symbol AAPL --output results.json
  trading-pattern-detector patterns
  trading-pattern-detector sample-data --days 100 --output sample.csv
        """
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze patterns in market data')
    analyze_parser.add_argument('input', help='Input CSV file path')
    analyze_parser.add_argument('--symbol', '-s', required=True, help='Symbol to analyze')
    analyze_parser.add_argument('--min-confidence', '-c', type=float, default=0.6,
                               help='Minimum confidence threshold (default: 0.6)')
    analyze_parser.add_argument('--max-lookback', '-l', type=int, default=100,
                               help='Maximum historical data to analyze (default: 100)')
    analyze_parser.add_argument('--timeframe', '-t', default='1d',
                               help='Data timeframe (default: 1d)')
    analyze_parser.add_argument('--volume-threshold', type=float, default=1000000.0,
                               help='Minimum volume threshold (default: 1000000.0)')
    analyze_parser.add_argument('--volatility-threshold', type=float, default=0.001,
                               help='Volatility threshold (default: 0.001)')
    analyze_parser.add_argument('--reward-ratio', type=float, default=2.0,
                               help='Risk/reward ratio (default: 2.0)')
    analyze_parser.add_argument('--output', '-o', help='Output JSON file path')

    # Patterns command
    patterns_parser = subparsers.add_parser('patterns', help='List available pattern types')

    # Sample data command
    sample_parser = subparsers.add_parser('sample-data', help='Generate sample market data')
    sample_parser.add_argument('--days', type=int, default=100, help='Number of days to generate (default: 100)')
    sample_parser.add_argument('--base-price', type=float, default=100.0, help='Base price (default: 100.0)')
    sample_parser.add_argument('--volatility', type=float, default=0.02, help='Volatility (default: 0.02)')
    sample_parser.add_argument('--output', '-o', required=True, help='Output CSV file path')

    return parser


def main() -> None:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)

    # Execute command
    if args.command == 'analyze':
        analyze_patterns(args)
    elif args.command == 'patterns':
        list_patterns(args)
    elif args.command == 'sample-data':
        generate_sample_data(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()