#!/usr/bin/env python3
"""
Comprehensive Pattern Detection Backtesting System
Runs pattern detection on top 50 stocks with performance analysis and Sharpe ratio-based position sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging
from typing import Dict, List, Tuple, Optional
import pickle
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.interfaces import PatternConfig, PatternEngine, PatternSignal, PatternType
from core.market_data import MarketDataIngestor
from core.interfaces import DataValidator
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternBacktester:
    """Comprehensive pattern detection backtester"""

    def __init__(self, start_date: str = "2014-01-01", end_date: str = None,
                 initial_capital: float = 1000000, position_size_method: str = "sharpe"):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        self.initial_capital = initial_capital
        self.position_size_method = position_size_method
        self.lookback_period = 2 * 252  # 2 years in trading days

        # Performance tracking
        self.all_trades = []
        self.portfolio_history = []
        self.pattern_performance = {}
        self.sharpe_ratios = {}

    def get_top_50_stocks(self) -> List[str]:
        """Get list of top 50 stocks to analyze"""
        # Major US stocks by market cap
        top_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'MA', 'PYPL', 'DIS', 'NFLX', 'ADBE', 'CRM', 'BAC',
            'XOM', 'T', 'CMCSA', 'ABT', 'COST', 'AVGO', 'KO', 'PEP', 'NFLX', 'CSCO',
            'VZ', 'NKE', 'WFC', 'IBM', 'INTC', 'TXN', 'MRK', 'GE', 'BA', 'MCD',
            'BA', 'CAT', 'HON', 'UPS', 'MDT', 'UNP', 'AMGN', 'DD', 'MDLZ', 'PLTR'
        ]
        return top_stocks[:50]

    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch historical data for a single stock"""
        try:
            ingestor = MarketDataIngestor()

            # Fetch data for the required period plus lookback
            fetch_start = start_date - timedelta(days=self.lookback_period + 365)
            data = ingestor.fetch_stock_data(symbol, period="max", interval="1d")

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Filter date range - handle timezone-aware vs naive timestamps
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                fetch_start_tz = fetch_start.tz_localize(data.index.tz)
                end_date_tz = end_date.tz_localize(data.index.tz)
                data = data[(data.index >= fetch_start_tz) & (data.index <= end_date_tz)]
            else:
                data = data[(data.index >= fetch_start) & (data.index <= end_date)]

            # Clean and validate
            data = DataValidator.clean_ohlc_data(data)

            logger.info(f"Fetched {len(data)} rows for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def run_pattern_detection(self, data: pd.DataFrame, symbol: str) -> List[PatternSignal]:
        """Run pattern detection on stock data"""
        try:
            # Setup detectors
            config = PatternConfig(min_confidence=0.5)
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

            engine = PatternEngine(detectors)
            signals = engine.detect_patterns(data, symbol)

            logger.info(f"Found {len(signals)} patterns for {symbol}")
            return signals

        except Exception as e:
            logger.error(f"Pattern detection failed for {symbol}: {e}")
            return []

    def simulate_trade(self, signal: PatternSignal, stock_data: pd.DataFrame,
                      entry_date: datetime, portfolio_value: float) -> Dict:
        """Simulate a single trade based on pattern signal"""
        try:
            # Calculate position size based on method
            if self.position_size_method == "sharpe":
                position_size = self._calculate_sharpe_based_size(signal, portfolio_value)
            else:
                position_size = portfolio_value * 0.1  # Default 10% of portfolio

            # Get entry price (close price on entry date)
            if entry_date not in stock_data.index:
                return None

            entry_price = stock_data.loc[entry_date, 'close']
            shares = position_size / entry_price

            # Calculate stop loss and target prices
            stop_loss_price = signal.stop_loss
            target_price = signal.target_price

            # Find exit conditions
            trade_duration = 0
            exit_date = None
            exit_price = None
            exit_reason = ""

            # Look for exit conditions in subsequent data
            future_dates = stock_data.index[stock_data.index > entry_date]

            for date in future_dates:
                if trade_duration > 252:  # Max 1 year holding period
                    exit_date = date
                    exit_price = stock_data.loc[date, 'close']
                    exit_reason = "max_holding_period"
                    break

                current_price = stock_data.loc[date, 'close']
                trade_duration += 1

                # Check stop loss
                if current_price <= stop_loss_price:
                    exit_date = date
                    exit_price = current_price
                    exit_reason = "stop_loss"
                    break

                # Check target
                if current_price >= target_price:
                    exit_date = date
                    exit_price = current_price
                    exit_reason = "target_reached"
                    break

            if exit_date is None:
                return None  # Trade not exited within timeframe

            # Calculate PnL
            entry_value = shares * entry_price
            exit_value = shares * exit_price
            pnl = exit_value - entry_value
            pnl_pct = (pnl / entry_value) * 100

            trade = {
                'symbol': signal.symbol,
                'pattern_type': signal.pattern_type.value,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': exit_date,
                'exit_price': exit_price,
                'stop_loss': stop_loss_price,
                'target_price': target_price,
                'shares': shares,
                'position_size': position_size,
                'entry_value': entry_value,
                'exit_value': exit_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'trade_duration': trade_duration,
                'confidence': signal.confidence,
                'risk_level': signal.risk_level,
                'expected_duration': signal.expected_duration
            }

            return trade

        except Exception as e:
            logger.error(f"Error simulating trade: {e}")
            return None

    def _calculate_sharpe_based_size(self, signal: PatternSignal, portfolio_value: float) -> float:
        """Calculate position size based on pattern Sharpe ratio"""
        # Get historical Sharpe ratio for this pattern type
        pattern_sharpe = self.sharpe_ratios.get(signal.pattern_type.value, 0.5)

        # Base position size is 10% of portfolio
        base_size = portfolio_value * 0.1

        # Scale by Sharpe ratio (max 2x for high Sharpe, min 0.1x for low Sharpe)
        sharpe_multiplier = max(0.1, min(2.0, pattern_sharpe))

        return base_size * sharpe_multiplier

    def analyze_pattern_performance(self, trades: List[Dict]) -> Dict:
        """Analyze performance by pattern type"""
        pattern_trades = {}

        for trade in trades:
            pattern = trade['pattern_type']
            if pattern not in pattern_trades:
                pattern_trades[pattern] = []
            pattern_trades[pattern].append(trade)

        performance_summary = {}
        for pattern_type, pattern_trades_list in pattern_trades.items():
            if not pattern_trades_list:
                continue

            pnl_list = [t['pnl'] for t in pattern_trades_list]
            returns = [t['pnl_pct']/100 for t in pattern_trades_list]

            # Calculate metrics
            total_trades = len(pattern_trades_list)
            winning_trades = len([p for p in pnl_list if p > 0])
            losing_trades = total_trades - winning_trades

            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = np.mean([p for p in pnl_list if p > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([p for p in pnl_list if p < 0]) if losing_trades > 0 else 0
            avg_pnl = np.mean(pnl_list)

            # Calculate Sharpe ratio
            if len(returns) > 1:
                sharpe_ratio = self._calculate_sharpe_ratio(returns)
            else:
                sharpe_ratio = 0

            performance_summary[pattern_type] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_pnl': avg_pnl,
                'sharpe_ratio': sharpe_ratio,
                'total_pnl': sum(pnl_list),
                'largest_win': max(pnl_list) if pnl_list else 0,
                'largest_loss': min(pnl_list) if pnl_list else 0
            }

            # Update master Sharpe ratios
            self.sharpe_ratios[pattern_type] = sharpe_ratio

        return performance_summary

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0

        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe

    def run_single_stock_backtest(self, symbol: str) -> Tuple[List[Dict], Dict]:
        """Run backtest for a single stock"""
        logger.info(f"Running backtest for {symbol}...")

        # Fetch data
        stock_data = self.fetch_stock_data(symbol, self.start_date, self.end_date)
        if stock_data.empty:
            return [], {}

        # Run pattern detection
        signals = self.run_pattern_detection(stock_data, symbol)

        # Simulate trades
        trades = []
        for signal in signals:
            # Filter signals within date range
            if signal.timestamp < self.start_date or signal.timestamp > self.end_date:
                continue

            trade = self.simulate_trade(signal, stock_data, signal.timestamp, self.initial_capital)
            if trade:
                trades.append(trade)

        logger.info(f"Completed {symbol}: {len(trades)} trades simulated")
        return trades, {}

    def run_full_backtest(self) -> Dict:
        """Run backtest on all top 50 stocks"""
        logger.info("Starting comprehensive backtest...")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Initial capital: ${self.initial_capital:,.0f}")

        top_stocks = self.get_top_50_stocks()
        all_trades = []
        pattern_performance = {}

        # Run backtest for each stock
        for symbol in top_stocks[:10]:  # Start with 10 stocks for testing
            logger.info(f"Processing {symbol}...")

            trades, perf = self.run_single_stock_backtest(symbol)
            all_trades.extend(trades)

            # Update pattern performance
            stock_perf = self.analyze_pattern_performance(trades)
            for pattern, metrics in stock_perf.items():
                if pattern not in pattern_performance:
                    pattern_performance[pattern] = []
                pattern_performance[pattern].append(metrics)

        # Aggregate pattern performance
        aggregated_performance = {}
        for pattern, metrics_list in pattern_performance.items():
            if not metrics_list:
                continue

            # Aggregate metrics
            total_trades = sum(m['total_trades'] for m in metrics_list)
            winning_trades = sum(m['winning_trades'] for m in metrics_list)
            losing_trades = sum(m['losing_trades'] for m in metrics_list)

            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_pnl = np.mean([m['avg_pnl'] for m in metrics_list])
            sharpe_ratio = np.mean([m['sharpe_ratio'] for m in metrics_list])

            aggregated_performance[pattern] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'sharpe_ratio': sharpe_ratio,
                'total_pnl': sum(m['total_pnl'] for m in metrics_list)
            }

        # Generate summary statistics
        summary = {
            'total_stocks_analyzed': len(top_stocks[:10]),
            'total_trades': len(all_trades),
            'total_pnl': sum(t['pnl'] for t in all_trades),
            'win_rate': len([t for t in all_trades if t['pnl'] > 0]) / len(all_trades) if all_trades else 0,
            'avg_trade_pnl': np.mean([t['pnl'] for t in all_trades]) if all_trades else 0,
            'pattern_performance': aggregated_performance,
            'all_trades': all_trades,
            'backtest_period': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d'),
                'years': (self.end_date - self.start_date).days / 365.25
            }
        }

        logger.info(f"Backtest completed: {summary['total_trades']} trades across {summary['total_stocks_analyzed']} stocks")
        logger.info(f"Total PnL: ${summary['total_pnl']:,.0f}")
        logger.info(f"Win Rate: {summary['win_rate']:.2%}")

        return summary

    def save_results(self, results: Dict, filename: str = None) -> str:
        """Save backtest results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"

        # Convert datetime objects to strings for JSON serialization
        serializable_results = self._make_json_serializable(results)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Also save as CSV for trades
        if 'all_trades' in results and results['all_trades']:
            trades_df = pd.DataFrame(results['all_trades'])
            trades_df.to_csv(f"backtest_trades_{timestamp}.csv", index=False)

        logger.info(f"Results saved to {filename}")
        return filename

    def _make_json_serializable(self, obj):
        """Convert datetime objects to strings for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return obj

def main():
    """Main function to run the comprehensive backtest"""
    print("🔍 Comprehensive Pattern Detection Backtesting System")
    print("=" * 60)

    # Initialize backtester
    backtester = PatternBacktester(
        start_date="2018-01-01",  # Start from 2018 for testing
        end_date="2023-12-31",    # End in 2023 for testing
        initial_capital=1000000,   # $1M starting capital
        position_size_method="sharpe"
    )

    # Run backtest
    results = backtester.run_full_backtest()

    # Display results
    print(f"\n📊 BACKTEST RESULTS SUMMARY")
    print("=" * 40)
    print(f"Total Stocks Analyzed: {results['total_stocks_analyzed']}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Total PnL: ${results['total_pnl']:,.0f}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Average Trade PnL: ${results['avg_trade_pnl']:,.0f}")
    print(f"Backtest Period: {results['backtest_period']['years']:.1f} years")

    print(f"\n🎯 PATTERN PERFORMANCE SUMMARY")
    print("-" * 40)
    for pattern, metrics in results['pattern_performance'].items():
        print(f"{pattern.upper()}:")
        print(f"  Trades: {metrics['total_trades']}, Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Avg PnL: ${metrics['avg_pnl']:,.0f}, Sharpe: {metrics['sharpe_ratio']:.3f}")
        print(f"  Total PnL: ${metrics['total_pnl']:,.0f}")

    # Save results
    results_file = backtester.save_results(results)
    print(f"\n💾 Results saved to: {results_file}")

    return results

if __name__ == "__main__":
    results = main()