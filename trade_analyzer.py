#!/usr/bin/env python3
"""
Advanced Trade Analyzer with Pattern Performance Metrics
Calculates win ratios, Sharpe ratios, and generates detailed trade tables
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.interfaces import PatternConfig, PatternEngine, DataValidator
from core.market_data import MarketDataIngestor
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

class TradeAnalyzer:
    """Advanced trade analyzer with performance metrics"""

    def __init__(self, start_date: str = "2020-01-01", end_date: str = None,
                 initial_capital: float = 100000):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        self.initial_capital = initial_capital
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

    def calculate_sharpe_ratio(self, returns: List[float], periods_per_year: int = 252) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - (self.risk_free_rate / periods_per_year)

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(periods_per_year)
        return sharpe

    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {}

        # Extract PnL and returns - use the correct column names from DataFrame
        pnl_list = [t.get('PnL', t.get('Pnl', 0)) for t in trades]
        returns_pct = [t.get('PnL_Percent', t.get('Pnl_Percent', 0)) / 100 for t in trades]

        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([p for p in pnl_list if p > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # PnL metrics
        total_pnl = sum(pnl_list)
        avg_pnl = np.mean(pnl_list)
        median_pnl = np.median(pnl_list)
        largest_win = max(pnl_list) if pnl_list else 0
        largest_loss = min(pnl_list) if pnl_list else 0

        # Win/loss metrics
        avg_win = np.mean([p for p in pnl_list if p > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([p for p in pnl_list if p < 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Risk metrics
        max_drawdown = self.calculate_max_drawdown(trades)
        volatility = np.std(returns_pct) * np.sqrt(252) if len(returns_pct) > 1 else 0

        # Performance ratios
        sharpe_ratio = self.calculate_sharpe_ratio(returns_pct)

        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'median_pnl': median_pnl,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': sharpe_ratio / (max_drawdown + 0.001)  # Avoid division by zero
        }

        return metrics

    def calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trade sequence"""
        if not trades:
            return 0.0

        # Sort trades by date
        sorted_trades = sorted(trades, key=lambda x: x.get('Entry_Date', x.get('entry_date', '1970-01-01')))

        # Calculate cumulative PnL
        cumulative_pnl = 0
        max_pnl = 0
        max_drawdown = 0

        for trade in sorted_trades:
            cumulative_pnl += trade.get('PnL', trade.get('pnl', 0))
            max_pnl = max(max_pnl, cumulative_pnl)
            drawdown = max_pnl - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def analyze_pattern_performance(self, trades: List[Dict]) -> Dict:
        """Analyze performance by pattern type"""
        pattern_metrics = {}

        # Group trades by pattern type
        pattern_groups = {}
        for trade in trades:
            pattern = trade.get('Pattern_Type', trade.get('pattern_type', 'unknown'))
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(trade)

        # Calculate metrics for each pattern
        for pattern, pattern_trades in pattern_groups.items():
            metrics = self.calculate_performance_metrics(pattern_trades)
            metrics['trade_count'] = len(pattern_trades)
            pattern_metrics[pattern] = metrics

        return pattern_metrics

    def generate_position_sizes(self, pattern_metrics: Dict, base_size: float = 10000) -> Dict:
        """Generate position sizes based on Sharpe ratios"""
        position_sizes = {}

        for pattern, metrics in pattern_metrics.items():
            sharpe = metrics.get('sharpe_ratio', 0)

            # Scale position size based on Sharpe ratio
            # Higher Sharpe = larger position (max 3x, min 0.1x)
            sharpe_multiplier = max(0.1, min(3.0, sharpe))
            position_size = base_size * sharpe_multiplier

            position_sizes[pattern] = {
                'position_size': position_size,
                'sharpe_ratio': sharpe,
                'multiplier': sharpe_multiplier
            }

        return position_sizes

    def create_detailed_trade_table(self, trades: List[Dict]) -> pd.DataFrame:
        """Create detailed trade table with all metrics"""
        if not trades:
            return pd.DataFrame()

        trade_data = []
        for trade in trades:
            trade_data.append({
                'Symbol': trade['symbol'],
                'Pattern_Type': trade['pattern_type'],
                'Entry_Date': trade['entry_date'],
                'Exit_Date': trade['exit_date'],
                'Days_Held': trade['trade_duration'],
                'Entry_Price': trade['entry_price'],
                'Exit_Price': trade['exit_price'],
                'Stop_Loss': trade['stop_loss'],
                'Target_Price': trade['target_price'],
                'Shares': trade['shares'],
                'Position_Size': trade['position_size'],
                'Entry_Value': trade['entry_value'],
                'Exit_Value': trade['exit_value'],
                'PnL': trade['pnl'],
                'PnL_Percent': trade['pnl_pct'],
                'Exit_Reason': trade['exit_reason'],
                'Confidence': trade['confidence'],
                'Risk_Level': trade['risk_level']
            })

        df = pd.DataFrame(trade_data)

        # Add derived columns
        df['Return_Percent'] = (df['Exit_Value'] / df['Entry_Value'] - 1) * 100
        df['Win_Loss'] = df['PnL'].apply(lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'Breakeven')

        return df

    def run_pattern_analysis(self, symbol: str, config: PatternConfig = None) -> Dict:
        """Run pattern analysis for a single symbol and return results as DataFrame"""
        if config is None:
            config = PatternConfig(min_confidence=0.5)

        logger.info(f"Analyzing {symbol}...")

        try:
            # Fetch data
            ingestor = MarketDataIngestor()
            data = ingestor.fetch_stock_data(symbol, period="2y", interval="1d")

            if data.empty:
                logger.warning(f"No data for {symbol}")
                return {}

            # Clean data
            cleaned_data = DataValidator.clean_ohlc_data(data)

            # Run pattern detection
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
            signals = engine.detect_patterns(cleaned_data, symbol)

            # Create trades DataFrame directly
            trades_data = []
            for signal in signals:
                try:
                    # Date filtering - handle timezone-aware timestamps
                    signal_date = signal.timestamp

                    # Ensure both dates are timezone-aware for comparison
                    if hasattr(signal_date, 'tz') and signal_date.tz is not None:
                        if hasattr(self.start_date, 'tz') and self.start_date.tz is None:
                            self.start_date = self.start_date.tz_localize(signal_date.tz)
                        if hasattr(self.end_date, 'tz') and self.end_date.tz is None:
                            self.end_date = self.end_date.tz_localize(signal_date.tz)

                    if signal_date < self.start_date or signal_date > self.end_date:
                        continue

                    # Check if entry date exists in data
                    if signal_date not in cleaned_data.index:
                        continue

                    # Get entry price
                    entry_price = cleaned_data.loc[signal_date, 'close']
                    position_size = self.initial_capital * 0.1  # 10% position
                    shares = position_size / entry_price

                    # Find exit conditions
                    future_dates = cleaned_data.index[cleaned_data.index > signal_date]
                    if len(future_dates) == 0:
                        continue
                except Exception as e:
                    logger.warning(f"Error processing signal: {e}")
                    continue

                exit_price = None
                exit_date = None
                exit_reason = None
                days_held = 0

                # Check each subsequent day for exit conditions
                for date in future_dates:
                    try:
                        days_held = (date - signal_date).days

                        # Stop loss check
                        if cleaned_data.loc[date, 'low'] <= signal.stop_loss:
                            exit_price = signal.stop_loss
                            exit_date = date
                            exit_reason = "stop_loss"
                            break

                        # Target check
                        if cleaned_data.loc[date, 'high'] >= signal.target_price:
                            exit_price = signal.target_price
                            exit_date = date
                            exit_reason = "target_reached"
                            break

                        # Max holding period
                        if days_held > 30:
                            exit_price = cleaned_data.loc[date, 'close']
                            exit_date = date
                            exit_reason = "max_holding"
                            break
                    except Exception as e:
                        logger.warning(f"Error processing exit conditions: {e}")
                        continue

                if exit_price is None:
                    # No exit condition met within timeframe
                    exit_price = cleaned_data.iloc[-1]['close']
                    exit_date = cleaned_data.index[-1]
                    exit_reason = "end_of_period"

                # Calculate metrics
                entry_value = shares * entry_price
                exit_value = shares * exit_price
                pnl = exit_value - entry_value
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

                # Add trade data
                trades_data.append({
                    'Symbol': symbol,
                    'Pattern_Type': signal.pattern_type.value,
                    'Entry_Date': signal_date.strftime('%Y-%m-%d') if signal_date else None,
                    'Exit_Date': exit_date.strftime('%Y-%m-%d') if exit_date else None,
                    'Days_Held': int(days_held) if days_held is not None else 0,
                    'Entry_Price': float(entry_price) if entry_price is not None else 0.0,
                    'Exit_Price': float(exit_price) if exit_price is not None else 0.0,
                    'Stop_Loss': float(signal.stop_loss) if signal.stop_loss is not None else 0.0,
                    'Target_Price': float(signal.target_price) if signal.target_price is not None else 0.0,
                    'Shares': float(shares) if shares is not None else 0.0,
                    'Position_Size': float(position_size) if position_size is not None else 0.0,
                    'Entry_Value': float(entry_value) if entry_value is not None else 0.0,
                    'Exit_Value': float(exit_value) if exit_value is not None else 0.0,
                    'PnL': float(pnl) if pnl is not None else 0.0,
                    'PnL_Percent': float(pnl_pct) if pnl_pct is not None else 0.0,
                    'Exit_Reason': exit_reason,
                    'Confidence': float(signal.confidence) if signal.confidence is not None else 0.0,
                    'Risk_Level': signal.risk_level if signal.risk_level else 'unknown',
                    'Expected_Duration': signal.expected_duration
                })

            # Convert to DataFrame
            try:
                trade_table = pd.DataFrame(trades_data)
                logger.info(f"Successfully created DataFrame with {len(trades_data)} trades")
                logger.info(f"DataFrame columns: {trade_table.columns.tolist()}")
                logger.info(f"Sample trade data: {trades_data[0] if trades_data else 'No trades'}")
            except Exception as e:
                logger.error(f"Error creating DataFrame: {e}")
                logger.error(f"Trades data: {trades_data}")
                return {}

            # Calculate metrics
            try:
                trades_list = trade_table.to_dict('records') if not trade_table.empty else []
                logger.info(f"Converting to records: {len(trades_list)} records")

                # Test first record
                if trades_list:
                    logger.info(f"First record keys: {list(trades_list[0].keys())}")

                all_metrics = self.calculate_performance_metrics(trades_list)
                logger.info(f"Performance metrics calculated successfully")
                pattern_metrics = self.analyze_pattern_performance(trades_list)
                logger.info(f"Pattern metrics calculated successfully")
                position_sizes = self.generate_position_sizes(pattern_metrics)
                logger.info(f"Position sizes calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
                import traceback
                traceback.print_exc()
                raise

            results = {
                'symbol': symbol,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'date_range': {
                    'start': self.start_date.strftime('%Y-%m-%d'),
                    'end': self.end_date.strftime('%Y-%m-%d')
                },
                'total_signals': len(signals),
                'total_trades': len(trades_data),
                'overall_metrics': all_metrics,
                'pattern_metrics': pattern_metrics,
                'position_sizes': position_sizes,
                'trade_table': trade_table
            }

            try:
                logger.info(f"Analysis complete for {symbol}: {len(trades_data)} trades, Total PnL: ${all_metrics.get('total_pnl', 0):,.0f}")
            except Exception as e:
                logger.info(f"Analysis complete for {symbol}: {len(trades_data)} trades, Total PnL: calculation error")
            return results

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {}

    def run_multi_symbol_analysis(self, symbols: List[str], output_file: str = None) -> Dict:
        """Run analysis on multiple symbols and save results to CSV"""
        logger.info(f"Running multi-symbol analysis on {len(symbols)} symbols")

        all_results = []
        combined_trades = []
        pattern_performance_summary = {}

        for symbol in symbols:
            try:
                results = self.run_pattern_analysis(symbol)
                if results:
                    all_results.append(results)
                    if 'trade_table' in results and not results['trade_table'].empty:
                        combined_trades.append(results['trade_table'])
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        # Combine all trades
        if combined_trades:
            combined_trades_df = pd.concat(combined_trades, ignore_index=True)

            # Calculate metrics
            all_trades_list = combined_trades_df.to_dict('records')
            overall_metrics = self.calculate_performance_metrics(all_trades_list)
            pattern_analysis = self.analyze_pattern_performance(all_trades_list)

            # Create pattern performance summary
            for pattern, metrics in pattern_analysis.items():
                if metrics.get('trade_count', 0) > 0:
                    pattern_performance_summary[pattern] = {
                        'total_trades': metrics['trade_count'],
                        'win_rate': f"{metrics['win_rate']:.2%}",
                        'sharpe_ratio': f"{metrics['sharpe_ratio']:.3f}",
                        'avg_pnl': f"${metrics['avg_pnl']:,.0f}",
                        'total_pnl': f"${metrics['total_pnl']:,.0f}"
                    }
        else:
            combined_trades_df = pd.DataFrame()
            overall_metrics = {}
            pattern_analysis = {}
            pattern_performance_summary = {}

        # Combined results
        final_results = {
            'analysis_summary': {
                'symbols_analyzed': len(symbols),
                'successful_analyses': len(all_results),
                'total_trades': len(combined_trades) if combined_trades else 0,
                'overall_metrics': overall_metrics,
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            },
            'individual_results': all_results,
            'combined_trades': combined_trades_df,
            'pattern_analysis': pattern_analysis,
            'pattern_performance_summary': pattern_performance_summary
        }

        # Save results to CSV and other formats
        if output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save detailed trade table
            if not combined_trades_df.empty:
                trades_file = f"{output_file}_trades_{timestamp}.csv"
                combined_trades_df.to_csv(trades_file, index=False)
                final_results['trades_file'] = trades_file
                logger.info(f"Trades saved to {trades_file}")

            # Create summary reports
            if pattern_performance_summary:
                pattern_summary_df = pd.DataFrame.from_dict(pattern_performance_summary, orient='index')
                pattern_summary_df.index.name = 'Pattern_Type'
                pattern_summary_df = pattern_summary_df.reset_index()

                pattern_file = f"{output_file}_pattern_summary_{timestamp}.csv"
                pattern_summary_df.to_csv(pattern_file, index=False)
                final_results['pattern_summary_file'] = pattern_file
                logger.info(f"Pattern summary saved to {pattern_file}")

            # Save overall summary
            summary_data = {
                'Metric': ['Total Symbols Analyzed', 'Successful Analyses', 'Total Trades', 'Total PnL', 'Win Rate', 'Sharpe Ratio'],
                'Value': [
                    len(symbols),
                    len(all_results),
                    overall_metrics.get('total_trades', 0),
                    f"${overall_metrics.get('total_pnl', 0):,.0f}",
                    f"{overall_metrics.get('win_rate', 0):.2%}",
                    f"{overall_metrics.get('sharpe_ratio', 0):.3f}"
                ]
            }

            summary_df = pd.DataFrame(summary_data)
            summary_file = f"{output_file}_overall_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            final_results['summary_file'] = summary_file
            logger.info(f"Overall summary saved to {summary_file}")

        return final_results

def main():
    """Main function to run the trade analyzer"""
    print("🔍 Advanced Trade Analyzer")
    print("=" * 50)

    # Initialize analyzer
    analyzer = TradeAnalyzer(
        start_date="2023-01-01",
        end_date="2024-12-31",
        initial_capital=100000
    )

    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    # Run analysis
    results = analyzer.run_multi_symbol_analysis(
        test_symbols,
        output_file="trade_analysis_results"
    )

    # Display summary
    summary = results['analysis_summary']
    print(f"\n📊 ANALYSIS SUMMARY")
    print("=" * 40)
    print(f"Symbols Analyzed: {summary['symbols_analyzed']}")
    print(f"Successful Analyses: {summary['successful_analyses']}")
    print(f"Total Trades: {summary['total_trades']}")

    if summary['overall_metrics']:
        metrics = summary['overall_metrics']
        print(f"Total PnL: ${metrics['total_pnl']:,.0f}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: ${metrics['max_drawdown']:,.0f}")

    # Show pattern performance
    print(f"\n🎯 PATTERN PERFORMANCE")
    print("-" * 40)
    for pattern, metrics in results['pattern_analysis'].items():
        if metrics.get('trade_count', 0) > 0:
            print(f"{pattern.upper()}: {metrics['trade_count']} trades, "
                  f"Win Rate: {metrics['win_rate']:.2%}, "
                  f"Sharpe: {metrics['sharpe_ratio']:.3f}")

    # Show sample trades
    if not results['combined_trades'].empty:
        print(f"\n📋 SAMPLE TRADES")
        print("-" * 40)
        sample_trades = results['combined_trades'].head(5)
        for _, trade in sample_trades.iterrows():
            print(f"{trade['Symbol']} - {trade['Pattern_Type']}")
            print(f"  Entry: ${trade['Entry_Price']:.2f} → Exit: ${trade['Exit_Price']:.2f}")
            print(f"  PnL: ${trade['PnL']:,.0f} ({trade['PnL_Percent']:.1f}%)")
            print(f"  Days: {trade['Days_Held']} - {trade['Exit_Reason']}")
            print()

    return results

if __name__ == "__main__":
    results = main()