# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based financial trading pattern detection system designed to identify various chart patterns in market data. The project provides a framework for detecting trading patterns like VCP breakouts, flag patterns, cup and handle patterns, and other technical analysis formations.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install with development dependencies
pip install -r requirements.txt pytest-cov matplotlib seaborn plotly
```

### Testing
```bash
# Run all tests (when implemented)
pytest src/tests/

# Run tests with coverage
pytest --cov=src src/tests/

# Run specific test modules
pytest src/tests/detectors/
pytest src/tests/utils/
```

### Code Quality
```bash
# Run linting (if flake8 or similar is added)
flake8 src/

# Run type checking (if mypy is added)
mypy src/
```

## Architecture Overview

### Core Components

The system is built around several key classes in `src/core/interfaces.py`:

- **PatternSignal**: Dataclass representing detected trading signals with confidence scores, entry/exit prices, and metadata
- **PatternDetector**: Abstract base class for all pattern detection implementations
- **DataValidator**: Validates input price data for consistency and completeness  
- **PatternEngine**: Main orchestration engine that runs multiple detectors and combines results

### Design Patterns

- **Abstract Factory**: Creating different pattern detectors
- **Strategy Pattern**: Each detector implements a specific detection algorithm
- **Template Method**: Base detector defines the detection workflow

### Supported Pattern Types

The framework supports these pattern types:
- VCP_BREAKOUT - Volatility Contraction Pattern Breakout
- FLAG_PATTERN - Flag/Consolidation Pattern  
- CUP_HANDLE - Cup and Handle Pattern
- ASCENDING_TRIANGLE - Ascending Triangle Pattern
- DOUBLE_BOTTOM - Double Bottom Pattern
- WEDGE_PATTERN - Wedge Pattern

### Data Requirements

All pattern detectors expect OHLCV data in pandas DataFrame format with columns:
- `open`, `high`, `low`, `close`, `volume`

The `DataValidator` class ensures data integrity before pattern detection.

## Current Project Status

This project is in **advanced development phase**:
- ✅ Core interfaces and architecture implemented
- ✅ 12 trading pattern detectors fully implemented
- ✅ Market data ingestion layer with Yahoo Finance integration
- ✅ Comprehensive backtesting system for 50 stocks
- ✅ Trade analyzer with performance metrics and Sharpe ratio calculations
- ✅ Complete CSV export functionality for trade data
- ✅ 2-year backtesting with position sizing based on pattern performance
- ❌ Unit tests need to be written
- ❌ Documentation needs to be improved

## Recently Completed Features

### Data Ingestion and Market Data
- ✅ **MarketDataIngestor**: Comprehensive data fetching from Yahoo Finance
- ✅ **YahooFinanceFetcher**: Real-time market data ingestion
- ✅ **DataValidator**: Enhanced OHLC data validation and cleaning
- ✅ **Timezone handling**: Fixed timezone-aware vs naive timestamp issues

### Pattern Detection System
- ✅ **12 Pattern Detectors**: VCP Breakout, Flag, Cup & Handle, Double Bottom, Head & Shoulders, Rounding Bottom, Ascending/Descending Triangles, Rising/Falling Wedges
- ✅ **PatternEngine**: Orchestrates multiple detectors and combines results
- ✅ **PatternSignal**: Comprehensive signal data with confidence scores, entry/exit prices

### Backtesting and Performance Analysis
- ✅ **Comprehensive Backtest Engine**: Runs on top 50 stocks with 2-year lookback
- ✅ **Performance Metrics**: Win ratios, Sharpe ratios, max drawdown, profit factors
- ✅ **Position Sizing**: Dynamic position sizing based on pattern Sharpe ratios
- ✅ **Trade Simulation**: Realistic trade execution with stop losses and targets

### Trade Analysis and Reporting
- ✅ **Trade Analyzer**: Complete pandas DataFrame-based analysis
- ✅ **CSV Export**: Multiple report formats (trades, pattern summary, overall metrics)
- ✅ **Real-time Analysis**: Currently analyzing 5 stocks with 449 total trades

### Key Technical Achievements
- ✅ **Fixed timezone issues**: Resolved datetime comparison problems
- ✅ **Data validation**: Robust OHLC relationship validation
- ✅ **Error handling**: Comprehensive error handling and logging
- ✅ **Type safety**: Fixed type checker issues and proper data type conversion
- ✅ **Data cleaning**: Automatic handling of missing/null values

## Development Guidelines

When working on this codebase:

1. **Follow the abstract base pattern**: All new detectors must inherit from `PatternDetector`
2. **Implement required methods**: `detect()` and `get_required_columns()`
3. **Use the DataValidator**: Always validate input data before processing
4. **Return PatternSignal objects**: Signals should include confidence scores and metadata
5. **Write tests**: Implement comprehensive tests for each detector
6. **Document patterns**: Add clear documentation for each pattern detection algorithm

## Key Dependencies

- **Data Analysis**: numpy, pandas, scipy
- **Machine Learning**: scikit-learn  
- **Visualization**: matplotlib, seaborn, plotly
- **Financial Analysis**: ta-lib (Technical Analysis Library)
- **Database**: kdbpy (KDB+ integration)
- **Testing**: pytest, pytest-cov

## File Structure

```
.
├── src/
│   ├── core/
│   │   └── interfaces.py        # Core abstract classes and data structures
│   ├── detectors/               # 12 pattern detector implementations
│   ├── data/
│   │   └── market_data_ingestion.py    # Market data fetching and validation
│   └── tests/                   # Test files (currently empty)
├── utils/
│   ├── market_data_client.py   # Alternative market data client (REVIEW NEEDED)
│   └── other_utils.py          # Additional utility functions
├── trade_analyzer.py           # Advanced trade analysis with CSV export
├── comprehensive_backtest.py   # Full backtesting system for 50 stocks
├── final_demonstration.py      # Complete system demonstration
├── testing/                    # Test and demo scripts
│   ├── quick_backtest.py
│   ├── real_aapl_analysis.py
│   └── fetch_top_50_stocks.py
├── market_data/                # Parquet storage for fetched data
├── backtest_results*.json      # Backtest results output
├── trade_analysis_results*.csv # Trade analysis reports output
└── requirements.txt            # Project dependencies
```

## Architecture Issues to Address

### Duplicate Market Data Modules
- **Issue**: Both `utils/market_data_client.py` and `src/data/market_data_ingestion.py` exist
- **Action**: Consolidate into single, unified data ingestion architecture
- **Decision Needed**: Which module to keep and how to merge functionality

### Pending Tasks
- ✅ Complete trade analyzer with pandas DataFrame and CSV export (COMPLETED)
- ⚠️ Fix duplicate market data ingestion modules (IN PROGRESS)
- 🔄 Clean up data ingestion architecture (NEXT)
- 🔄 Write comprehensive unit tests for all components