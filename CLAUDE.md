# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive Python-based financial trading pattern detection system with both powerful backend engine and user-friendly web frontend. The system identifies various chart patterns in market data through an intuitive Streamlit interface or direct API access, supporting 12+ technical analysis patterns with real-time data fetching and intelligent caching.

## Development Commands

### Installation and Setup

#### Quick Start with Web Frontend
```bash
# Clone the repository
git clone https://github.com/tradingpatterns/trading-pattern-detector.git
cd trading-pattern-detector

# Install dependencies
pip install -r requirements.txt

# Install Streamlit frontend dependencies
cd frontend
pip install -r requirements.txt
cd ..

# Launch the web application
streamlit run frontend/app.py
```

#### Backend-Only Installation
```bash
# Clone the repository
git clone https://github.com/tradingpatterns/trading-pattern-detector.git
cd trading-pattern-detector

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

#### Development Installation
```bash
git clone https://github.com/tradingpatterns/trading-pattern-detector.git
cd trading-pattern-detector
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run frontend tests
pytest frontend/tests/

# Run specific test modules
pytest src/tests/detectors/
pytest src/tests/utils/
pytest frontend/tests/test_data_manager.py
pytest frontend/tests/test_integration.py
pytest frontend/tests/test_ui_components.py
```

### Code Quality
```bash
# Format code
black src/ frontend/ tests/

# Sort imports
isort src/ frontend/ tests/

# Lint code
flake8 src/ frontend/ tests/

# Type checking
mypy src/ frontend/
```

## Architecture Overview

### Core Components

The system consists of two main layers:

**Backend Engine (`src/core/interfaces.py`)**:
- **PatternSignal**: Dataclass representing detected trading signals with confidence scores, entry/exit prices, and metadata
- **PatternDetector**: Abstract base class for all pattern detection implementations
- **DataValidator**: Validates input price data for consistency and completeness
- **PatternEngine**: Main orchestration engine that runs multiple detectors and combines results

**Frontend Layer (`frontend/`)**:
- **DataManager**: Handles market data fetching, parquet storage, and intelligent caching
- **PatternDetectionEngine**: Integration layer connecting frontend to backend pattern detection
- **UI Components**: Input forms, results display, pattern selection with Streamlit interface
- **Configuration**: Settings management and application configuration

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

The `DataManager` class handles data fetching, parquet storage, and intelligent caching automatically. The `DataValidator` class ensures data integrity before pattern detection.

### Frontend Data Management

**Parquet Storage**: Market data is stored in `data/` subfolder, one file per symbol
**Smart Caching**: Automatic data fetching from Yahoo Finance with TTL-based caching
**Data Validation**: OHLC relationship validation and cleaning to prevent invalid data
**Append-Only Updates**: New data is fetched and appended to existing parquet files

## Current Project Status

This project is in **production-ready phase**:
- ✅ **Complete Frontend**: Streamlit web application with interactive UI
- ✅ **Backend Pattern Engine**: 12+ technical analysis pattern detectors
- ✅ **Data Management**: Parquet storage with intelligent caching and auto-updates
- ✅ **Market Data Integration**: Yahoo Finance API with automatic data fetching
- ✅ **Comprehensive Testing**: Full test suite for frontend and backend components
- ✅ **Performance Analysis**: Trade analyzer with Sharpe ratios and backtesting
- ✅ **Export Capabilities**: CSV/JSON export for results and analysis
- ✅ **User Interface**: Interactive charts, pattern selection, and real-time feedback
- ✅ **Documentation**: Complete README and API documentation

## Recently Completed Features

### Frontend Architecture and Web Interface
- ✅ **Streamlit Web Application**: Interactive UI for pattern detection with real-time feedback
- ✅ **User Input Forms**: Stock symbol, date range selection, pattern selection interface
- ✅ **Results Display**: Interactive charts, performance metrics, and comprehensive signal visualization
- ✅ **Pattern Selection**: 12+ technical pattern categories with detailed descriptions
- ✅ **Export Functionality**: CSV and JSON export for results and analysis data

### Data Management Layer
- ✅ **DataManager**: Handles market data fetching, parquet storage, and intelligent caching
- ✅ **Parquet Storage**: Efficient local storage with automatic updates, one file per symbol
- ✅ **Smart Caching**: Memory caching with TTL, automatic data fetching from Yahoo Finance
- ✅ **Data Validation**: OHLC relationship validation and cleaning to prevent invalid data
- ✅ **Append-Only Updates**: New data fetched and appended to existing parquet files

### Integration Layer
- ✅ **PatternDetectionEngine**: Async integration between frontend and backend pattern detection
- ✅ **Progress Manager**: Real-time progress tracking for pattern detection operations
- ✅ **Error Handler**: Centralized error management with user-friendly feedback
- ✅ **Configuration Management**: Settings management and application configuration

### Backend Pattern Detection System
- ✅ **12 Pattern Detectors**: VCP Breakout, Flag, Cup & Handle, Double Bottom, Head & Shoulders, Rounding Bottom, Ascending/Descending Triangles, Rising/Falling Wedges
- ✅ **PatternEngine**: Orchestrates multiple detectors and combines results
- ✅ **PatternSignal**: Comprehensive signal data with confidence scores, entry/exit prices

### Testing Framework
- ✅ **Comprehensive Test Suite**: Unit tests for all frontend and backend components
- ✅ **Async Testing**: Proper async test patterns for data operations
- ✅ **UI Component Tests**: Streamlit component testing with proper mocking
- ✅ **Integration Tests**: End-to-end testing of complete workflows
- ✅ **Data Management Tests**: Parquet storage, caching, and validation testing

## Development Guidelines

When working on this codebase:

1. **Follow the abstract base pattern**: All new detectors must inherit from `PatternDetector`
2. **Implement required methods**: `detect()` and `get_required_columns()`
3. **Use the DataValidator**: Always validate input data before processing
4. **Return PatternSignal objects**: Signals should include confidence scores and metadata
5. **Write tests**: Implement comprehensive tests for each detector
6. **Document patterns**: Add clear documentation for each pattern detection algorithm

## Key Dependencies

### Core Dependencies
- **Data Analysis**: numpy, pandas, scipy
- **Machine Learning**: scikit-learn
- **Financial Analysis**: ta-lib (Technical Analysis Library)
- **Database**: kdbpy (KDB+ integration)
- **Testing**: pytest, pytest-cov

### Frontend Dependencies
- **Web Framework**: streamlit>=1.28.0
- **Data Storage**: pyarrow>=12.0.0 (parquet files)
- **Market Data**: yfinance>=0.2.0, aiohttp>=3.8.0
- **Visualization**: plotly>=5.15.0
- **Async Operations**: asyncio, aiohttp
- **Testing**: pytest>=7.0.0, pytest-asyncio>=0.21.0, pytest-mock>=3.10.0

## File Structure

```
.
├── src/
│   ├── core/
│   │   └── interfaces.py        # Core abstract classes and data structures
│   ├── detectors/               # 12 pattern detector implementations
│   ├── data/
│   │   └── market_data_ingestion.py    # Market data fetching and validation
│   └── tests/                   # Test files
├── frontend/                    # Streamlit web application
│   ├── app.py                   # Main Streamlit application
│   ├── components/              # UI components
│   │   └── ui/
│   │       ├── input_form.py   # User input forms and validation
│   │       ├── results_display.py  # Results visualization and export
│   │       └── pattern_selector.py  # Pattern selection interface
│   ├── data/                   # Data management layer
│   │   └── manager.py          # DataManager for parquet storage and caching
│   ├── integration/            # Integration layer
│   │   ├── engine.py           # Pattern detection integration
│   │   ├── progress_manager.py # Progress tracking for long operations
│   │   └── error_handler.py    # Centralized error management
│   ├── config/                 # Configuration and settings
│   │   └── settings.py         # Application settings and constants
│   ├── tests/                  # Frontend test suite
│   │   ├── test_ui_components.py
│   │   ├── test_integration.py
│   │   └── test_data_manager.py
│   └── requirements.txt         # Frontend dependencies
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
├── data/                       # Parquet storage for fetched data (frontend)
├── market_data/                # Parquet storage for fetched data (backend)
├── backtest_results*.json      # Backtest results output
├── trade_analysis_results*.csv # Trade analysis reports output
└── requirements.txt            # Project dependencies
```

## Frontend Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt
cd frontend
pip install -r requirements.txt

# Launch the web application
streamlit run frontend/app.py
```

### Frontend Features
- **Interactive Interface**: User-friendly web application for pattern detection
- **Real-time Data**: Automatic market data fetching with Yahoo Finance integration
- **Smart Caching**: Efficient data management with Parquet file storage
- **Pattern Selection**: Choose from 12+ technical analysis patterns
- **Visual Analytics**: Interactive charts and comprehensive result visualization
- **Export Capabilities**: Download results in CSV, JSON formats
- **Progress Tracking**: Real-time feedback during analysis

### Frontend Development
```bash
# Run frontend tests
pytest frontend/tests/

# Format frontend code
black frontend/ tests/

# Type check frontend code
mypy frontend/
```

## Architecture Issues to Address

### Duplicate Market Data Modules
- **Issue**: Both `utils/market_data_client.py` and `src/data/market_data_ingestion.py` exist
- **Action**: Consolidate into single, unified data ingestion architecture
- **Decision Needed**: Which module to keep and how to merge functionality

### Pending Tasks
- ✅ Complete trade analyzer with pandas DataFrame and CSV export (COMPLETED)
- ✅ Build comprehensive frontend with Streamlit interface (COMPLETED)
- ✅ Implement testing framework for all components (COMPLETED)
- ⚠️ Fix duplicate market data ingestion modules (IN PROGRESS)
- 🔄 Clean up data ingestion architecture (NEXT)