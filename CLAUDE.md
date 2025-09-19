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

This project is in **early development phase**:
- ✅ Core interfaces and architecture implemented
- ✅ Abstract base classes and validation framework
- ✅ Requirements specification defined
- ❌ No concrete pattern detectors implemented
- ❌ No tests written
- ❌ No documentation or examples
- ❌ No build/deployment configuration

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
src/
├── core/
│   └── interfaces.py        # Core abstract classes and data structures
├── detectors/               # Concrete pattern detector implementations (to be created)
├── patterns/                # Pattern definitions and configurations
├── tests/                   # Test files (currently empty)
└── utils/                   # Utility functions (currently empty)
```