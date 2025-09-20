# Trading Pattern Detection System

A comprehensive Python-based financial trading pattern detection system with both powerful backend engine and user-friendly web frontend. Identify various chart patterns in market data through an intuitive Streamlit interface or direct API access.

## Features

### 🎯 Web Frontend (Streamlit)
- **Interactive Interface**: User-friendly web application for pattern detection
- **Real-time Data**: Automatic market data fetching with Yahoo Finance integration
- **Smart Caching**: Efficient data management with Parquet file storage
- **Pattern Selection**: Choose from 12+ technical analysis patterns
- **Visual Analytics**: Interactive charts and comprehensive result visualization
- **Export Capabilities**: Download results in CSV, JSON formats
- **Progress Tracking**: Real-time feedback during analysis

### 🔍 Backend Pattern Detection Engine
- **12 Technical Patterns**: Comprehensive pattern detection including:
  - VCP (Volatility Contraction Pattern) Breakout
  - Flag Patterns
  - Cup and Handle Patterns
  - Double Bottom Patterns
  - Triangle Patterns (Ascending, Descending)
  - Wedge Patterns (Rising, Falling)
  - Head and Shoulders Patterns
  - Rounding Bottom Patterns

- **Advanced Analysis Tools**:
  - Volatility Analysis (ATR, Bollinger Bands)
  - Volume Analysis (Volume Profile, OBV)
  - Trend Analysis (Moving Averages, Trend Detection)
  - Support/Resistance Level Detection

- **Professional Features**:
  - Signal aggregation and filtering
  - Risk management parameters
  - Confidence scoring system
  - Performance metrics calculation
  - Sharpe ratio analysis
  - Comprehensive test coverage
  - Well-documented API

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start with Web Frontend

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

### Backend-Only Installation

```bash
# Clone the repository
git clone https://github.com/tradingpatterns/trading-pattern-detector.git
cd trading-pattern-detector

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### Development Installation

```bash
git clone https://github.com/tradingpatterns/trading-pattern-detector.git
cd trading-pattern-detector
pip install -e ".[dev]"
```

## Quick Start

### 🌐 Web Frontend Usage

Launch the interactive web application:

```bash
streamlit run frontend/app.py
```

The web interface provides:
- **Symbol Input**: Enter any stock symbol
- **Date Range Selection**: Choose your analysis period
- **Pattern Selection**: Pick from 12+ technical patterns
- **Confidence Threshold**: Adjust detection sensitivity
- **Real-time Analysis**: Automatic data fetching and pattern detection
- **Interactive Visualizations**: Charts and detailed results
- **Export Options**: Download results in multiple formats

### 🐍 Backend API Usage

```python
from src.core.interfaces import PatternEngine, PatternConfig
from src.core.market_data import MarketDataIngestor
from detectors import (
    VCPBreakoutDetector, FlagPatternDetector,
    CupHandleDetector, DoubleBottomDetector,
    HeadAndShouldersDetector, RoundingBottomDetector,
    AscendingTriangleDetector, DescendingTriangleDetector,
    RisingWedgeDetector, FallingWedgeDetector
)
import pandas as pd

# Fetch market data automatically
ingestor = MarketDataIngestor()
data = ingestor.fetch_stock_data("AAPL", period="2y", interval="1d")

# Configure pattern detection
config = PatternConfig(
    min_confidence=0.7,
    max_lookback=100,
    volume_threshold=1000000.0
)

# Create detectors
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
    FallingWedgeDetector(config)
]

# Initialize pattern engine
engine = PatternEngine(detectors)

# Detect patterns
signals = engine.detect_patterns(data, 'AAPL')

# Print results
for signal in signals:
    print(f"Pattern: {signal.pattern_type}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Stop Loss: ${signal.stop_loss:.2f}")
    print(f"Target: ${signal.target_price:.2f}")
    print("---")
```

## Command Line Interface

```bash
# Analyze a CSV file using backend
python analysis_scripts/trade_analyzer.py

# Run comprehensive backtesting
python analysis_scripts/comprehensive_backtest.py

# Real AAPL analysis
python analysis_scripts/real_aapl_analysis.py

# Quick pattern test
python testing/quick_backtest.py
```

## Supported Patterns

### Reversal Patterns
- **VCP Breakout**: Detects volatility contraction patterns followed by breakouts with ATR analysis
- **Head and Shoulders**: Classic reversal pattern with three peaks and neckline breakdown
- **Double Bottom**: W-shaped bullish reversal with neckline formation validation
- **Rounding Bottom**: Smooth U-shaped bullish reversal with volume confirmation

### Continuation Patterns
- **Flag Pattern**: Short-term continuation patterns with volume decline analysis
- **Cup and Handle**: Bullish continuation with cup formation and handle consolidation
- **Ascending Triangle**: Bullish continuation with horizontal resistance and ascending support
- **Descending Triangle**: Bearish continuation with horizontal support and descending resistance

### Breakout Patterns
- **Rising Wedge**: Bearish reversal pattern with ascending trendlines converging downward
- **Falling Wedge**: Bullish continuation pattern with descending trendlines converging upward

## Web Interface Features

### 📊 Data Management
- **Automatic Data Fetching**: Real-time market data from Yahoo Finance
- **Parquet Storage**: Efficient local storage with automatic updates
- **Smart Caching**: Memory caching for improved performance
- **Data Validation**: OHLC relationship validation and cleaning

### 🎯 Pattern Detection
- **Interactive Selection**: Choose from 12+ technical analysis patterns
- **Confidence Scoring**: Pattern-specific confidence metrics
- **Real-time Analysis**: Progress tracking during detection
- **Batch Processing**: Multiple patterns analyzed simultaneously

### 📈 Visualization & Analytics
- **Interactive Charts**: Price charts with pattern overlays
- **Performance Metrics**: Win ratios, Sharpe ratios, maximum drawdown
- **Pattern Analysis**: Detailed pattern characteristics and statistics
- **Export Capabilities**: CSV, JSON export options

## API Reference

### Core Backend Classes

```python
from src.core.interfaces import PatternEngine, PatternConfig, PatternSignal
from src.core.market_data import MarketDataIngestor
from src.core.interfaces import DataValidator

# Configure pattern detection
config = PatternConfig(
    min_confidence=0.7,
    max_lookback=100,
    volume_threshold=1000000.0
)

# Initialize pattern engine
engine = PatternEngine(detectors)

# Fetch market data
ingestor = MarketDataIngestor()
data = ingestor.fetch_stock_data("AAPL", period="2y", interval="1d")

# Detect patterns
signals = engine.detect_patterns(data, 'AAPL')

# Process signals
for signal in signals:
    print(f"Pattern: {signal.pattern_type}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Stop Loss: ${signal.stop_loss:.2f}")
    print(f"Target: ${signal.target_price:.2f}")
```

### Pattern Signal Structure

```python
signal = PatternSignal(
    symbol="AAPL",
    pattern_type=PatternType.DOUBLE_BOTTOM,
    confidence=0.85,
    entry_price=150.0,
    stop_loss=145.0,
    target_price=170.0,
    risk_level="medium",
    expected_duration="1-2 weeks",
    signal_strength=0.8,
    metadata={'key': 'value'}
)
```

### Frontend Integration

```python
from frontend.integration import PatternDetectionEngine
from frontend.data import DataManager
from frontend.components import InputForm, ResultsDisplay

# Initialize components
engine = PatternDetectionEngine()
input_form = InputForm()
results_display = ResultsDisplay()

# Run pattern detection
results = await engine.detect_patterns(
    symbol="AAPL",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 4, 1),
    pattern_types=["DOUBLE_BOTTOM", "FLAG_PATTERN"],
    confidence_threshold=0.5
)

# Display results
results_display.display_results(results, form_data)
```

## Complete Pattern List

The Trading Pattern Detector supports **12 major pattern types**:

### Reversal Patterns
- **VCP Breakout** (Volatility Contraction Pattern)
- **Head and Shoulders** - Classic three-peak reversal pattern
- **Double Bottom** - W-shaped bullish reversal
- **Rounding Bottom** - Smooth U-shaped bullish reversal
- **Rising Wedge** - Bearish reversal pattern

### Continuation Patterns
- **Flag Pattern** - Short-term continuation pattern
- **Cup and Handle** - Bullish continuation with handle formation

### Triangle Patterns
- **Ascending Triangle** - Bullish continuation with horizontal resistance
- **Descending Triangle** - Bearish continuation with horizontal support
- **Falling Wedge** - Bullish continuation/reversal pattern

### Additional Patterns
- **Wedge Pattern** - General wedge detection (both rising and falling)

## Configuration

### Pattern Parameters

```python
config = PatternConfig(
    min_confidence=0.6,           # Minimum confidence threshold
    max_lookback=100,             # Maximum historical data to analyze
    timeframe="1d",               # Data timeframe
    volume_threshold=1000000.0,   # Minimum volume threshold
    volatility_threshold=0.001,   # Volatility threshold
    reward_ratio=2.0             # Risk/reward ratio
)
```

### Custom Parameters

Each detector type has specific parameters that can be customized:

```python
# VCP Detector Parameters
vcp_config = PatternConfig(
    min_confidence=0.7,
    max_lookback=150,
    volatility_threshold=0.002
)

# Cup Handle Parameters
cup_config = PatternConfig(
    min_confidence=0.65,
    max_lookback=120,
    volume_threshold=500000.0
)
```

## Data Requirements

Input data should be a pandas DataFrame with the following columns:

```python
required_columns = ['open', 'high', 'low', 'close', 'volume']
```

Example DataFrame structure:
```python
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100)
prices = np.random.normal(100, 5, 100)

data = pd.DataFrame({
    'open': prices + np.random.normal(0, 1, 100),
    'high': prices + np.random.normal(0, 2, 100),
    'low': prices - np.random.normal(0, 2, 100),
    'close': prices + np.random.normal(0, 1, 100),
    'volume': np.random.lognormal(15, 1, 100)
}, index=dates)
```

## API Reference

### PatternEngine
Main orchestrator for pattern detection.

```python
engine = PatternEngine(detectors, config)
signals = engine.detect_patterns(data, symbol)
```

### PatternSignal
Represents a detected trading pattern.

```python
signal = PatternSignal(
    symbol="AAPL",
    pattern_type=PatternType.VCP_BREAKOUT,
    confidence=0.85,
    entry_price=150.0,
    stop_loss=145.0,
    target_price=170.0,
    timeframe="1d",
    timestamp=datetime.now(),
    metadata={}
)
```

### PatternDetector
Base class for all pattern detectors.

```python
class CustomDetector(BaseDetector):
    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        # Implementation
        pass

    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test modules
pytest tests/test_detectors/

# Run specific test
pytest tests/test_core_interfaces.py::TestPatternConfig
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/tradingpatterns/trading-pattern-detector.git
cd trading-pattern-detector
pip install -e ".[dev]"
pre-commit install
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Patterns

1. Create a new detector class inheriting from `BaseDetector`
2. Implement `detect_pattern` and `get_required_columns` methods
3. Add pattern type to `PatternType` enum
4. Write comprehensive tests
5. Update documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### v0.1.0 (2024-01-XX)
- Initial release
- Core pattern detection framework
- Support for 6 major pattern types
- Comprehensive test suite
- CLI interface
- Plugin system

## Support

- Documentation: [https://rain111.github.io/trading-pattern-detector/](https://rain111.github.io/trading-pattern-detector/)
- Issues: [GitHub Issues](https://github.com/rain111/trading-pattern-detector/issues)
- Discussions: [GitHub Discussions](https://github.com/rain111/trading-pattern-detector/discussions)

## Acknowledgments

- Technical Analysis Library (TA-Lib) for financial indicators
- Pandas and NumPy for data manipulation
- Scikit-learn for machine learning utilities
- Matplotlib and Plotly for visualization