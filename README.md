# Trading Pattern Detector

A sophisticated Python-based financial trading pattern detection system designed to identify various chart patterns in market data.

## Features

- **Pattern Detection**: Supports multiple trading patterns including:
  - VCP (Volatility Contraction Pattern) Breakout
  - Flag Patterns
  - Cup and Handle Patterns
  - Double Bottom Patterns
  - Triangle Patterns
  - Wedge Patterns

- **Comprehensive Analysis Tools**:
  - Volatility Analysis (ATR, Bollinger Bands)
  - Volume Analysis (Volume Profile, OBV)
  - Trend Analysis (Moving Averages, Trend Detection)
  - Support/Resistance Level Detection

- **Advanced Architecture**:
  - Plugin system for extensibility
  - Configurable pattern parameters
  - Robust data validation
  - Confidence scoring system
  - Comprehensive error handling

- **Professional Features**:
  - Signal aggregation and filtering
  - Multiple timeframe analysis
  - Risk management parameters
  - Extensive test coverage
  - Well-documented API

## Installation

### From Source

```bash
git clone https://github.com/tradingpatterns/trading-pattern-detector.git
cd trading-pattern-detector
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/tradingpatterns/trading-pattern-detector.git
cd trading-pattern-detector
pip install -e ".[dev]"
```

### Using pip (when published)

```bash
pip install trading-pattern-detector
```

## Quick Start

```python
from trading_pattern_detector import PatternEngine, PatternConfig
from trading_pattern_detector.detectors import (
    VCPBreakoutDetector, FlagPatternDetector,
    CupHandleDetector, DoubleBottomDetector,
    HeadAndShouldersDetector, RoundingBottomDetector,
    AscendingTriangleDetector, DescendingTriangleDetector,
    RisingWedgeDetector, FallingWedgeDetector
)
import pandas as pd

# Load your market data
data = pd.read_csv('your_market_data.csv')

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
# Analyze a CSV file
trading-pattern-detector analyze data.csv --symbol AAPL --min-confidence 0.7

# List available patterns
trading-pattern-detector patterns

# Generate example data
trading-pattern-detector sample-data --output sample.csv
```

## Supported Patterns

### VCP Breakout
- Detects volatility contraction patterns followed by breakouts
- Uses ATR analysis and volume validation
- Includes stage-by-stage pattern recognition

### Flag Pattern
- Identifies flag and pennant continuation patterns
- Analyzes volume decline during consolidation
- Validates breakout with volume surge

### Cup and Handle
- Detects cup-shaped reversal patterns with handle
- Analyzes cup depth and handle formation
- Validates breakout with volume confirmation

### Double Bottom
- Identifies W-shaped reversal patterns
- Analyzes neckline formation and breakout
- Validates with volume surge confirmation

### Triangle Patterns
- **Ascending Triangle**: Breakout patterns with horizontal resistance and ascending support
- **Descending Triangle**: Breakdown patterns with horizontal support and descending resistance

### Wedge Patterns
- **Rising Wedge**: Reversal pattern with ascending trendlines converging downward
- **Falling Wedge**: Reversal pattern with descending trendlines converging upward

### Head and Shoulders
- Classic reversal pattern with three peaks (head higher than shoulders)
- Detects neckline breakdown and validates with volume analysis
- Provides risk/reward calculations based on pattern height

### Rounding Bottom
- Bullish reversal pattern with smooth U-shaped formation
- Detects gradual transition from downtrend to uptrend
- Validates with volume confirmation and neckline breakout

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
    pattern_type=PatternType.HEAD_AND_SHOULDERS,
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