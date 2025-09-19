# Examples

This directory contains usage examples for the Trading Pattern Detector.

## Examples Overview

### 1. `basic_usage.py`
A simple demonstration of the core functionality:

- **What it shows**: Basic pattern detection workflow
- **Features demonstrated**:
  - Data generation
  - Configuration setup
  - Pattern detection
  - Results interpretation
  - Summary statistics

- **Use case**: Getting started with the library
- **Run with**: `python basic_usage.py`

### 2. `advanced_usage.py`
Comprehensive example showing advanced features:

- **What it shows**: Multi-symbol analysis with performance simulation
- **Features demonstrated**:
  - Multi-symbol processing
  - Multiple configuration strategies
  - Signal filtering and ranking
  - Trading performance simulation
  - Statistical analysis
  - Cross-symbol comparison

- **Use case**: Professional analysis and strategy development
- **Run with**: `python advanced_usage.py`

## Customization Examples

### Custom Detector Configuration

```python
# Create custom configuration for specific trading style
conservative_config = PatternConfig(
    min_confidence=0.8,           # High confidence threshold
    max_lookback=200,             # Long-term analysis
    volume_threshold=5000000.0,   # Large volume requirement
    reward_ratio=3.0             # High risk/reward ratio
)

aggressive_config = PatternConfig(
    min_confidence=0.4,           # Lower confidence threshold
    max_lookback=30,              # Short-term analysis
    volume_threshold=100000.0,    # Lower volume requirement
    reward_ratio=1.5              # Lower risk/reward ratio
)
```

### Adding Custom Detectors

```python
from trading_pattern_detector.core.interfaces import BaseDetector, PatternSignal, PatternType

class CustomPatternDetector(BaseDetector):
    def get_required_columns(self):
        return ['open', 'high', 'low', 'close', 'volume']

    def detect_pattern(self, data):
        # Your custom pattern detection logic
        signals = []

        # Example: Detect simple moving average crossover
        ma20 = data['close'].rolling(20).mean()
        ma50 = data['close'].rolling(50).mean()

        # Find crossover points
        crossover = (ma20 > ma50) & (ma20.shift() <= ma50.shift())

        for idx in crossover[crossover].index:
            signal = PatternSignal(
                symbol="",
                pattern_type=PatternType.CUSTOM_PATTERN,
                confidence=0.7,
                entry_price=data.loc[idx, 'close'],
                stop_loss=data.loc[idx, 'close'] * 0.95,
                target_price=data.loc[idx, 'close'] * 1.1,
                timeframe="1d",
                timestamp=idx,
                metadata={'ma20': ma20[idx], 'ma50': ma50[idx]}
            )
            signals.append(signal)

        return signals
```

### Real-World Data Integration

```python
import yfinance as yf

def fetch_stock_data(symbol, period="1y", interval="1d"):
    """Fetch real market data using Yahoo Finance"""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)

    # Convert to required format
    df = pd.DataFrame({
        'open': data['Open'],
        'high': data['High'],
        'low': data['Low'],
        'close': data['Close'],
        'volume': data['Volume']
    })

    return df

# Usage
apple_data = fetch_stock_data('AAPL', period='6mo')
signals = engine.detect_patterns(apple_data, 'AAPL')
```

### Signal Filtering and Risk Management

```python
def filter_signals_by_risk(signals, max_risk_per_trade=0.02, account_size=100000):
    """Filter signals based on position sizing and risk"""
    filtered_signals = []

    for signal in signals:
        # Calculate position size based on risk
        risk_amount = account_size * max_risk_per_trade
        risk_per_share = signal.entry_price - signal.stop_loss
        position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0

        if position_size > 0:
            signal.metadata.update({
                'position_size': position_size,
                'risk_amount': risk_amount,
                'risk_pct': risk_per_share / signal.entry_price
            })
            filtered_signals.append(signal)

    return filtered_signals
```

## Integration with Other Libraries

### Integration with Backtesting Libraries

```python
import backtrader as bt

class PatternStrategy(bt.Strategy):
    params = (
        ('min_confidence', 0.6),
        ('position_size', 100),
    )

    def __init__(self):
        self.detectors = self._create_detectors()
        self.signals = []

    def _create_detectors(self):
        # Create detectors using your configuration
        config = PatternConfig(min_confidence=self.params.min_confidence)
        return [
            VCPBreakoutDetector(config),
            FlagPatternDetector(config),
            # Add other detectors...
        ]

    def next(self):
        if len(self.data) < 100:  # Wait for enough data
            return

        # Convert backtrader data to DataFrame
        df = pd.DataFrame({
            'open': self.data.open.array,
            'high': self.data.high.array,
            'low': self.data.low.array,
            'close': self.data.close.array,
            'volume': self.data.volume.array
        }, index=self.data.datetime.array)

        # Detect patterns
        engine = PatternEngine(self.detectors)
        signals = engine.detect_patterns(df, self._get_symbol())

        # Execute trades based on signals
        for signal in signals:
            if signal.confidence >= self.params.min_confidence:
                self.buy(size=self.params.position_size)

    def _get_symbol(self):
        return self.datas[0]._name
```

### Integration with Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_signals_with_price(data, signals, symbol):
    """Plot price data with detected signals"""
    plt.figure(figsize=(15, 10))

    # Price chart
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'], label='Price', color='blue', alpha=0.7)

    # Plot signals
    for signal in signals:
        plt.scatter(signal.timestamp, signal.entry_price,
                   marker='^', s=100, c='green',
                   label=f'Buy {signal.pattern_type.name}')
        plt.scatter(signal.timestamp, signal.target_price,
                   marker='v', s=100, c='red',
                   label=f'Target {signal.pattern_type.name}')

    plt.title(f'{symbol} Price with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Volume chart
    plt.subplot(2, 1, 2)
    plt.bar(data.index, data['volume'], alpha=0.7, color='gray')
    plt.title('Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')

    plt.tight_layout()
    plt.show()

# Usage
plot_signals_with_price(apple_data, apple_signals, 'AAPL')
```

## Performance Optimization Tips

### 1. Data Preprocessing

```python
def preprocess_data(data):
    """Optimize data for pattern detection"""
    # Remove weekends and holidays
    data = data[data.index.dayofweek < 5]

    # Handle missing values
    data = data.dropna()

    # Sort by index
    data = data.sort_index()

    return data
```

### 2. Caching Results

```python
import pickle
import os

def cache_results(cache_file, data, signals):
    """Cache detection results for faster loading"""
    cache_data = {
        'data_hash': hash(str(data.values.tobytes())),
        'signals': signals,
        'timestamp': datetime.now()
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

def load_cached_results(cache_file, data):
    """Load cached results if available"""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        # Verify data hasn't changed
        if cache_data['data_hash'] == hash(str(data.values.tobytes())):
            return cache_data['signals']

    return None
```

### 3. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def analyze_multiple_symbols_parallel(symbols, data_dict):
    """Analyze multiple symbols in parallel"""
    def analyze_symbol(symbol):
        config = PatternConfig(min_confidence=0.6)
        detectors = [
            VCPBreakoutDetector(config),
            FlagPatternDetector(config),
            # Add other detectors...
        ]

        engine = PatternEngine(detectors)
        signals = engine.detect_patterns(data_dict[symbol], symbol)
        return symbol, signals

    # Use all available CPU cores
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(analyze_symbol, symbols))

    return dict(results)
```

## Testing Your Setup

Before running the examples, verify your installation:

```bash
# Test imports
python -c "import trading_pattern_detector; print('✓ Import successful')"

# Test CLI
trading-pattern-detector patterns

# Generate sample data
trading-pattern-detector sample-data --days 100 --output test_data.csv

# Analyze sample data
trading-pattern-detector analyze test_data.csv --symbol TEST --min-confidence 0.6
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   Solution: Ensure src/ is in your Python path or install the package
   ```

2. **Missing Dependencies**
   ```
   Solution: pip install -r requirements.txt
   ```

3. **Data Format Issues**
   ```
   Solution: Ensure your data has columns: open, high, low, close, volume
   ```

4. **No Patterns Detected**
   ```
   Solution: Try lowering min_confidence or use sample-data to generate test data
   ```

### Getting Help

- Check the main README.md for detailed documentation
- Run examples with verbose logging: `python basic_usage.py -v`
- Use the CLI help: `trading-pattern-detector --help`