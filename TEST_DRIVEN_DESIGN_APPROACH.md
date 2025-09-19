# Test-Driven Development Approach

## Testing Philosophy

This document outlines the comprehensive test-driven development (TDD) approach for the trading pattern detection system, ensuring reliability, accuracy, and maintainability.

## Testing Principles

### 1. **Red-Green-Refactor Cycle**
- **Red**: Write a failing test first
- **Green**: Write minimal code to make test pass
- **Refactor**: Improve code while keeping tests green

### 2. **First Class Testing**
- Tests are as important as production code
- Comprehensive test coverage for all components
- Integration tests for real-world scenarios

### 3. **Continuous Testing**
- Run tests on every code change
- Automated CI/CD pipeline integration
- Performance and regression testing

## Testing Strategy

### 1. **Unit Testing**
- Test individual components in isolation
- Mock external dependencies
- Test business logic and algorithms

### 2. **Integration Testing**
- Test component interactions
- Test with real market data
- Test data processing pipelines

### 3. **Performance Testing**
- Test with large datasets
- Benchmark detection algorithms
- Test memory usage and execution time

### 4. **Error Testing**
- Test edge cases and error conditions
- Test data validation
- Test error handling and recovery

## Test Structure

### 1. **Test Organization**
```
tests/
├── unit/
│   ├── test_core/
│   │   ├── test_interfaces.py
│   │   ├── test_pattern_config.py
│   │   └── test_pattern_signal.py
│   ├── test_detectors/
│   │   ├── test_vcp_detector.py
│   │   ├── test_flag_detector.py
│   │   ├── test_triangle_detector.py
│   │   └── test_wedge_detector.py
│   ├── test_analysis/
│   │   ├── test_volatility_analyzer.py
│   │   ├── test_volume_analyzer.py
│   │   ├── test_trend_analyzer.py
│   │   └── test_support_resistance.py
│   └── test_utils/
│       ├── test_data_preprocessor.py
│       ├── test_signal_aggregator.py
│       └── test_market_data_client.py
├── integration/
│   ├── test_detection_pipeline.py
│   ├── test_multi_symbol_processing.py
│   └── test_signal_aggregation.py
├── performance/
│   ├── test_detector_performance.py
│   ├── test_large_dataset_processing.py
│   └── test_memory_usage.py
├── fixtures/
│   ├── sample_data/
│   │   ├── vcp_patterns.csv
│   │   ├── flag_patterns.csv
│   │   ├── triangle_patterns.csv
│   │   └── wedge_patterns.csv
│   └── expected_results/
│       ├── vcp_results.json
│       ├── flag_results.json
│       ├── triangle_results.json
│       └── wedge_results.json
└── conftest.py
```

### 2. **Test Configuration**
```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

@pytest.fixture
def sample_vcp_data() -> pd.DataFrame:
    """Sample VCP pattern data for testing"""
    # Create realistic VCP pattern data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Stage 1: Initial decline
    stage1_prices = np.linspace(100, 85, 20)
    
    # Stage 2: Volatility contraction
    stage2_base = 85
    stage2_volatility = np.linspace(0.5, 0.2, 30)  # Decreasing volatility
    stage2_prices = stage2_base + np.random.normal(0, stage2_volatility, 30)
    
    # Stage 3: Consolidation
    stage3_base = 85
    stage3_volatility = 0.2
    stage3_prices = stage3_base + np.random.normal(0, stage3_volatility, 25)
    
    # Stage 4: Breakout
    stage4_base = 85
    stage4_decline = np.linspace(0, -8, 25)  # Breakdown
    stage4_prices = stage4_base + stage4_decline
    
    # Combine stages
    prices = np.concatenate([stage1_prices, stage2_prices, stage3_prices, stage4_prices])
    
    # Generate OHLCV data
    high = prices + np.random.uniform(0, 0.5, len(prices))
    low = prices - np.random.uniform(0, 0.5, len(prices))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    close = prices
    volume = np.random.uniform(1000000, 5000000, len(prices))
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

@pytest.fixture
def sample_flag_data() -> pd.DataFrame:
    """Sample flag pattern data for testing"""
    dates = pd.date_range('2023-01-01', periods=80, freq='D')
    
    # Flagpole - sharp decline
    flagpole_length = 15
    flagpole_start = 100
    flagpole_end = 82  # 18% decline
    flagpole_prices = np.linspace(flagpole_start, flagpole_end, flagpole_length)
    
    # Flag - consolidation with slight downward slope
    flag_length = 30
    flag_start = flagpole_end
    flag_end = flagpole_end - 2  # Slight decline
    flag_prices = np.linspace(flag_start, flag_end, flag_length)
    
    # Breakout - continuation downward
    breakout_length = 35
    breakout_start = flag_end
    breakout_end = 70  # Further decline
    breakout_prices = np.linspace(breakout_start, breakout_end, breakout_length)
    
    # Combine stages
    prices = np.concatenate([flagpole_prices, flag_prices, breakout_prices])
    
    # Generate OHLCV data
    high = prices + np.random.uniform(0, 0.3, len(prices))
    low = prices - np.random.uniform(0, 0.3, len(prices))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    close = prices
    volume = np.random.uniform(800000, 3000000, len(prices))
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

@pytest.fixture
def sample_triangle_data() -> pd.DataFrame:
    """Sample triangle pattern data for testing"""
    dates = pd.date_range('2023-01-01', periods=90, freq='D')
    
    # Triangle pattern - converging trendlines
    base_price = 90
    convergence_factor = 0.98  # 2% convergence per period
    
    prices = [base_price]
    for i in range(1, 90):
        # Add some randomness but maintain convergence
        random_change = np.random.normal(0, 0.5)
        convergent_change = (base_price - prices[-1]) * 0.01
        new_price = prices[-1] + random_change + convergent_change
        prices.append(new_price)
    
    # Generate OHLCV data
    high = np.array(prices) + np.random.uniform(0, 0.4, len(prices))
    low = np.array(prices) - np.random.uniform(0, 0.4, len(prices))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    close = prices
    volume = np.random.uniform(600000, 2500000, len(prices))
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

@pytest.fixture
def invalid_data() -> pd.DataFrame:
    """Invalid data for testing validation"""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    
    return pd.DataFrame({
        'open': [100] * 50,
        'high': [90] * 50,  # Invalid: high < open
        'low': [110] * 50,  # Invalid: low > high
        'close': [95] * 50,
        'volume': [0] * 50  # Invalid: zero volume
    }, index=dates)

@pytest.fixture
def expected_vcp_results() -> Dict:
    """Expected results for VCP detection"""
    return {
        'pattern_count': 1,
        'expected_confidence': 0.75,
        'expected_entry_price': 85.0,
        'expected_stop_loss': 85.0,
        'expected_target_price': 81.0,
        'pattern_type': 'vcp_breakout'
    }

@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        'min_confidence': 0.6,
        'max_lookback': 100,
        'timeframe': '1d',
        'volume_threshold': 1000000.0,
        'volatility_threshold': 0.001,
        'reward_ratio': 2.0
    }
```

## Test Implementation Examples

### 1. **Core Interface Tests**
```python
# tests/unit/test_core/test_interfaces.py
import pytest
import pandas as pd
from datetime import datetime
from src.core.interfaces import PatternSignal, PatternType, PatternConfig
from typing import Dict, Any

class TestPatternSignal:
    """Test PatternSignal dataclass"""
    
    def test_pattern_signal_creation(self):
        """Test PatternSignal creation with all fields"""
        signal = PatternSignal(
            symbol="AAPL",
            pattern_type=PatternType.VCP_BREAKOUT,
            confidence=0.85,
            entry_price=100.0,
            stop_loss=105.0,
            target_price=95.0,
            timeframe="1d",
            timestamp=datetime.now(),
            metadata={"test": "data"}
        )
        
        assert signal.symbol == "AAPL"
        assert signal.pattern_type == PatternType.VCP_BREAKOUT
        assert signal.confidence == 0.85
        assert signal.entry_price == 100.0
        assert signal.stop_loss == 105.0
        assert signal.target_price == 95.0
        assert signal.timeframe == "1d"
        assert signal.metadata == {"test": "data"}
    
    def test_pattern_signal_validation(self):
        """Test PatternSignal validation"""
        with pytest.raises(ValueError):
            # Invalid confidence
            PatternSignal(
                symbol="AAPL",
                pattern_type=PatternType.VCP_BREAKOUT,
                confidence=1.5,  # Invalid confidence
                entry_price=100.0,
                stop_loss=105.0,
                target_price=95.0,
                timeframe="1d",
                timestamp=datetime.now(),
                metadata={}
            )

class TestPatternConfig:
    """Test PatternConfig validation"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = PatternConfig()
        
        assert config.min_confidence == 0.6
        assert config.max_lookback == 100
        assert config.timeframe == "1d"
        assert config.volume_threshold == 1000000.0
        assert config.volatility_threshold == 0.001
        assert config.reward_ratio == 2.0

class TestDataValidator:
    """Test DataValidator functionality"""
    
    def test_valid_data_validation(self, sample_vcp_data):
        """Test validation of valid data"""
        from src.core.interfaces import DataValidator
        
        # Should not raise exception
        DataValidator.validate_price_data(sample_vcp_data)
    
    def test_invalid_data_validation(self, invalid_data):
        """Test validation of invalid data"""
        from src.core.interfaces import DataValidator
        
        # Should raise exception
        with pytest.raises(ValueError):
            DataValidator.validate_price_data(invalid_data)
    
    def test_missing_columns_validation(self):
        """Test validation with missing columns"""
        from src.core.interfaces import DataValidator
        
        incomplete_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            # Missing low, close, volume
        })
        
        with pytest.raises(ValueError):
            DataValidator.validate_price_data(incomplete_data)
    
    def test_nan_values_validation(self):
        """Test validation with NaN values"""
        from src.core.interfaces import DataValidator
        
        data_with_nan = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 99],
            'close': [100, 101],
            'volume': [1000000, None]  # NaN volume
        }, index=[datetime.now(), datetime.now()])
        
        with pytest.raises(ValueError):
            DataValidator.validate_price_data(data_with_nan)

# tests/unit/test_core/test_pattern_engine.py
import pytest
from unittest.mock import Mock, patch
from src.core.interfaces import PatternEngine, PatternDetector, PatternSignal, PatternType

class MockDetector(PatternDetector):
    """Mock detector for testing"""
    
    def detect(self, data, symbol):
        return [PatternSignal(
            symbol=symbol,
            pattern_type=PatternType.VCP_BREAKOUT,
            confidence=0.8,
            entry_price=100.0,
            stop_loss=105.0,
            target_price=95.0,
            timeframe="1d",
            timestamp=data.index[-1],
            metadata={}
        )]
    
    def get_required_columns(self):
        return ['open', 'high', 'low', 'close', 'volume']

class TestPatternEngine:
    """Test PatternEngine functionality"""
    
    def test_engine_initialization(self):
        """Test PatternEngine initialization"""
        mock_detector = MockDetector()
        engine = PatternEngine([mock_detector])
        
        assert len(engine.detectors) == 1
        assert engine.config == {}
    
    def test_engine_with_config(self):
        """Test PatternEngine with configuration"""
        mock_detector = MockDetector()
        config = {'min_confidence': 0.7}
        engine = PatternEngine([mock_detector], config)
        
        assert engine.config == config
        assert engine.config['min_confidence'] == 0.7
    
    def test_detect_patterns(self, sample_vcp_data):
        """Test pattern detection with mock detector"""
        mock_detector = MockDetector()
        engine = PatternEngine([mock_detector])
        
        signals = engine.detect_patterns(sample_vcp_data, "AAPL")
        
        assert len(signals) == 1
        assert signals[0].symbol == "AAPL"
        assert signals[0].pattern_type == PatternType.VCP_BREAKOUT
    
    def test_detect_patterns_error_handling(self, sample_vcp_data):
        """Test error handling in pattern detection"""
        class ErrorDetector(PatternDetector):
            def detect(self, data, symbol):
                raise Exception("Detection failed")
            
            def get_required_columns(self):
                return ['open', 'high', 'low', 'close', 'volume']
        
        error_detector = ErrorDetector()
        working_detector = MockDetector()
        engine = PatternEngine([error_detector, working_detector])
        
        # Should not raise exception, should log error
        signals = engine.detect_patterns(sample_vcp_data, "AAPL")
        
        # Should return signals from working detector
        assert len(signals) == 1
        assert signals[0].symbol == "AAPL"
    
    def test_filter_signals_by_confidence(self, sample_vcp_data):
        """Test filtering signals by confidence"""
        low_confidence_detector = MockDetector()
        high_confidence_detector = MockDetector()
        
        # Modify mock to return different confidence
        def mock_detect_low(data, symbol):
            return [PatternSignal(
                symbol=symbol,
                pattern_type=PatternType.VCP_BREAKOUT,
                confidence=0.5,  # Below threshold
                entry_price=100.0,
                stop_loss=105.0,
                target_price=95.0,
                timeframe="1d",
                timestamp=data.index[-1],
                metadata={}
            )]
        
        low_confidence_detector.detect = mock_detect_low
        
        engine = PatternEngine([low_confidence_detector, high_confidence_detector], 
                             config={'min_confidence': 0.6})
        
        signals = engine.detect_patterns(sample_vcp_data, "AAPL")
        
        # Should only return high confidence signals
        for signal in signals:
            assert signal.confidence >= 0.6
    
    def test_add_detector(self, sample_vcp_data):
        """Test adding detector to engine"""
        engine = PatternEngine([])
        
        mock_detector = MockDetector()
        engine.add_detector(mock_detector)
        
        assert len(engine.detectors) == 1
        
        signals = engine.detect_patterns(sample_vcp_data, "AAPL")
        assert len(signals) == 1
    
    def test_remove_detector(self, sample_vcp_data):
        """Test removing detector from engine"""
        mock_detector = MockDetector()
        engine = PatternEngine([mock_detector])
        
        engine.remove_detector(MockDetector)
        
        assert len(engine.detectors) == 0
        
        signals = engine.detect_patterns(sample_vcp_data, "AAPL")
        assert len(signals) == 0
```

### 2. **VCP Detector Tests**
```python
# tests/unit/test_detectors/test_vcp_detector.py
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.detectors.vcp_detector import VCPBreakoutDetector
from src.core.interfaces import PatternConfig, PatternType

class TestVCPBreakoutDetector:
    """Test VCP Breakout Detector"""
    
    def test_detector_initialization(self, mock_config):
        """Test VCP detector initialization"""
        config = PatternConfig(**mock_config)
        detector = VCPBreakoutDetector(config)
        
        assert detector.config == config
        assert detector.min_confidence == 0.6
        assert detector.max_lookback == 100
    
    def test_get_required_columns(self):
        """Test required columns for VCP detection"""
        config = PatternConfig()
        detector = VCPBreakoutDetector(config)
        
        required_columns = detector.get_required_columns()
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        
        assert required_columns == expected_columns
    
    def test_vcp_detection_on_sample_data(self, sample_vcp_data, mock_config):
        """Test VCP detection on sample VCP data"""
        config = PatternConfig(**mock_config)
        detector = VCPBreakoutDetector(config)
        
        signals = detector.detect_pattern(sample_vcp_data)
        
        # Should detect VCP pattern
        assert len(signals) > 0
        
        # Check signal properties
        signal = signals[0]
        assert signal.pattern_type == PatternType.VCP_BREAKOUT
        assert signal.confidence >= 0.6
        assert signal.entry_price > 0
        assert signal.stop_loss > 0
        assert signal.target_price > 0
        assert signal.symbol == "UNKNOWN"  # Default value
    
    def test_no_detection_on_non_vcp_data(self, sample_triangle_data, mock_config):
        """Test that VCP detector doesn't trigger on non-VCP data"""
        config = PatternConfig(**mock_config)
        detector = VCPBreakoutDetector(config)
        
        signals = detector.detect_pattern(sample_triangle_data)
        
        # Should not detect VCP pattern
        assert len(signals) == 0
    
    def test_data_validation(self, invalid_data, mock_config):
        """Test data validation in VCP detector"""
        config = PatternConfig(**mock_config)
        detector = VCPBreakoutDetector(config)
        
        signals = detector.detect_pattern(invalid_data)
        
        # Should return empty list for invalid data
        assert len(signals) == 0
    
    def test_insufficient_data(self, mock_config):
        """Test VCP detector with insufficient data"""
        config = PatternConfig(**mock_config)
        detector = VCPBreakoutDetector(config)
        
        # Create small dataset
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 99],
            'close': [100, 101],
            'volume': [1000000, 2000000]
        }, index=[datetime.now(), datetime.now()])
        
        signals = detector.detect_pattern(small_data)
        
        # Should return empty list for insufficient data
        assert len(signals) == 0
    
    def test_vcp_stage_detection(self, mock_config):
        """Test individual VCP stage detection"""
        config = PatternConfig(**mock_config)
        detector = VCPBreakoutDetector(config)
        
        # Test initial decline detection
        initial_decline_data = self._create_initial_decline_data()
        has_initial_decline = detector._is_potential_vcp_start(initial_decline_data, 0)
        assert has_initial_decline == True
        
        # Test non-decline data
        rising_data = self._create_rising_data()
        has_decline = detector._is_potential_vcp_start(rising_data, 0)
        assert has_decline == False
    
    def _create_initial_decline_data(self):
        """Create data with initial decline"""
        dates = pd.date_range('2023-01-01', periods=25, freq='D')
        prices = np.linspace(100, 85, 25)  # 15% decline
        
        return pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 0.5, 25),
            'low': prices - np.random.uniform(0, 0.5, 25),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 25)
        }, index=dates)
    
    def _create_rising_data(self):
        """Create rising price data"""
        dates = pd.date_range('2023-01-01', periods=25, freq='D')
        prices = np.linspace(85, 100, 25)  # 18% increase
        
        return pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 0.5, 25),
            'low': prices - np.random.uniform(0, 0.5, 25),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 25)
        }, index=dates)
    
    def test_signal_generation(self, mock_config):
        """Test signal generation from VCP detection"""
        config = PatternConfig(**mock_config)
        detector = VCPBreakoutDetector(config)
        
        # Create mock pattern data
        pattern_data = {
            'symbol': 'AAPL',
            'pattern_type': PatternType.VCP_BREAKOUT,
            'confidence': 0.8,
            'entry_price': 85.0,
            'stop_loss': 85.0,
            'target_price': 81.0,
            'timestamp': datetime.now(),
            'metadata': {
                'breakout_strength': 0.1,
                'consolidation_range': 2.0,
                'volume_spike': 2000000,
                'support_level': 85.0,
                'resistance_level': 87.0
            }
        }
        
        signals = detector.generate_signals([pattern_data])
        
        assert len(signals) == 1
        signal = signals[0]
        assert signal.symbol == 'AAPL'
        assert signal.pattern_type == PatternType.VCP_BREAKOUT
        assert signal.confidence == 0.8
        assert signal.metadata == pattern_data['metadata']
    
    def test_confidence_calculation(self, mock_config):
        """Test confidence score calculation"""
        config = PatternConfig(**mock_config)
        detector = VCPBreakoutDetector(config)
        
        # Test with high confidence pattern data
        high_confidence_pattern = {
            'volume_ratio': 3.0,
            'volatility_score': 0.05,
            'trend_strength': 0.9
        }
        
        confidence = detector.calculate_confidence(high_confidence_pattern)
        assert confidence > 0.8  # Should be high confidence
        
        # Test with low confidence pattern data
        low_confidence_pattern = {
            'volume_ratio': 1.0,
            'volatility_score': 0.001,
            'trend_strength': 0.1
        }
        
        confidence = detector.calculate_confidence(low_confidence_pattern)
        assert confidence < 0.7  # Should be lower confidence
```

### 3. **Analysis Component Tests**
```python
# tests/unit/test_analysis/test_volatility_analyzer.py
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.analysis.volatility_analyzer import VolatilityAnalyzer

class TestVolatilityAnalyzer:
    """Test Volatility Analyzer"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = VolatilityAnalyzer(atr_period=14, bb_period=20)
        
        assert analyzer.atr_period == 14
        assert analyzer.bb_period == 20
    
    def test_calculate_atr(self):
        """Test ATR calculation"""
        analyzer = VolatilityAnalyzer(atr_period=14)
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(90, 110, 30),
            'high': np.random.uniform(95, 115, 30),
            'low': np.random.uniform(85, 105, 30),
            'close': np.random.uniform(90, 110, 30),
            'volume': np.random.uniform(1000000, 5000000, 30)
        }, index=dates)
        
        atr_series = analyzer.calculate_atr(data)
        
        # Check ATR properties
        assert len(atr_series) == len(data)
        assert not atr_series.isna().all()  # Should have some non-NaN values
        assert (atr_series >= 0).all()  # ATR should be non-negative
    
    def test_volatility_contraction_detection(self):
        """Test volatility contraction detection"""
        analyzer = VolatilityAnalyzer()
        
        # Create data with volatility contraction
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # High volatility first, then low volatility
        high_vol_data = np.random.normal(100, 5, 25)  # High volatility
        low_vol_data = np.random.normal(100, 1, 25)   # Low volatility
        
        prices = np.concatenate([high_vol_data, low_vol_data])
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 2, 50),
            'low': prices - np.random.uniform(0, 2, 50),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 50)
        }, index=dates)
        
        contraction_info = analyzer.calculate_volatility_contraction(data, period=25)
        
        # Check contraction info structure
        assert 'atr_trend' in contraction_info
        assert 'contraction_ratio' in contraction_info
        assert 'volatility_score' in contraction_info
        assert 'is_contracting' in contraction_info
        
        # Should detect contraction in second half
        assert isinstance(contraction_info['is_contracting'], bool)
        assert isinstance(contraction_info['contraction_ratio'], float)
        assert contraction_info['contraction_ratio'] > 0
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        analyzer = VolatilityAnalyzer(bb_period=20)
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        base_prices = np.linspace(95, 105, 50)
        prices = base_prices + np.random.normal(0, 2, 50)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 1, 50),
            'low': prices - np.random.uniform(0, 1, 50),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 50)
        }, index=dates)
        
        bb_info = analyzer.calculate_bollinger_bands(data)
        
        # Check Bollinger Bands structure
        assert 'sma' in bb_info
        assert 'upper_band' in bb_info
        assert 'lower_band' in bb_info
        assert 'band_width' in bb_info
        
        # Check properties
        assert len(bb_info['sma']) == len(data)
        assert len(bb_info['upper_band']) == len(data)
        assert len(bb_info['lower_band']) == len(data)
        assert len(bb_info['band_width']) == len(data)
        
        # Check upper band > lower band
        assert (bb_info['upper_band'] > bb_info['lower_band']).all()

# tests/unit/test_analysis/test_volume_analyzer.py
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from src.analysis.volume_analyzer import VolumeAnalyzer

class TestVolumeAnalyzer:
    """Test Volume Analyzer"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = VolumeAnalyzer(volume_window=20)
        
        assert analyzer.volume_window == 20
    
    def test_volume_pattern_analysis(self):
        """Test volume pattern analysis"""
        analyzer = VolumeAnalyzer()
        
        # Create test data with volume pattern
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # Volume spike in middle
        base_volume = 1000000
        volume_spike = 3000000
        
        volumes = np.concatenate([
            np.random.normal(base_volume, 200000, 15),  # Normal volume
            np.random.normal(volume_spike, 500000, 10),   # Volume spike
            np.random.normal(base_volume, 200000, 25)   # Normal volume
        ])
        
        prices = np.linspace(100, 110, 50) + np.random.normal(0, 1, 50)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 1, 50),
            'low': prices - np.random.uniform(0, 1, 50),
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        volume_info = analyzer.analyze_volume_patterns(data)
        
        # Check volume info structure
        assert 'volume_ratio' in volume_info
        assert 'volume_trend' in volume_info
        assert 'recent_volume_trend' in volume_info
        assert 'volume_spike_count' in volume_info
        assert 'volume_spike_ratio' in volume_info
        assert 'volume_spike_active' in volume_info
        
        # Check types
        assert isinstance(volume_info['volume_ratio'], float)
        assert isinstance(volume_info['volume_trend'], float)
        assert isinstance(volume_info['volume_spike_count'], int)
        assert isinstance(volume_info['volume_spike_ratio'], float)
        assert isinstance(volume_info['volume_spike_active'], bool)
    
    def test_volume_breakout_confirmation(self):
        """Test volume breakout confirmation"""
        analyzer = VolumeAnalyzer()
        
        # Create test data with volume breakout
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        # Normal volume, then spike
        normal_volume = 1000000
        spike_volume = 3000000
        
        volumes = np.concatenate([
            np.random.normal(normal_volume, 200000, 20),
            np.random.normal(spike_volume, 500000, 10)
        ])
        
        prices = np.linspace(100, 105, 30)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 1, 30),
            'low': prices - np.random.uniform(0, 1, 30),
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Test breakout confirmation
        breakout_confirmed = analyzer.confirm_volume_breakout(data, 25)  # Spike at index 25
        
        assert isinstance(breakout_confirmed, bool)
    
    def test_volume_spike_detection(self):
        """Test volume spike detection"""
        analyzer = VolumeAnalyzer()
        
        # Create data with volume spike
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        
        # Base volume + one spike
        base_volume = 1000000
        spike_volume = 4000000
        
        volumes = np.concatenate([
            np.random.normal(base_volume, 200000, 28),
            [spike_volume, base_volume]  # Spike at second to last
        ])
        
        prices = np.linspace(100, 110, 30)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 1, 30),
            'low': prices - np.random.uniform(0, 1, 30),
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        volume_info = analyzer.analyze_volume_patterns(data)
        
        # Should detect spike
        assert volume_info['volume_spike_count'] > 0
        assert volume_info['volume_spike_ratio'] > 0
```

### 4. **Integration Tests**
```python
# tests/integration/test_detection_pipeline.py
import pytest
import pandas as pd
from datetime import datetime
from src.core.interfaces import PatternEngine, PatternConfig
from src.detectors.vcp_detector import VCPBreakoutDetector
from src.detectors.flag_detector import FlagPatternDetector
from src.utils.data_preprocessor import DataPreprocessor
from src.utils.signal_aggregator import SignalAggregator

class TestDetectionPipeline:
    """Test complete detection pipeline"""
    
    def test_end_to_end_pipeline(self, sample_vcp_data, sample_flag_data):
        """Test end-to-end detection pipeline"""
        # Setup detectors
        config = PatternConfig(min_confidence=0.6)
        vcp_detector = VCPBreakoutDetector(config)
        flag_detector = FlagPatternDetector(config)
        
        # Create engine
        engine = PatternEngine([vcp_detector, flag_detector])
        
        # Process VCP data
        vcp_signals = engine.detect_patterns(sample_vcp_data, "AAPL")
        
        # Process flag data
        flag_signals = engine.detect_patterns(sample_flag_data, "MSFT")
        
        # Check results
        assert len(vcp_signals) > 0
        assert len(flag_signals) > 0
        
        # Check that we got different patterns
        vcp_patterns = [s.pattern_type for s in vcp_signals]
        flag_patterns = [s.pattern_type for s in flag_signals]
        
        assert PatternType.VCP_BREAKOUT in vcp_patterns
        assert PatternType.FLAG_PATTERN in flag_patterns
    
    def test_multi_symbol_processing(self):
        """Test processing multiple symbols"""
        # Create sample data for multiple symbols
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # VCP pattern for AAPL
        aapl_data = self._create_vcp_pattern_data(dates)
        
        # Flag pattern for MSFT
        msft_data = self._create_flag_pattern_data(dates)
        
        # Triangle pattern for GOOGL
        googl_data = self._create_triangle_pattern_data(dates)
        
        data_dict = {
            'AAPL': aapl_data,
            'MSFT': msft_data,
            'GOOGL': googl_data
        }
        
        # Setup detectors
        config = PatternConfig(min_confidence=0.6)
        vcp_detector = VCPBreakoutDetector(config)
        flag_detector = FlagPatternDetector(config)
        
        engine = PatternEngine([vcp_detector, flag_detector])
        
        # Process all symbols
        all_signals = []
        for symbol, data in data_dict.items():
            signals = engine.detect_patterns(data, symbol)
            all_signals.extend(signals)
        
        # Should detect patterns in multiple symbols
        assert len(all_signals) > 0
        
        # Check symbols are correct
        symbols = [s.symbol for s in all_signals]
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'GOOGL' in symbols
    
    def test_data_preprocessing_pipeline(self):
        """Test data preprocessing pipeline"""
        preprocessor = DataPreprocessor()
        
        # Create messy data
        messy_data = pd.DataFrame({
            'open': [100, 101, None, 103, 104, 105],
            'high': [102, 103, 104, None, 106, 107],
            'low': [98, 99, 100, 101, None, 103],
            'close': [100, 101, 102, 103, 104, 105],
            'volume': [1000000, 2000000, None, 4000000, 5000000, 0]
        }, index=[datetime.now() + pd.Timedelta(days=i) for i in range(6)])
        
        # Clean data
        clean_data = preprocessor.clean_data(messy_data)
        
        # Check data is cleaned
        assert not clean_data.isna().any().any()
        assert (clean_data['volume'] > 0).all()
    
    def test_signal_aggregation_pipeline(self, sample_vcp_data):
        """Test signal aggregation pipeline"""
        config = PatternConfig(min_confidence=0.6)
        vcp_detector = VCPBreakoutDetector(config)
        
        # Get signals from detector
        signals = vcp_detector.detect_pattern(sample_vcp_data)
        
        # Aggregate signals
        aggregator = SignalAggregator()
        aggregator.add_signals(signals)
        
        # Test ranking
        ranked_signals = aggregator.rank_signals('confidence')
        assert len(ranked_signals) == len(signals)
        
        # Test filtering
        filtered_signals = aggregator.filter_signals(min_confidence=0.7)
        assert len(filtered_signals) <= len(signals)
        
        for signal in filtered_signals:
            assert signal.confidence >= 0.7
    
    def _create_vcp_pattern_data(self, dates):
        """Create VCP pattern data"""
        # Implementation similar to sample_vcp_data fixture
        pass
    
    def _create_flag_pattern_data(self, dates):
        """Create flag pattern data"""
        # Implementation similar to sample_flag_data fixture
        pass
    
    def _create_triangle_pattern_data(self, dates):
        """Create triangle pattern data"""
        # Implementation similar to sample_triangle_data fixture
        pass

# tests/integration/test_performance.py
import pytest
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.detectors.vcp_detector import VCPBreakoutDetector
from src.core.interfaces import PatternConfig

class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        # Create large dataset (10,000 rows)
        dates = pd.date_range('2020-01-01', periods=10000, freq='H')
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0, 0.001, 10000)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        large_data = pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': np.array(prices) + np.random.uniform(0, 0.5, 10000),
            'low': np.array(prices) - np.random.uniform(0, 0.5, 10000),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 10000)
        }, index=dates)
        
        config = PatternConfig()
        detector = VCPBreakoutDetector(config)
        
        # Measure performance
        start_time = time.time()
        signals = detector.detect_pattern(large_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 30.0  # Should complete in under 30 seconds
        assert len(large_data) == 10000
        
        print(f"Processed 10,000 rows in {processing_time:.2f} seconds")
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage characteristics"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=50000, freq='H')
        
        # Generate realistic price data
        base_price = 100
        returns = np.random.normal(0, 0.001, 50000)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        large_data = pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': np.array(prices) + np.random.uniform(0, 0.5, 50000),
            'low': np.array(prices) - np.random.uniform(0, 0.5, 50000),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 50000)
        }, index=dates)
        
        config = PatternConfig()
        detector = VCPBreakoutDetector(config)
        
        # Process data
        signals = detector.detect_pattern(large_data)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage assertions
        assert memory_increase < 500  # Should use less than 500MB additional memory
        print(f"Memory increase: {memory_increase:.2f} MB for 50,000 rows")
```

### 5. **Test Configuration and Utilities**
```python
# tests/test_utils.py
import pytest
import tempfile
import os
import json
from pathlib import Path
from src.utils.data_preprocessor import DataPreprocessor
from src.utils.signal_aggregator import SignalAggregator
from src.core.interfaces import PatternSignal, PatternType

class TestDataPreprocessor:
    """Test data preprocessor utilities"""
    
    def test_clean_data(self):
        """Test data cleaning"""
        preprocessor = DataPreprocessor()
        
        # Create data with duplicates and missing values
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        
        # Add duplicate timestamp
        duplicate_dates = dates.tolist() + [dates[0]]
        
        messy_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000000, 2000000, None, 4000000, 5000000, 0, 6000000, 7000000, 8000000, 9000000, 10000000]
        }, index=duplicate_dates)
        
        clean_data = preprocessor.clean_data(messy_data)
        
        # Check cleaning results
        assert len(clean_data) == len(dates)  # Duplicates removed
        assert not clean_data.isna().any().any()  # No NaN values
        assert (clean_data['volume'] > 0).all()  # No zero volume
    
    def test_resample_data(self):
        """Test data resampling"""
        preprocessor = DataPreprocessor()
        
        # Create hourly data
        hourly_dates = pd.date_range('2023-01-01', periods=48, freq='H')
        
        hourly_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 48),
            'high': np.random.uniform(105, 115, 48),
            'low': np.random.uniform(95, 105, 48),
            'close': np.random.uniform(100, 110, 48),
            'volume': np.random.uniform(1000000, 5000000, 48)
        }, index=hourly_dates)
        
        # Resample to daily
        daily_data = preprocessor.resample_data(hourly_data, '1D')
        
        # Check resampling results
        assert len(daily_data) == 2  # 48 hours = 2 days
        assert 'open' in daily_data.columns
        assert 'high' in daily_data.columns
        assert 'low' in daily_data.columns
        assert 'close' in daily_data.columns
        assert 'volume' in daily_data.columns
    
    def test_add_technical_indicators(self):
        """Test adding technical indicators"""
        preprocessor = DataPreprocessor()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        prices = np.linspace(100, 150, 100) + np.random.normal(0, 2, 100)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.uniform(0, 1, 100),
            'low': prices - np.random.uniform(0, 1, 100),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)
        
        enhanced_data = preprocessor.add_technical_indicators(data)
        
        # Check indicators added
        assert 'sma_20' in enhanced_data.columns
        assert 'sma_50' in enhanced_data.columns
        assert 'rsi' in enhanced_data.columns
        assert 'macd' in enhanced_data.columns
        assert 'macd_signal' in enhanced_data.columns
        assert 'macd_histogram' in enhanced_data.columns

class TestSignalAggregator:
    """Test signal aggregator utilities"""
    
    def test_signal_aggregation(self):
        """Test signal aggregation"""
        aggregator = SignalAggregator()
        
        # Create test signals
        signals = [
            PatternSignal(
                symbol="AAPL",
                pattern_type=PatternType.VCP_BREAKOUT,
                confidence=0.8,
                entry_price=100.0,
                stop_loss=105.0,
                target_price=95.0,
                timeframe="1d",
                timestamp=pd.Timestamp.now(),
                metadata={}
            ),
            PatternSignal(
                symbol="MSFT",
                pattern_type=PatternType.FLAG_PATTERN,
                confidence=0.7,
                entry_price=200.0,
                stop_loss=205.0,
                target_price=190.0,
                timeframe="1d",
                timestamp=pd.Timestamp.now(),
                metadata={}
            )
        ]
        
        # Add signals
        aggregator.add_signals(signals)
        
        # Check aggregation
        assert len(aggregator.signals) == 2
        
        # Test filtering
        filtered = aggregator.filter_signals(min_confidence=0.75)
        assert len(filtered) == 1
        assert filtered[0].symbol == "AAPL"
        
        # Test ranking
        ranked = aggregator.rank_signals('confidence')
        assert ranked[0].confidence >= ranked[1].confidence
    
    def test_signal_serialization(self):
        """Test signal serialization"""
        aggregator = SignalAggregator()
        
        # Create test signal
        signal = PatternSignal(
            symbol="AAPL",
            pattern_type=PatternType.VCP_BREAKOUT,
            confidence=0.8,
            entry_price=100.0,
            stop_loss=105.0,
            target_price=95.0,
            timeframe="1d",
            timestamp=pd.Timestamp.now(),
            metadata={}
        )
        
        aggregator.add_signal(signal)
        
        # Test saving and loading
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            json.dump([signal.__dict__ for signal in aggregator.signals], f)
            temp_file = f.name
        
        try:
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert len(loaded_data) == 1
            assert loaded_data[0]['symbol'] == "AAPL"
            
        finally:
            os.unlink(temp_file)
```

## Test Execution Configuration

### 1. **pytest Configuration**
```python
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    performance: marks tests as performance tests
```

### 2. **Test Coverage Configuration**
```python
# tests/conftest.py
import pytest
import coverage

@pytest.fixture(scope="session")
def coverage_config():
    """Configure coverage for tests"""
    cov = coverage.Coverage(
        source=["src"],
        omit=[
            "*/tests/*",
            "*/test_*",
            "*/conftest.py",
            "*/__pycache__/*"
        ]
    )
    cov.start()
    yield cov
    cov.stop()
    cov.save()

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Reset any global state
```

### 3. **Continuous Integration Configuration**
```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov pytest-mock coverage
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

## Testing Best Practices

### 1. **Test Organization**
- Group tests by functionality
- Use descriptive test names
- Follow the `test_` prefix convention

### 2. **Test Data Management**
- Use fixtures for test data
- Create realistic market data scenarios
- Store test data in separate files

### 3. **Test Isolation**
- Each test should be independent
- Use mocks for external dependencies
- Reset state between tests

### 4. **Performance Testing**
- Test with realistic data sizes
- Measure execution time
- Monitor memory usage

### 5. **Error Testing**
- Test edge cases
- Test invalid inputs
- Test error conditions

### 6. **Integration Testing**
- Test component interactions
- Test with real market data
- Test the full pipeline

This comprehensive test-driven development approach ensures that the trading pattern detection system is reliable, maintainable, and production-ready.