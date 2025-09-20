import sys
from pathlib import Path
# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

from src.core.interfaces import (
    PatternConfig,
    PatternSignal,
    PatternType,
    EnhancedPatternDetector,
    BaseDetector,
    DataValidator,
    PatternEngine,
)

# Import fixtures
from .test_fixtures import (
    sample_market_data,
    pattern_config,
    invalid_dataframe,
    corrupted_dataframe,
    empty_dataframe,
)


class TestPatternConfig:
    """Test PatternConfig dataclass"""

    def test_pattern_config_creation(self):
        """Test creating PatternConfig with default values"""
        config = PatternConfig()

        assert config.min_confidence == 0.6
        assert config.max_lookback == 100
        assert config.timeframe == "1d"
        assert config.volume_threshold == 1000000.0
        assert config.volatility_threshold == 0.001
        assert config.reward_ratio == 2.0

    def test_pattern_config_custom_values(self):
        """Test creating PatternConfig with custom values"""
        config = PatternConfig(
            min_confidence=0.8,
            max_lookback=200,
            timeframe="1h",
            volume_threshold=500000.0,
            volatility_threshold=0.002,
            reward_ratio=1.5,
        )

        assert config.min_confidence == 0.8
        assert config.max_lookback == 200
        assert config.timeframe == "1h"
        assert config.volume_threshold == 500000.0
        assert config.volatility_threshold == 0.002
        assert config.reward_ratio == 1.5


class TestPatternSignal:
    """Test PatternSignal dataclass"""

    def test_pattern_signal_creation(self):
        """Test creating PatternSignal with all values"""
        metadata = {"test": "value"}
        signal = PatternSignal(
            symbol="AAPL",
            pattern_type=PatternType.VCP_BREAKOUT,
            confidence=0.75,
            entry_price=150.0,
            stop_loss=145.0,
            target_price=170.0,
            timeframe="1d",
            timestamp=datetime.now(),
            metadata=metadata,
            signal_strength=0.8,
            risk_level="medium",
            expected_duration="2-4 weeks",
            probability_target=0.65,
        )

        assert signal.symbol == "AAPL"
        assert signal.pattern_type == PatternType.VCP_BREAKOUT
        assert signal.confidence == 0.75
        assert signal.entry_price == 150.0
        assert signal.stop_loss == 145.0
        assert signal.target_price == 170.0
        assert signal.timeframe == "1d"
        assert signal.metadata == metadata
        assert signal.signal_strength == 0.8
        assert signal.risk_level == "medium"
        assert signal.expected_duration == "2-4 weeks"
        assert signal.probability_target == 0.65

    def test_pattern_signal_defaults(self):
        """Test PatternSignal with default values"""
        signal = PatternSignal(
            symbol="AAPL",
            pattern_type=PatternType.VCP_BREAKOUT,
            confidence=0.75,
            entry_price=150.0,
            stop_loss=145.0,
            target_price=170.0,
            timeframe="1d",
            timestamp=datetime.now(),
            metadata={},
        )

        # Check default values
        assert signal.signal_strength == 0.0
        assert signal.risk_level == "medium"
        assert signal.expected_duration is None
        assert signal.probability_target is None


class TestDataValidator:
    """Test DataValidator class"""

    def test_valid_price_data_validation(self, sample_market_data):
        """Test validation of valid price data"""
        try:
            DataValidator.validate_price_data(sample_market_data)
            # If no exception is raised, validation passed
            assert True
        except Exception:
            assert False, "Valid price data should not raise an exception"

    def test_missing_columns_validation(self, invalid_dataframe):
        """Test validation with missing columns"""
        with pytest.raises(ValueError, match="Missing required columns"):
            DataValidator.validate_price_data(invalid_dataframe)

    def test_nan_values_validation(self, corrupted_dataframe):
        """Test validation with NaN values"""
        with pytest.raises(ValueError, match="contains NaN values"):
            DataValidator.validate_price_data(corrupted_dataframe)

    def test_price_logic_validation(self, corrupted_dataframe):
        """Test validation with illogical price relationships"""
        # Fix NaN values first to test price logic
        corrupted_clean = corrupted_dataframe.dropna()
        with pytest.raises(
            ValueError, match="High prices cannot be lower than low prices"
        ):
            DataValidator.validate_price_data(corrupted_clean)

    def test_empty_dataframe_validation(self, empty_dataframe):
        """Test validation of empty DataFrame"""
        with pytest.raises(ValueError, match="Missing required columns"):
            DataValidator.validate_price_data(empty_dataframe)


class TestEnhancedPatternDetector:
    """Test EnhancedPatternDetector abstract base class"""

    class MockDetector(EnhancedPatternDetector):
        """Mock implementation for testing"""

        def __init__(self, config: PatternConfig):
            super().__init__(config)
            self.setup_called = False

        def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
            return []

        def get_required_columns(self) -> List[str]:
            return ["open", "high", "low", "close", "volume"]

        def setup_called_check(self):
            self.setup_called = True

    def test_detector_initialization(self, pattern_config):
        """Test detector initialization"""
        detector = TestEnhancedPatternDetector.MockDetector(pattern_config)

        assert detector.config == pattern_config
        assert hasattr(detector, "logger")
        assert hasattr(detector, "volatility_analyzer")
        assert hasattr(detector, "volume_analyzer")
        assert hasattr(detector, "trend_analyzer")

    def test_validate_data_valid(self, sample_market_data):
        """Test data validation with valid data"""
        detector = TestEnhancedPatternDetector.MockDetector(PatternConfig())
        result = detector.validate_data(sample_market_data)

        assert result is True

    def test_validate_data_insufficient_length(self):
        """Test data validation with insufficient data"""
        detector = TestEnhancedPatternDetector.MockDetector(PatternConfig())
        short_data = pd.DataFrame(
            {"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [1000]}
        )

        result = detector.validate_data(short_data)
        assert result is False

    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns"""
        detector = TestEnhancedPatternDetector.MockDetector(PatternConfig())
        incomplete_data = pd.DataFrame(
            {"open": [1], "high": [2], "low": [0.5], "close": [1.5]}
        )

        result = detector.validate_data(incomplete_data)
        assert result is False

    def test_calculate_confidence(self, pattern_config):
        """Test confidence calculation"""
        detector = TestEnhancedPatternDetector.MockDetector(pattern_config)

        # Test with various pattern data
        pattern_data = {
            "volume_ratio": 1.5,
            "volatility_score": 0.02,
            "trend_strength": 0.8,
        }

        confidence = detector.calculate_confidence(pattern_data)

        # Should be between 0.5 and 1.0
        assert 0.5 <= confidence <= 1.0

        # Should be higher with better metrics
        better_data = {
            "volume_ratio": 2.0,
            "volatility_score": 0.05,
            "trend_strength": 0.9,
        }
        better_confidence = detector.calculate_confidence(better_data)
        assert better_confidence > confidence

    def test_generate_signals(self, pattern_config):
        """Test signal generation"""
        detector = TestEnhancedPatternDetector.MockDetector(pattern_config)

        patterns = [
            {
                "symbol": "AAPL",
                "pattern_type": PatternType.VCP_BREAKOUT,
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "target_price": 170.0,
                "timestamp": datetime.now(),
                "metadata": {},
                "volume_ratio": 1.5,
                "volatility_score": 0.02,
                "trend_strength": 0.8,
            }
        ]

        signals = detector.generate_signals(patterns)

        assert len(signals) == 1
        assert signals[0].symbol == "AAPL"
        assert signals[0].pattern_type == PatternType.VCP_BREAKOUT
        assert signals[0].confidence >= 0.6

    def test_generate_signals_low_confidence(self, pattern_config):
        """Test signal generation with low confidence"""
        detector = TestEnhancedPatternDetector.MockDetector(pattern_config)

        patterns = [
            {
                "symbol": "AAPL",
                "pattern_type": PatternType.VCP_BREAKOUT,
                "entry_price": 150.0,
                "stop_loss": 145.0,
                "target_price": 170.0,
                "timestamp": datetime.now(),
                "metadata": {"volume_ratio": 0.5},  # Low confidence
            }
        ]

        signals = detector.generate_signals(patterns)

        # Should not generate signals below confidence threshold
        assert len(signals) == 0


class TestBaseDetector:
    """Test BaseDetector class"""

    class MockBaseDetector(BaseDetector):
        """Mock implementation for testing"""

        def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
            return []

        def get_required_columns(self) -> List[str]:
            return ["open", "high", "low", "close", "volume"]

    def test_preprocess_data(self, sample_market_data):
        """Test data preprocessing"""
        detector = TestBaseDetector.MockBaseDetector(PatternConfig())
        processed = detector.preprocess_data(sample_market_data)

        # Check that preprocessing was applied
        assert isinstance(processed, pd.DataFrame)
        assert len(processed) <= len(sample_market_data)  # NaN values removed
        assert "returns" in processed.columns

    def test_validate_signals(self, pattern_config):
        """Test signal validation"""
        detector = TestBaseDetector.MockBaseDetector(pattern_config)

        signals = [
            PatternSignal(
                symbol="AAPL",
                pattern_type=PatternType.VCP_BREAKOUT,
                confidence=0.7,
                entry_price=150.0,
                stop_loss=145.0,
                target_price=170.0,
                timeframe="1d",
                timestamp=datetime.now(),
                metadata={},
            ),
            PatternSignal(
                symbol="AAPL",
                pattern_type=PatternType.VCP_BREAKOUT,
                confidence=0.5,  # Below threshold
                entry_price=150.0,
                stop_loss=145.0,
                target_price=170.0,
                timeframe="1d",
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        validated = detector.validate_signals(signals)

        # Should only keep signals above confidence threshold
        assert len(validated) == 1
        assert validated[0].confidence >= 0.6


class TestPatternEngine:
    """Test PatternEngine class"""

    class MockDetector(EnhancedPatternDetector):
        """Mock detector for testing"""

        def __init__(self, should_fail=False):
            super().__init__(PatternConfig())
            self.should_fail = should_fail

        def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
            if self.should_fail:
                raise Exception("Mock detector error")

            return [
                PatternSignal(
                    symbol="TEST",
                    pattern_type=PatternType.VCP_BREAKOUT,
                    confidence=0.8,
                    entry_price=100.0,
                    stop_loss=95.0,
                    target_price=110.0,
                    timeframe="1d",
                    timestamp=datetime.now(),
                    metadata={},
                )
            ]

        def get_required_columns(self) -> List[str]:
            return ["open", "high", "low", "close", "volume"]

    def test_engine_initialization(self):
        """Test engine initialization"""
        detector = TestPatternEngine.MockDetector()
        engine = PatternEngine([detector])

        assert len(engine.detectors) == 1
        assert engine.config == {}

    def test_engine_multiple_detectors(self):
        """Test engine with multiple detectors"""
        detector1 = TestPatternEngine.MockDetector()
        detector2 = TestPatternEngine.MockDetector()

        engine = PatternEngine([detector1, detector2])

        assert len(engine.detectors) == 2

    def test_detect_patterns_valid_data(self, sample_market_data):
        """Test pattern detection with valid data"""
        detector = TestPatternEngine.MockDetector()
        engine = PatternEngine([detector])

        signals = engine.detect_patterns(sample_market_data, "AAPL")

        assert len(signals) == 1
        assert signals[0].symbol == "AAPL"

    def test_detect_patterns_invalid_data(self, invalid_dataframe):
        """Test pattern detection with invalid data"""
        detector = TestPatternEngine.MockDetector()
        engine = PatternEngine([detector])

        # Should return empty list for invalid data, not raise exception
        signals = engine.detect_patterns(invalid_dataframe, "AAPL")
        assert len(signals) == 0

    def test_detect_patterns_detector_failure(self, sample_market_data):
        """Test pattern detection with detector failure"""
        failing_detector = TestPatternEngine.MockDetector(should_fail=True)
        working_detector = TestPatternEngine.MockDetector()

        engine = PatternEngine([failing_detector, working_detector])

        # Should handle detector failure gracefully
        signals = engine.detect_patterns(sample_market_data, "AAPL")

        # Should still get signals from working detector
        assert len(signals) == 1

    def test_add_detector(self, sample_market_data):
        """Test adding detector to engine"""
        detector1 = TestPatternEngine.MockDetector()
        detector2 = TestPatternEngine.MockDetector()

        engine = PatternEngine([detector1])
        assert len(engine.detectors) == 1

        engine.add_detector(detector2)
        assert len(engine.detectors) == 2

        # Should detect with both detectors
        signals = engine.detect_patterns(sample_market_data, "AAPL")
        assert len(signals) == 2

    def test_remove_detector(self, sample_market_data):
        """Test removing detector from engine"""
        detector1 = TestPatternEngine.MockDetector()
        detector2 = TestPatternEngine.MockDetector()

        engine = PatternEngine([detector1, detector2])
        assert len(engine.detectors) == 2

        # Note: current implementation removes all detectors of this type
        engine.remove_detector(TestPatternEngine.MockDetector)
        assert len(engine.detectors) == 0

        # Should detect with no detectors
        signals = engine.detect_patterns(sample_market_data, "AAPL")
        assert len(signals) == 0
