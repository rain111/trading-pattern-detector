"""
Tests for the integration layer
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from frontend.integration import PatternDetectionEngine, ProgressManager, error_handler
from frontend.integration.error_handler import ErrorType, ErrorInfo
from frontend.config import settings

class TestPatternDetectionEngine:
    """Test cases for PatternDetectionEngine"""

    @pytest.fixture
    def engine(self):
        """Create a PatternDetectionEngine instance"""
        return PatternDetectionEngine()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': [100 + i for i in range(100)],
            'high': [105 + i for i in range(100)],
            'low': [95 + i for i in range(100)],
            'close': [100 + i for i in range(100)],
            'volume': [1000000 + i * 1000 for i in range(100)]
        }, index=dates)
        return data

    @pytest.fixture
    def test_parameters(self):
        """Test parameters for pattern detection"""
        return {
            'symbol': 'AAPL',
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 4, 1),
            'pattern_types': ['DOUBLE_BOTTOM', 'FLAG_PATTERN'],
            'confidence_threshold': 0.5
        }

    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization"""
        await engine.initialize()
        assert engine.data_manager is not None
        assert engine.ingestor is not None
        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_detect_patterns_success(self, engine, test_parameters, sample_data):
        """Test successful pattern detection"""
        with patch.object(engine, 'data_manager') as mock_data_manager:
            mock_data_manager.get_stock_data.return_value = sample_data

            with patch.object(engine, '_create_detector') as mock_create_detector:
                mock_detector = Mock()
                mock_detector.detect_patterns.return_value = []
                mock_create_detector.return_value = mock_detector

                with patch.object(engine, '_process_signal') as mock_process_signal:
                    mock_process_signal.return_value = {
                        'symbol': 'AAPL',
                        'pattern_type': 'DOUBLE_BOTTOM',
                        'confidence': 0.8,
                        'entry_price': 100.0,
                        'stop_loss': 95.0,
                        'target_price': 110.0
                    }

                results = await engine.detect_patterns(**test_parameters)

                assert results['success'] is True
                assert results['symbol'] == 'AAPL'
                assert len(results['signals']) == 1
                assert results['metrics']['total_signals'] == 1

    @pytest.mark.asyncio
    async def test_detect_patterns_no_data(self, engine, test_parameters):
        """Test pattern detection with no data"""
        with patch.object(engine, 'data_manager') as mock_data_manager:
            mock_data_manager.get_stock_data.return_value = pd.DataFrame()

            results = await engine.detect_patterns(**test_parameters)

            assert results['success'] is False
            assert 'No data available' in results['error']

    @pytest.mark.asyncio
    async def test_detect_patterns_invalid_symbol(self, engine, test_parameters):
        """Test pattern detection with invalid symbol"""
        test_parameters['symbol'] = ''
        errors = await engine.validate_inputs(**test_parameters)
        assert len(errors) > 0
        assert 'Stock symbol must be alphanumeric' in errors

    def test_create_detector(self, engine):
        """Test detector creation"""
        config = Mock()
        detector = engine._create_detector('DOUBLE_BOTTOM', config)
        assert detector is not None

    def test_create_detector_invalid_pattern(self, engine):
        """Test detector creation with invalid pattern"""
        config = Mock()
        detector = engine._create_detector('INVALID_PATTERN', config)
        assert detector is None

    def test_process_signal(self, engine, sample_data):
        """Test signal processing"""
        mock_signal = Mock()
        mock_signal.symbol = 'AAPL'
        mock_signal.pattern_type = Mock()
        mock_signal.pattern_type.value = 'DOUBLE_BOTTOM'
        mock_signal.timestamp = sample_data.index[0]
        mock_signal.entry_price = 100.0
        mock_signal.stop_loss = 95.0
        mock_signal.target_price = 110.0
        mock_signal.risk_level = 'medium'
        mock_signal.expected_duration = '1-2 weeks'
        mock_signal.signal_strength = 0.8
        mock_signal.metadata = {'key': 'value'}

        processed = engine._process_signal(mock_signal, sample_data)

        assert processed is not None
        assert processed['symbol'] == 'AAPL'
        assert processed['pattern_type'] == 'DOUBLE_BOTTOM'
        assert processed['confidence'] == mock_signal.signal_strength
        assert processed['entry_price'] == 100.0

    def test_process_signal_missing_date(self, engine, sample_data):
        """Test signal processing with missing date"""
        mock_signal = Mock()
        mock_signal.symbol = 'AAPL'
        mock_signal.pattern_type = Mock()
        mock_signal.pattern_type.value = 'DOUBLE_BOTTOM'
        mock_signal.timestamp = datetime(2022, 1, 1)  # Not in sample data
        mock_signal.entry_price = 100.0
        mock_signal.stop_loss = 95.0
        mock_signal.target_price = 110.0

        processed = engine._process_signal(mock_signal, sample_data)

        assert processed is None

    def test_calculate_metrics(self, engine):
        """Test metrics calculation"""
        signals = [
            {
                'confidence': 0.8,
                'potential_return': 0.1,
                'risk_reward_ratio': 2.0,
                'risk_level': 'high',
                'pattern_type': 'DOUBLE_BOTTOM'
            },
            {
                'confidence': 0.6,
                'potential_return': 0.05,
                'risk_reward_ratio': 1.5,
                'risk_level': 'medium',
                'pattern_type': 'FLAG_PATTERN'
            }
        ]

        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [100, 101],
            'volume': [1000000, 2000000]
        }, index=pd.date_range(start='2023-01-01', periods=2, freq='D'))

        metrics = engine._calculate_metrics(signals, data)

        assert metrics['total_signals'] == 2
        assert metrics['avg_confidence'] == 0.7
        assert metrics['avg_potential_return'] == 0.075
        assert metrics['avg_risk_reward_ratio'] == 1.75
        assert metrics['profitable_signals'] == 2
        assert metrics['risk_level_distribution']['high'] == 1
        assert metrics['risk_level_distribution']['medium'] == 1

    @pytest.mark.asyncio
    async def test_get_available_symbols(self, engine):
        """Test getting available symbols"""
        with patch.object(engine, 'data_manager') as mock_data_manager:
            mock_data_manager.get_available_symbols.return_value = ['AAPL', 'MSFT', 'GOOGL']

            symbols = await engine.get_available_symbols()

            assert symbols == ['AAPL', 'MSFT', 'GOOGL']

    def test_validate_inputs_valid(self, engine, test_parameters):
        """Test input validation with valid inputs"""
        errors = await engine.validate_inputs(**test_parameters)
        assert len(errors) == 0

    def test_validate_inputs_invalid_symbol(self, engine, test_parameters):
        """Test input validation with invalid symbol"""
        test_parameters['symbol'] = ''
        errors = await engine.validate_inputs(**test_parameters)
        assert len(errors) > 0

    def test_validate_inputs_invalid_date_range(self, engine, test_parameters):
        """Test input validation with invalid date range"""
        test_parameters['start_date'] = datetime(2023, 4, 1)
        test_parameters['end_date'] = datetime(2023, 1, 1)
        errors = await engine.validate_inputs(**test_parameters)
        assert len(errors) > 0
        assert 'Start date must be before end date' in errors

    def test_validate_inputs_invalid_patterns(self, engine, test_parameters):
        """Test input validation with invalid patterns"""
        test_parameters['pattern_types'] = ['INVALID_PATTERN']
        errors = await engine.validate_inputs(**test_parameters)
        assert len(errors) > 0
        assert 'Invalid pattern types' in errors

    def test_validate_inputs_invalid_confidence(self, engine, test_parameters):
        """Test input validation with invalid confidence threshold"""
        test_parameters['confidence_threshold'] = 1.5
        errors = await engine.validate_inputs(**test_parameters)
        assert len(errors) > 0
        assert 'Confidence threshold must be between 0.0 and 1.0' in errors

    def test_get_supported_patterns(self, engine):
        """Test getting supported patterns"""
        patterns = engine.get_supported_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert 'DOUBLE_BOTTOM' in patterns
        assert 'FLAG_PATTERN' in patterns


class TestProgressManager:
    """Test cases for ProgressManager"""

    @pytest.fixture
    def progress_manager(self):
        """Create a ProgressManager instance"""
        return ProgressManager()

    def test_progress_manager_initialization(self, progress_manager):
        """Test ProgressManager initialization"""
        assert progress_manager.progress_key == "progress_manager"
        assert len(progress_manager.steps) == 0
        assert progress_manager.current_step is None
        assert progress_manager.total_steps == 0

    def test_add_step(self, progress_manager):
        """Test adding a progress step"""
        progress_manager.add_step("step1", "First step")
        progress_manager.add_step("step2", "Second step")

        assert len(progress_manager.steps) == 2
        assert progress_manager.total_steps == 2
        assert "step1" in progress_manager.steps
        assert "step2" in progress_manager.steps

    def test_start_step(self, progress_manager):
        """Test starting a step"""
        progress_manager.add_step("step1", "First step")
        progress_manager.start_step("step1")

        assert progress_manager.current_step == "step1"
        assert progress_manager.steps["step1"].start_time is not None
        assert progress_manager.steps["step1"].progress == 0.0

    def test_update_step(self, progress_manager):
        """Test updating a step"""
        progress_manager.add_step("step1", "First step")
        progress_manager.update_step("step1", 0.5, "Half done")

        assert progress_manager.steps["step1"].progress == 0.5
        assert progress_manager.steps["step1"].description == "Half done"

    def test_complete_step(self, progress_manager):
        """Test completing a step"""
        progress_manager.add_step("step1", "First step")
        progress_manager.complete_step("step1", True)

        assert progress_manager.steps["step1"].progress == 1.0
        assert progress_manager.steps["step1"].completed is True
        assert progress_manager.steps["step1"].end_time is not None

    def test_get_overall_progress(self, progress_manager):
        """Test overall progress calculation"""
        progress_manager.add_step("step1", "First step")
        progress_manager.add_step("step2", "Second step")

        # Halfway through step1
        progress_manager.update_step("step1", 0.5)

        progress = progress_manager.get_overall_progress()
        assert progress == 0.25  # 0.5 / 2 = 0.25

    def test_is_complete(self, progress_manager):
        """Test completion check"""
        progress_manager.add_step("step1", "First step")
        progress_manager.add_step("step2", "Second step")

        assert not progress_manager.is_complete()

        progress_manager.complete_step("step1", True)
        progress_manager.complete_step("step2", True)

        assert progress_manager.is_complete()

    def test_get_elapsed_time(self, progress_manager):
        """Test elapsed time calculation"""
        import time

        progress_manager.start_time = datetime.now()
        time.sleep(0.1)  # Small delay
        elapsed = progress_manager.get_elapsed_time()

        assert isinstance(elapsed, type(progress_manager.start_time - datetime.now()))

    def test_get_estimated_remaining_time(self, progress_manager):
        """Test estimated remaining time calculation"""
        import time

        progress_manager.start_time = datetime.now()
        time.sleep(0.1)

        progress_manager.add_step("step1", "First step")
        progress_manager.start_step("step1")
        progress_manager.update_step("step1", 0.5)

        remaining = progress_manager.get_estimated_remaining_time()
        assert remaining is not None


class TestErrorHandler:
    """Test cases for ErrorHandler"""

    @pytest.fixture
    def error_handler_instance(self):
        """Create an ErrorHandler instance"""
        return ErrorHandler()

    def test_error_handler_initialization(self, error_handler_instance):
        """Test ErrorHandler initialization"""
        assert len(error_handler_instance.errors) == 0
        assert len(error_handler_instance.error_callbacks) == 0

    def test_handle_error(self, error_handler_instance):
        """Test error handling"""
        error = Exception("Test error")
        error_info = error_handler_instance.handle_error(
            error,
            ErrorType.VALIDATION,
            {"field": "test"},
            "User-friendly message",
            ["Suggestion 1", "Suggestion 2"],
            "medium"
        )

        assert error_info.error_type == ErrorType.VALIDATION
        assert error_info.message == "User-friendly message"
        assert len(error_info.suggestions) == 2
        assert error_info.severity == "medium"
        assert len(error_handler_instance.errors) == 1

    def test_display_error(self, error_handler_instance):
        """Test error display"""
        error = Exception("Test error")
        error_info = error_handler_instance.handle_error(
            error,
            ErrorType.VALIDATION,
            {"field": "test"},
            "User-friendly message"
        )

        # This would normally display in Streamlit, but we'll just test the method exists
        assert hasattr(error_handler_instance, 'display_error')

    def test_get_errors_by_type(self, error_handler_instance):
        """Test getting errors by type"""
        error_handler_instance.handle_error(Exception("Error 1"), ErrorType.VALIDATION)
        error_handler_instance.handle_error(Exception("Error 2"), ErrorType.NETWORK)

        validation_errors = error_handler_instance.get_errors_by_type(ErrorType.VALIDATION)
        network_errors = error_handler_instance.get_errors_by_type(ErrorType.NETWORK)

        assert len(validation_errors) == 1
        assert len(network_errors) == 1

    def test_get_errors_by_severity(self, error_handler_instance):
        """Test getting errors by severity"""
        error_handler_instance.handle_error(Exception("Error 1"), ErrorType.VALIDATION, severity="high")
        error_handler_instance.handle_error(Exception("Error 2"), ErrorType.NETWORK, severity="low")

        high_errors = error_handler_instance.get_errors_by_severity("high")
        low_errors = error_handler_instance.get_errors_by_severity("low")

        assert len(high_errors) == 1
        assert len(low_errors) == 1

    def test_clear_errors(self, error_handler_instance):
        """Test clearing errors"""
        error_handler_instance.handle_error(Exception("Error 1"), ErrorType.VALIDATION)
        error_handler_instance.handle_error(Exception("Error 2"), ErrorType.NETWORK)

        assert len(error_handler_instance.errors) == 2

        error_handler_instance.clear_errors(ErrorType.VALIDATION)

        assert len(error_handler_instance.errors) == 1

    def test_clear_all_errors(self, error_handler_instance):
        """Test clearing all errors"""
        error_handler_instance.handle_error(Exception("Error 1"), ErrorType.VALIDATION)
        error_handler_instance.handle_error(Exception("Error 2"), ErrorType.NETWORK)

        error_handler_instance.clear_errors()

        assert len(error_handler_instance.errors) == 0

    def test_get_error_statistics(self, error_handler_instance):
        """Test error statistics"""
        error_handler_instance.handle_error(Exception("Error 1"), ErrorType.VALIDATION, severity="high")
        error_handler_instance.handle_error(Exception("Error 2"), ErrorType.NETWORK, severity="medium")
        error_handler_instance.handle_error(Exception("Error 3"), ErrorType.DATA, severity="high")

        stats = error_handler_instance.get_error_statistics()

        assert stats['total_errors'] == 3
        assert stats['error_types']['validation'] == 1
        assert stats['error_types']['network'] == 1
        assert stats['error_types']['data'] == 1
        assert stats['severity_distribution']['high'] == 2
        assert stats['severity_distribution']['medium'] == 1

    def test_convenience_functions(self):
        """Test convenience error handling functions"""
        # Test validation error
        error_info = handle_validation_error("Test validation error", {"field": "test"})
        assert error_info.error_type == ErrorType.VALIDATION

        # Test network error
        error_info = handle_network_error(Exception("Network error"), {"context": "test"})
        assert error_info.error_type == ErrorType.NETWORK

        # Test data error
        error_info = handle_data_error(Exception("Data error"), "AAPL")
        assert error_info.error_type == ErrorType.DATA

        # Test pattern detection error
        error_info = handle_pattern_detection_error(Exception("Pattern detection error"))
        assert error_info.error_type == ErrorType.PATTERN_DETECTION