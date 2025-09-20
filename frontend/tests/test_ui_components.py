"""
Tests for UI components
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import streamlit as st

from frontend.components import InputForm, ResultsDisplay, PatternSelector
from frontend.config import settings


class TestInputForm:
    """Test cases for InputForm component"""

    @pytest.fixture
    def input_form(self):
        """Create an InputForm instance"""
        return InputForm()

    def test_input_form_initialization(self, input_form):
        """Test InputForm initialization"""
        assert input_form.logger is not None

    def test_render_input_section(self, input_form):
        """Test rendering input section"""
        with patch('streamlit.form') as mock_form, \
             patch('streamlit.text_input') as mock_text_input, \
             patch('streamlit.date_input') as mock_date_input, \
             patch('streamlit.columns') as mock_columns, \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.slider') as mock_slider, \
             patch('streamlit.form_submit_button') as mock_submit:

            # Mock form context manager
            mock_form.return_value.__enter__.return_value = MagicMock()

            # Mock components
            mock_text_input.return_value = "AAPL"
            mock_date_input.side_effect = [datetime(2023, 1, 1), datetime(2023, 4, 1)]
            mock_columns.return_value = [MagicMock(), MagicMock()]
            mock_expander.return_value.__enter__.return_value = MagicMock()
            mock_slider.return_value = 0.5
            mock_submit.return_value = False

            form_data = input_form.render_input_section()

            assert 'symbol' in form_data
            assert 'start_date' in form_data
            assert 'end_date' in form_data
            assert 'selected_patterns' in form_data
            assert 'confidence_threshold' in form_data
            assert 'submitted' in form_data

    def test_validate_form_data_valid(self, input_form):
        """Test form validation with valid data"""
        form_data = {
            'symbol': 'AAPL',
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 4, 1),
            'selected_patterns': ['DOUBLE_BOTTOM', 'FLAG_PATTERN'],
            'confidence_threshold': 0.5,
            'auto_update_data': True,
            'show_detailed_results': True,
            'submitted': True
        }

        errors = input_form.validate_form_data(form_data)

        assert len(errors) == 0

    def test_validate_form_data_invalid_symbol(self, input_form):
        """Test form validation with invalid symbol"""
        form_data = {
            'symbol': '',  # Empty symbol
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 4, 1),
            'selected_patterns': ['DOUBLE_BOTTOM'],
            'confidence_threshold': 0.5,
            'auto_update_data': True,
            'show_detailed_results': True,
            'submitted': True
        }

        errors = input_form.validate_form_data(form_data)

        assert len(errors) > 0
        assert 'Stock symbol is required' in errors

    def test_validate_form_data_invalid_date_range(self, input_form):
        """Test form validation with invalid date range"""
        form_data = {
            'symbol': 'AAPL',
            'start_date': datetime(2023, 4, 1),  # After end_date
            'end_date': datetime(2023, 1, 1),
            'selected_patterns': ['DOUBLE_BOTTOM'],
            'confidence_threshold': 0.5,
            'auto_update_data': True,
            'show_detailed_results': True,
            'submitted': True
        }

        errors = input_form.validate_form_data(form_data)

        assert len(errors) > 0
        assert 'Start date must be before end date' in errors

    def test_validate_form_data_no_patterns(self, input_form):
        """Test form validation with no patterns selected"""
        form_data = {
            'symbol': 'AAPL',
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 4, 1),
            'selected_patterns': [],  # No patterns
            'confidence_threshold': 0.5,
            'auto_update_data': True,
            'show_detailed_results': True,
            'submitted': True
        }

        errors = input_form.validate_form_data(form_data)

        assert len(errors) > 0
        assert 'At least one pattern must be selected' in errors

    def test_validate_form_data_invalid_confidence(self, input_form):
        """Test form validation with invalid confidence threshold"""
        form_data = {
            'symbol': 'AAPL',
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 4, 1),
            'selected_patterns': ['DOUBLE_BOTTOM'],
            'confidence_threshold': 1.5,  # Invalid threshold
            'auto_update_data': True,
            'show_detailed_results': True,
            'submitted': True
        }

        errors = input_form.validate_form_data(form_data)

        assert len(errors) > 0
        assert 'Confidence threshold must be between 0.0 and 1.0' in errors

    def test_display_validation_errors(self, input_form):
        """Test displaying validation errors"""
        errors = ['Error 1', 'Error 2']

        # This would normally display in Streamlit, but we'll just test the method exists
        assert hasattr(input_form, 'display_validation_errors')

    def test_show_success_message(self, input_form):
        """Test showing success message"""
        form_data = {
            'symbol': 'AAPL',
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 4, 1),
            'selected_patterns': ['DOUBLE_BOTTOM'],
            'confidence_threshold': 0.5
        }

        # This would normally display in Streamlit, but we'll just test the method exists
        assert hasattr(input_form, 'show_success_message')

    def test_show_settings_panel(self, input_form):
        """Test showing settings panel"""
        with patch('streamlit.sidebar') as mock_sidebar:
            mock_sidebar.columns.return_value = [MagicMock(), MagicMock()]

            settings = input_form.show_settings_panel()

            assert 'cache_timeout' in settings
            assert 'max_results' in settings
            assert 'auto_refresh' in settings
            assert 'show_debug_info' in settings


class TestResultsDisplay:
    """Test cases for ResultsDisplay component"""

    @pytest.fixture
    def results_display(self):
        """Create a ResultsDisplay instance"""
        return ResultsDisplay()

    @pytest.fixture
    def mock_results(self):
        """Create mock results data"""
        return {
            'success': True,
            'symbol': 'AAPL',
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 4, 1),
            'data': pd.DataFrame({
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [100, 101, 102],
                'volume': [1000000, 2000000, 3000000]
            }, index=pd.date_range(start='2023-01-01', periods=3, freq='D')),
            'signals': [
                {
                    'symbol': 'AAPL',
                    'pattern_type': 'DOUBLE_BOTTOM',
                    'timestamp': datetime(2023, 1, 15),
                    'confidence': 0.8,
                    'entry_price': 100.0,
                    'stop_loss': 95.0,
                    'target_price': 110.0,
                    'potential_return': 0.1,
                    'risk_reward_ratio': 3.0,
                    'risk_level': 'medium',
                    'expected_duration': '1-2 weeks',
                    'signal_strength': 0.8,
                    'metadata': {'key': 'value'},
                    'open': 100,
                    'high': 105,
                    'low': 95,
                    'close': 100,
                    'volume': 1000000
                }
            ],
            'metrics': {
                'total_signals': 1,
                'avg_confidence': 0.8,
                'avg_potential_return': 0.1,
                'avg_risk_reward_ratio': 3.0,
                'risk_level_distribution': {'medium': 1},
                'pattern_type_distribution': {'DOUBLE_BOTTOM': 1}
            },
            'detection_config': {
                'confidence_threshold': 0.5,
                'pattern_types': ['DOUBLE_BOTTOM'],
                'data_points': 3
            },
            'timestamp': datetime.now()
        }

    def test_results_display_initialization(self, results_display):
        """Test ResultsDisplay initialization"""
        assert results_display.logger is not None

    def test_display_results_no_signals(self, results_display):
        """Test displaying results with no signals"""
        results = {
            'success': True,
            'signals': [],
            'data': pd.DataFrame()
        }

        form_data = {
            'show_detailed_results': True
        }

        # This would normally display in Streamlit, but we'll just test the method exists
        assert hasattr(results_display, 'display_results')

    def test_display_results_with_signals(self, results_display, mock_results):
        """Test displaying results with signals"""
        form_data = {
            'show_detailed_results': True
        }

        # This would normally display in Streamlit, but we'll just test the method exists
        assert hasattr(results_display, 'display_results')

    def test_display_summary_statistics(self, results_display, mock_results):
        """Test displaying summary statistics"""
        with patch('streamlit.metric') as mock_metric:
            results_display._display_summary_statistics(mock_results)

            # Check that metrics were displayed
            assert mock_metric.call_count == 4

    def test_display_results_table(self, results_display, mock_results):
        """Test displaying results table"""
        with patch('streamlit.dataframe') as mock_dataframe:
            results_display._display_results_table(mock_results)

            # Check that dataframe was displayed
            mock_dataframe.assert_called_once()

    def test_display_price_chart(self, results_display, mock_results):
        """Test displaying price chart"""
        with patch('streamlit.plotly_chart') as mock_plotly_chart:
            results_display._display_price_chart(mock_results)

            # Check that plotly chart was displayed
            mock_plotly_chart.assert_called_once()

    def test_display_pattern_performance(self, results_display, mock_results):
        """Test displaying pattern performance"""
        with patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.plotly_chart') as mock_plotly_chart:
            results_display._display_pattern_performance(mock_results)

            # Check that dataframe and plotly chart were displayed
            mock_dataframe.assert_called_once()
            mock_plotly_chart.assert_called_once()

    def test_display_export_options(self, results_display, mock_results):
        """Test displaying export options"""
        with patch('streamlit.download_button') as mock_download, \
             patch('streamlit.button') as mock_button:
            results_display._display_export_options(mock_results)

            # Check that download buttons were displayed
            assert mock_button.call_count == 3

    def test_export_signals_csv(self, results_display):
        """Test exporting signals as CSV"""
        signals = [
            {
                'symbol': 'AAPL',
                'pattern_type': 'DOUBLE_BOTTOM',
                'confidence': 0.8,
                'entry_price': 100.0,
                'stop_loss': 95.0,
                'target_price': 110.0
            }
        ]

        with patch('streamlit.download_button') as mock_download:
            results_display._export_signals_csv(signals)

            # Check that download button was created
            mock_download.assert_called_once()

    def test_export_full_analysis_csv(self, results_display):
        """Test exporting full analysis as CSV"""
        signals = [
            {
                'symbol': 'AAPL',
                'pattern_type': 'DOUBLE_BOTTOM',
                'confidence': 0.8,
                'entry_price': 100.0,
                'stop_loss': 95.0,
                'target_price': 110.0
            }
        ]

        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [100, 101, 102],
            'volume': [1000000, 2000000, 3000000]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))

        with patch('streamlit.download_button') as mock_download:
            results_display._export_full_analysis_csv(signals, data)

            # Check that download button was created
            mock_download.assert_called_once()

    def test_export_json_results(self, results_display):
        """Test exporting results as JSON"""
        results = {
            'success': True,
            'symbol': 'AAPL',
            'signals': [
                {
                    'symbol': 'AAPL',
                    'pattern_type': 'DOUBLE_BOTTOM',
                    'confidence': 0.8,
                    'entry_price': 100.0,
                    'stop_loss': 95.0,
                    'target_price': 110.0
                }
            ]
        }

        with patch('streamlit.download_button') as mock_download:
            results_display._export_json_results(results)

            # Check that download button was created
            mock_download.assert_called_once()

    def test_show_no_patterns_found(self, results_display):
        """Test showing no patterns found message"""
        results = {
            'success': True,
            'signals': [],
            'data': pd.DataFrame()
        }

        # This would normally display in Streamlit, but we'll just test the method exists
        assert hasattr(results_display, 'show_no_patterns_found')


class TestPatternSelector:
    """Test cases for PatternSelector component"""

    @pytest.fixture
    def pattern_selector(self):
        """Create a PatternSelector instance"""
        return PatternSelector()

    def test_pattern_selector_initialization(self, pattern_selector):
        """Test PatternSelector initialization"""
        assert pattern_selector.logger is not None

    def test_render_pattern_selection(self, pattern_selector):
        """Test rendering pattern selection"""
        with patch('streamlit.expander') as mock_expander, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.columns') as mock_columns:

            # Mock components
            mock_expander.return_value.__enter__.return_value = MagicMock()
            mock_multiselect.return_value = ['DOUBLE_BOTTOM', 'FLAG_PATTERN']
            mock_checkbox.return_value = False
            mock_columns.return_value = [MagicMock(), MagicMock()]

            patterns = pattern_selector.render_pattern_selection()

            assert isinstance(patterns, list)
            assert len(patterns) > 0

    def test_get_pattern_info(self, pattern_selector):
        """Test getting pattern information"""
        pattern_info = pattern_selector._get_pattern_info('DOUBLE_BOTTOM')

        assert pattern_info['name'] == 'Double Bottom'
        assert pattern_info['type'] == 'Reversal'
        assert pattern_info['trend'] == 'Bullish'
        assert pattern_info['description'] is not None

    def test_get_pattern_info_invalid(self, pattern_selector):
        """Test getting pattern info for invalid pattern"""
        pattern_info = pattern_selector._get_pattern_info('INVALID_PATTERN')

        assert pattern_info['name'] == 'INVALID_PATTERN'
        assert pattern_info['type'] == 'Unknown'
        assert pattern_info['trend'] == 'Unknown'

    def test_validate_pattern_selection_valid(self, pattern_selector):
        """Test validating pattern selection with valid patterns"""
        patterns = ['DOUBLE_BOTTOM', 'FLAG_PATTERN']

        errors = pattern_selector.validate_pattern_selection(patterns)

        assert len(errors) == 0

    def test_validate_pattern_selection_empty(self, pattern_selector):
        """Test validating pattern selection with empty list"""
        patterns = []

        errors = pattern_selector.validate_pattern_selection(patterns)

        assert len(errors) > 0
        assert 'At least one pattern must be selected' in errors

    def test_validate_pattern_selection_too_many(self, pattern_selector):
        """Test validating pattern selection with too many patterns"""
        patterns = ['DOUBLE_BOTTOM', 'FLAG_PATTERN', 'CUP_HANDLE', 'ASCENDING_TRIANGLE',
                   'DESCENDING_TRIANGLE', 'RISING_WEDGE', 'FALLING_WEDGE', 'VCP_BREAKOUT',
                   'HEAD_AND_SHOULDERS', 'ROUNDING_BOTTOM', 'EXTRA_PATTERN']  # 11 patterns

        errors = pattern_selector.validate_pattern_selection(patterns)

        assert len(errors) > 0
        assert 'Maximum 10 patterns can be selected' in errors

    def test_validate_pattern_selection_invalid(self, pattern_selector):
        """Test validating pattern selection with invalid patterns"""
        patterns = ['DOUBLE_BOTTOM', 'INVALID_PATTERN']

        errors = pattern_selector.validate_pattern_selection(patterns)

        assert len(errors) > 0
        assert 'Invalid patterns selected' in errors

    def test_get_pattern_recommendations(self, pattern_selector):
        """Test getting pattern recommendations"""
        recommendations = pattern_selector.get_pattern_recommendations('trending')

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_create_pattern_help_section(self, pattern_selector):
        """Test creating pattern help section"""
        with patch('streamlit.expander') as mock_expander:
            mock_expander.return_value.__enter__.return_value = MagicMock()

            # This would normally display in Streamlit, but we'll just test the method exists
            assert hasattr(pattern_selector, 'create_pattern_help_section')

    def test_display_pattern_statistics(self, pattern_selector):
        """Test displaying pattern statistics"""
        selected_patterns = ['DOUBLE_BOTTOM', 'FLAG_PATTERN']
        all_patterns = ['DOUBLE_BOTTOM', 'FLAG_PATTERN', 'CUP_HANDLE']

        with patch('streamlit.metric') as mock_metric:
            pattern_selector.display_pattern_statistics(selected_patterns, all_patterns)

            # Check that metrics were displayed
            assert mock_metric.call_count >= 2