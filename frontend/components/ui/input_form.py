"""
Input Form UI Component - Handles user input forms and validation
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

from ...config import settings

class InputForm:
    """Handles user input forms and validation"""

    def __init__(self):
        self.logger = st.logger.get_logger(__name__)

    def render_input_section(self) -> Dict[str, Any]:
        """Render the main input section and return form data"""
        with st.form("pattern_detection_form"):
            # Symbol input
            st.subheader("📊 Stock Selection")
            symbol = st.text_input(
                "Stock Symbol",
                value="AAPL",
                placeholder="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)",
                help="Enter the stock symbol you want to analyze"
            )

            # Date range selection
            st.subheader("📅 Date Range")
            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.strptime(settings.DEFAULT_START_DATE, '%Y-%m-%d'),
                    min_value=datetime(2010, 1, 1),
                    max_value=datetime.now() - timedelta(days=1),
                    help="Select the start date for analysis"
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.strptime(settings.DEFAULT_END_DATE, '%Y-%m-%d'),
                    min_value=start_date,
                    max_value=datetime.now(),
                    help="Select the end date for analysis"
                )

            # Pattern selection
            st.subheader("🔍 Pattern Detection")
            st.markdown("**Select patterns to detect:**")

            # Pattern selection with categories
            pattern_categories = {
                "Reversal Patterns": [
                    "DOUBLE_BOTTOM",
                    "HEAD_AND_SHOULDERS",
                    "ROUNDING_BOTTOM"
                ],
                "Continuation Patterns": [
                    "FLAG_PATTERN",
                    "CUP_HANDLE",
                    "ASCENDING_TRIANGLE",
                    "DESCENDING_TRIANGLE",
                    "RISING_WEDGE",
                    "FALLING_WEDGE"
                ],
                "Breakout Patterns": [
                    "VCP_BREAKOUT"
                ]
            }

            selected_patterns = []
            for category, patterns in pattern_categories.items():
                with st.expander(f"📋 {category} ({len(patterns)} patterns)", expanded=True):
                    selected = st.multiselect(
                        f"Select {category.lower()} patterns",
                        options=patterns,
                        default=patterns,
                        help=f"Choose which {category.lower()} patterns to detect"
                    )
                    selected_patterns.extend(selected)

            # Confidence threshold
            st.subheader("⚙️ Detection Settings")
            col1, col2 = st.columns(2)

            with col1:
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=settings.DEFAULT_CONFIDENCE_THRESHOLD,
                    step=0.05,
                    help="Minimum confidence score for pattern detection (0.0-1.0)"
                )

            with col2:
                st.markdown("**Confidence Level:**")
                if confidence_threshold >= 0.8:
                    st.success("High confidence threshold (>0.8)")
                elif confidence_threshold >= 0.5:
                    st.info("Medium confidence threshold (0.5-0.8)")
                else:
                    st.warning("Low confidence threshold (<0.5)")

            # Additional options
            st.subheader("🎯 Additional Options")
            col1, col2 = st.columns(2)

            with col1:
                auto_update_data = st.checkbox(
                    "Auto-update data",
                    value=True,
                    help="Automatically fetch and update market data"
                )

            with col2:
                show_detailed_results = st.checkbox(
                    "Show detailed results",
                    value=True,
                    help="Display detailed information about detected patterns"
                )

            # Submit button
            submitted = st.form_submit_button(
                "🚀 Run Pattern Detection",
                type="primary",
                use_container_width=True
            )

            # Collect form data
            form_data = {
                'symbol': symbol.upper().strip(),
                'start_date': start_date,
                'end_date': end_date,
                'selected_patterns': selected_patterns,
                'confidence_threshold': confidence_threshold,
                'auto_update_data': auto_update_data,
                'show_detailed_results': show_detailed_results,
                'submitted': submitted
            }

            return form_data

    def validate_form_data(self, form_data: Dict[str, Any]) -> List[str]:
        """Validate form data and return list of errors"""
        errors = []

        # Validate symbol
        if not form_data['symbol']:
            errors.append("Stock symbol is required")
        elif not form_data['symbol'].isalnum():
            errors.append("Stock symbol should contain only letters and numbers")

        # Validate date range
        if form_data['start_date'] >= form_data['end_date']:
            errors.append("Start date must be before end date")

        # Validate date range length
        date_range_days = (form_data['end_date'] - form_data['start_date']).days
        if date_range_days > 365 * 10:  # 10 years max
            errors.append("Date range cannot exceed 10 years")
        elif date_range_days < 30:  # 1 month min
            errors.append("Date range must be at least 30 days")

        # Validate pattern selection
        if not form_data['selected_patterns']:
            errors.append("At least one pattern must be selected")

        # Validate confidence threshold
        if not 0.0 <= form_data['confidence_threshold'] <= 1.0:
            errors.append("Confidence threshold must be between 0.0 and 1.0")

        return errors

    def display_validation_errors(self, errors: List[str]):
        """Display validation errors"""
        if errors:
            st.error("⚠️ Please fix the following errors:")
            for error in errors:
                st.error(f"• {error}")

    def show_success_message(self, form_data: Dict[str, Any]):
        """Show success message with summary"""
        st.success("✅ Form validated successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Symbol:** {form_data['symbol']}")
            st.info(f"**Date Range:** {form_data['start_date']} to {form_data['end_date']}")

        with col2:
            st.info(f"**Patterns:** {len(form_data['selected_patterns'])} selected")
            st.info(f"**Confidence:** {form_data['confidence_threshold']:.2f}")

    @staticmethod
    def create_pattern_help_section():
        """Create pattern help section"""
        with st.expander("📚 Pattern Descriptions", expanded=False):
            st.markdown("""
            ### Available Patterns:

            **Reversal Patterns:**
            - **Double Bottom:** Bullish reversal pattern formed by two consecutive lows
            - **Head and Shoulders:** Bearish reversal pattern with three peaks
            - **Rounding Bottom:** Bullish reversal pattern showing gradual trend change

            **Continuation Patterns:**
            - **Flag Pattern:** Short consolidation followed by trend continuation
            - **Cup and Handle:** Bullish continuation pattern with cup and consolidation
            - **Ascending Triangle:** Bullish continuation pattern with higher highs
            - **Descending Triangle:** Bearish continuation pattern with lower lows
            - **Rising Wedge:** Bearish continuation pattern with tightening range
            - **Falling Wedge:** Bullish continuation pattern with tightening range

            **Breakout Patterns:**
            - **VCP Breakout:** Bullish breakout from volatility contraction pattern
            """)

    @staticmethod
    def show_settings_panel():
        """Show settings panel"""
        with st.sidebar:
            st.markdown("⚙️ Settings")

            # Data settings
            st.markdown("**Data Settings:**")
            cache_timeout = st.slider(
                "Cache Timeout (hours)",
                min_value=1,
                max_value=24,
                value=1,
                help="How long to cache data in memory"
            )

            # Display settings
            st.markdown("**Display Settings:**")
            max_results = st.slider(
                "Max Results Display",
                min_value=10,
                max_value=1000,
                value=settings.MAX_RESULTS_DISPLAY,
                help="Maximum number of results to display"
            )

            # Advanced settings
            if st.checkbox("Advanced Settings"):
                st.markdown("**Advanced Options:**")
                auto_refresh = st.checkbox(
                    "Auto-refresh data",
                    value=False,
                    help="Automatically refresh data every hour"
                )

                show_debug_info = st.checkbox(
                    "Show debug information",
                    value=False,
                    help="Display detailed debug information"
                )

            return {
                'cache_timeout': cache_timeout,
                'max_results': max_results,
                'auto_refresh': auto_refresh if 'auto_refresh' in locals() else False,
                'show_debug_info': show_debug_info if 'show_debug_info' in locals() else False
            }