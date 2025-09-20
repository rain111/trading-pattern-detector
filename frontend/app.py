"""
Main Streamlit Application - Trading Pattern Detection Frontend
"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import sys
from pathlib import Path

# Add src to path for backend imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frontend.config import settings
from frontend.components import InputForm, ResultsDisplay, PatternSelector
from frontend.integration import PatternDetectionEngine, ProgressManager, error_handler
from frontend.utils import (
    format_currency, format_percentage, format_date, format_confidence,
    validate_stock_symbol, validate_date_range, validate_pattern_selection,
    validate_confidence_threshold, create_error_message, create_success_message
)

class TradingPatternApp:
    """Main application class for the trading pattern detection system"""

    def __init__(self):
        self.logger = self._setup_logging()
        self.engine = PatternDetectionEngine()
        self.input_form = InputForm()
        self.results_display = ResultsDisplay()
        self.pattern_selector = PatternSelector()
        self.session_state = {}

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    def run(self):
        """Run the main application"""
        # Configure Streamlit page
        st.set_page_config(**settings.PAGE_CONFIG)

        # Main app header
        self._render_header()

        # Initialize session state
        self._initialize_session_state()

        # Initialize the pattern detection engine
        self._initialize_engine()

        # Main application layout
        self._render_main_layout()

    def _render_header(self):
        """Render the application header"""
        st.title(f"📈 {settings.APP_NAME}")
        st.markdown(f"**Version:** {settings.APP_VERSION}")
        st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown("---")

    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None

    def _initialize_engine(self):
        """Initialize the pattern detection engine"""
        try:
            asyncio.run(self.engine.initialize())
            st.success("Pattern detection engine initialized successfully")
        except Exception as e:
            error_handler.handle_error(
                e,
                error_handler.ErrorType.CONFIGURATION,
                {"component": "engine"},
                "Failed to initialize pattern detection engine",
                ["Check dependencies are installed", "Restart the application"],
                "critical"
            )
            st.error("Failed to initialize pattern detection engine")

    def _render_main_layout(self):
        """Render the main application layout"""
        # Sidebar for navigation and settings
        self._render_sidebar()

        # Main content area
        self._render_main_content()

        # Results area
        self._render_results_area()

    def _render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("🎯 Trading Pattern Detection")

        # Navigation menu
        options = [
            "📊 New Analysis",
            "📋 Analysis History",
            "⚙️ Settings",
            "📚 Help"
        ]

        choice = st.sidebar.selectbox("Navigation", options)

        # Render selected option
        if choice == "📊 New Analysis":
            self._render_new_analysis()
        elif choice == "📋 Analysis History":
            self._render_analysis_history()
        elif choice == "⚙️ Settings":
            self._render_settings()
        elif choice == "📚 Help":
            self._render_help()

    def _render_new_analysis(self):
        """Render new analysis interface"""
        st.subheader("🔍 New Pattern Analysis")

        # Get form data
        form_data = self.input_form.render_input_section()

        # Validate form
        if form_data['submitted']:
            errors = self._validate_form_data(form_data)
            if errors:
                self.input_form.display_validation_errors(errors)
            else:
                self.input_form.show_success_message(form_data)
                self._run_pattern_analysis(form_data)

    def _validate_form_data(self, form_data: Dict[str, Any]) -> List[str]:
        """Validate form data"""
        errors = []

        # Validate inputs
        errors.extend(validate_stock_symbol(form_data['symbol']))
        errors.extend(validate_date_range(form_data['start_date'], form_data['end_date']))
        errors.extend(validate_pattern_selection(form_data['selected_patterns']))
        errors.extend(validate_confidence_threshold(form_data['confidence_threshold']))

        return errors

    def _run_pattern_analysis(self, form_data: Dict[str, Any]):
        """Run pattern analysis with progress tracking"""
        with ProgressManager("pattern_analysis") as progress:
            try:
                # Add progress steps
                progress.add_step("data_fetching", "Fetching market data...")
                progress.add_step("pattern_detection", "Detecting patterns...")
                progress.add_step("result_processing", "Processing results...")

                # Start data fetching
                progress.start_step("data_fetching")
                progress.update_step("data_fetching", 0.3, "Fetching market data...")

                # Run pattern detection
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                results = loop.run_until_complete(
                    self.engine.detect_patterns(
                        symbol=form_data['symbol'],
                        start_date=form_data['start_date'],
                        end_date=form_data['end_date'],
                        pattern_types=form_data['selected_patterns'],
                        confidence_threshold=form_data['confidence_threshold']
                    )
                )

                # Update progress
                progress.update_step("data_fetching", 1.0, "Data fetching completed")
                progress.complete_step("data_fetching", True)

                progress.start_step("pattern_detection")
                progress.update_step("pattern_detection", 0.5, "Detecting patterns...")

                # Process results
                if results['success']:
                    progress.update_step("pattern_detection", 1.0, "Pattern detection completed")
                    progress.complete_step("pattern_detection", True)

                    progress.start_step("result_processing")
                    progress.update_step("result_processing", 0.5, "Processing results...")

                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.current_analysis = {
                        'form_data': form_data,
                        'results': results,
                        'timestamp': datetime.now(),
                        'duration': progress.get_elapsed_time()
                    }

                    progress.complete_step("result_processing", True)

                    st.success("Pattern analysis completed successfully!")

                else:
                    progress.complete_step("pattern_detection", False, results.get('error', 'Unknown error'))
                    error_handler.handle_error(
                        Exception(results.get('error', 'Pattern detection failed')),
                        error_handler.ErrorType.PATTERN_DETECTION,
                        {'symbol': form_data['symbol']},
                        "Pattern detection failed",
                        ["Try adjusting confidence threshold", "Select different patterns"]
                    )
                    st.error("Pattern detection failed. Please check the error message.")

            except Exception as e:
                progress.complete_step("pattern_detection", False, str(e))
                error_handler.handle_error(
                    e,
                    error_handler.ErrorType.UNKNOWN,
                    {'symbol': form_data['symbol']},
                    "Analysis failed",
                    ["Try again later", "Check your connection"]
                )
                st.error(create_error_message(e, "Pattern analysis failed"))

    def _render_analysis_history(self):
        """Render analysis history"""
        st.subheader("📋 Analysis History")

        if not st.session_state.analysis_history:
            st.info("No analysis history available")
            return

        # Display history items
        for i, analysis in enumerate(st.session_state.analysis_history[-10:], 1):  # Show last 10
            with st.expander(f"Analysis {i}: {analysis['form_data']['symbol']} - {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"**Symbol:** {analysis['form_data']['symbol']}")
                st.write(f"**Date Range:** {analysis['form_data']['start_date']} to {analysis['form_data']['end_date']}")
                st.write(f"**Patterns:** {', '.join(analysis['form_data']['selected_patterns'])}")
                st.write(f**Confidence Threshold:** {analysis['form_data']['confidence_threshold']:.2f}")
                st.write(f"**Duration:** {analysis['duration']}")

                if analysis['results']['success']:
                    st.write(f"**Signals Found:** {len(analysis['results']['signals'])}")
                    st.write(f"**Average Confidence:** {format_currency(analysis['results']['metrics']['avg_confidence'])}")
                else:
                    st.write(f"**Error:** {analysis['results'].get('error', 'Unknown error')}")

    def _render_settings(self):
        """Render settings page"""
        st.subheader("⚙️ Settings")

        # Data settings
        st.write("**Data Settings:**")
        data_timeout = st.slider(
            "Data Fetch Timeout (seconds)",
            min_value=10,
            max_value=60,
            value=settings.DATA_FETCH_TIMEOUT
        )

        # UI settings
        st.write("**UI Settings:**")
        max_results = st.slider(
            "Max Results Display",
            min_value=10,
            max_value=500,
            value=settings.MAX_RESULTS_DISPLAY
        )

        # Cache settings
        st.write("**Cache Settings:**")
        cache_timeout = st.slider(
            "Cache Timeout (hours)",
            min_value=1,
            max_value=24,
            value=1
        )

        # Save settings
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

    def _render_help(self):
        """Render help page"""
        st.subheader("📚 Help")

        # Pattern detection guide
        with st.expander("🔍 Pattern Detection Guide", expanded=True):
            st.markdown("""
            ### How to Use Pattern Detection

            **1. Enter Stock Symbol**
            - Enter a valid stock symbol (e.g., AAPL, MSFT, GOOGL)
            - Symbol should be alphanumeric and 10 characters or less

            **2. Select Date Range**
            - Choose start and end dates for analysis
            - Minimum 30 days, maximum 10 years
            - Historical data will be fetched automatically

            **3. Select Patterns**
            - Choose which patterns to detect
            - Each pattern type has specific characteristics
            - Multiple patterns can be selected

            **4. Set Confidence Threshold**
            - Minimum confidence level for pattern detection
            - Higher threshold = fewer, more reliable patterns
            - Lower threshold = more patterns, potentially less reliable

            **5. Run Analysis**
            - Click "Run Pattern Detection" to start
            - Progress indicators will show analysis progress
            - Results will be displayed when complete

            **Understanding Results**
            - Each signal shows entry, stop loss, and target prices
            - Confidence indicates pattern reliability
            - Risk level shows potential risk level
            - Potential return shows expected gain
            """)

        # Data sources
        with st.expander("📊 Data Sources", expanded=False):
            st.markdown("""
            ### Data Sources and Storage

            **Market Data Source:**
            - Yahoo Finance (yfinance API)
            - Real-time market data
            - Historical OHLCV data

            **Data Storage:**
            - Local Parquet files (data/parquet/)
            - One file per symbol
            - Automatic data caching
            - Append-only updates

            **Data Management:**
            - Data is cached in memory for performance
            - Parquet files are stored locally
            - Missing data is fetched automatically
            - Data validation ensures quality
            """)

        # Technical patterns
        self.pattern_selector.create_pattern_help_section()

        # Error handling
        with st.expander("🚨 Error Handling", expanded=False):
            st.markdown("""
            ### Common Errors and Solutions

            **Network Errors:**
            - Check internet connection
            - Try again in a few moments
            - Verify Yahoo Finance API availability

            **Data Errors:**
            - Check if symbol is valid
            - Try different date range
            - Verify data availability

            **Pattern Detection Errors:**
            - Adjust confidence threshold
            - Select different patterns
            - Extend date range

            **Configuration Errors:**
            - Check dependencies
            - Verify system requirements
            - Restart application
            """)

    def _render_main_content(self):
        """Render main content area"""
        if st.session_state.current_analysis:
            # Show current analysis summary
            analysis = st.session_state.current_analysis
            st.subheader("📊 Current Analysis Summary")

            with st.expander(f"Analysis Summary: {analysis['form_data']['symbol']}", expanded=True):
                st.write(f"**Symbol:** {analysis['form_data']['symbol']}")
                st.write(f"**Date Range:** {analysis['form_data']['start_date']} to {analysis['form_data']['end_date']}")
                st.write(f"**Patterns Selected:** {len(analysis['form_data']['selected_patterns'])}")
                st.write(f"**Confidence Threshold:** {analysis['form_data']['confidence_threshold']:.2f}")
                st.write(f"**Analysis Duration:** {analysis['duration']}")

                if analysis['results']['success']:
                    st.write(f"**Signals Found:** {len(analysis['results']['signals'])}")
                    metrics = analysis['results']['metrics']
                    st.write(f"**Average Confidence:** {format_currency(metrics['avg_confidence'])}")
                    st.write(f"**Average Return:** {format_percentage(metrics['avg_potential_return'])}")
                    st.write(f"**High Confidence Signals:** {metrics['high_confidence_signals']}")

        else:
            # Show welcome screen
            st.subheader("🎯 Welcome to Trading Pattern Detection")

            col1, col2 = st.columns(2)

            with col1:
                st.write("""
                ### Features:
                - **12 Technical Patterns**: Detect various chart patterns
                - **Real-time Data**: Fetch data from Yahoo Finance
                - **Smart Caching**: Efficient data management
                - **Comprehensive Analysis**: Detailed pattern analysis
                - **Export Results**: Download results in multiple formats
                """)

            with col2:
                st.write("""
                ### Supported Patterns:
                - **Reversal**: Double Bottom, Head and Shoulders, Rounding Bottom
                - **Continuation**: Flag, Cup & Handle, Triangles, Wedges
                - **Breakout**: VCP Breakout patterns
                """)

            st.write("""
            ### Getting Started:
            1. Select "📊 New Analysis" from the sidebar
            2. Enter a stock symbol (e.g., AAPL)
            3. Choose your date range
            4. Select patterns to detect
            5. Set confidence threshold
            6. Run analysis and view results
            """)

    def _render_results_area(self):
        """Render results area"""
        if st.session_state.results and st.session_state.current_analysis:
            st.subheader("🎯 Pattern Detection Results")

            # Display results
            results = st.session_state.results
            form_data = st.session_state.current_analysis['form_data']

            if results['success']:
                self.results_display.display_results(results, form_data)
            else:
                st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")

def main():
    """Main function to run the application"""
    try:
        app = TradingPatternApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logging.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()