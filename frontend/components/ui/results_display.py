"""
Results Display Component - Handles pattern detection results display
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ...config import settings

class ResultsDisplay:
    """Handles display of pattern detection results"""

    def __init__(self):
        self.logger = st.logger.get_logger(__name__)

    def display_results(self, results: Dict[str, Any], form_data: Dict[str, Any]):
        """Display pattern detection results"""
        st.subheader("🎯 Pattern Detection Results")

        if not results or not results.get('signals'):
            st.warning("No patterns detected in the specified date range.")
            self._show_no_patterns_found()
            return

        # Summary statistics
        self._display_summary_statistics(results)

        # Results table
        if form_data.get('show_detailed_results', True):
            st.subheader("📊 Detected Patterns")
            self._display_results_table(results)

        # Visualization
        st.subheader("📈 Price Chart with Patterns")
        self._display_price_chart(results)

        # Pattern performance analysis
        st.subheader("📈 Pattern Performance Analysis")
        self._display_pattern_performance(results)

        # Export options
        st.subheader("💾 Export Results")
        self._display_export_options(results)

    def _display_summary_statistics(self, results: Dict[str, Any]):
        """Display summary statistics"""
        signals = results['signals']
        data = results['data']

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Signals",
                len(signals),
                f"{len([s for s in signals if s.get('risk_level') == 'high'])} High Risk"
            )

        with col2:
            avg_confidence = np.mean([s.get('confidence', 0) for s in signals])
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.2f}",
                f"{avg_confidence - settings.DEFAULT_CONFIDENCE_THRESHOLD:.2f} vs threshold"
            )

        with col3:
            profitable_signals = len([s for s in signals if s.get('potential_return', 0) > 0])
            st.metric(
                "Profitable Patterns",
                f"{profitable_signals}",
                f"{profitable_signals/len(signals)*100:.1f}%" if signals else "0%"
            )

        with col4:
            avg_return = np.mean([s.get('potential_return', 0) for s in signals]) if signals else 0
            st.metric(
                "Avg Potential Return",
                f"{avg_return:.1%}",
                f"{avg_return*100:.1f}%"
            )

    def _display_results_table(self, results: Dict[str, Any]):
        """Display results in a table format"""
        signals = results['signals']

        # Create DataFrame for display
        df_data = []
        for signal in signals:
            df_data.append({
                'Pattern': signal.get('pattern_type', 'Unknown'),
                'Date': signal.get('timestamp', datetime.now()).strftime('%Y-%m-%d'),
                'Confidence': f"{signal.get('confidence', 0):.2f}",
                'Entry Price': f"${signal.get('entry_price', 0):.2f}",
                'Stop Loss': f"${signal.get('stop_loss', 0):.2f}",
                'Target Price': f"${signal.get('target_price', 0):.2f}",
                'Risk/Reward': f"{signal.get('target_price', 0)/signal.get('stop_loss', 1):.2f}:1" if signal.get('stop_loss', 0) > 0 else "N/A",
                'Potential Return': f"{signal.get('potential_return', 0):.1%}",
                'Risk Level': signal.get('risk_level', 'Unknown').upper(),
                'Duration': signal.get('expected_duration', 'Unknown')
            })

        if df_data:
            df = pd.DataFrame(df_data)

            # Display table with pagination
            st.dataframe(
                df,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Confidence level of the pattern detection",
                        format="%.2f",
                        min_value=0,
                        max_value=1
                    ),
                    "Potential Return": st.column_config.ProgressColumn(
                        "Return",
                        help="Expected return percentage",
                        format="%.1f%%",
                        min_value=0,
                        max_value=1
                    )
                },
                hide_index=True,
                use_container_width=True
            )

            # Pattern count by type
            st.subheader("📋 Pattern Distribution")
            pattern_counts = df['Pattern'].value_counts()
            fig = px.bar(
                x=pattern_counts.values,
                y=pattern_counts.index,
                title="Pattern Detection Count by Type",
                labels={'x': 'Count', 'y': 'Pattern Type'},
                color=pattern_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(width=800, height=400)
            st.plotly_chart(fig, use_container_width=True)

    def _display_price_chart(self, results: Dict[str, Any]):
        """Display price chart with detected patterns"""
        data = results['data']
        signals = results['signals']

        if data.empty:
            st.warning("No data available for chart")
            return

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Price Action with Patterns', 'Volume')
        )

        # Add price data
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=1
        )

        # Add pattern signals
        for signal in signals:
            signal_date = signal.get('timestamp')
            if signal_date and signal_date in data.index:
                signal_data = data.loc[signal_date]

                # Determine signal color based on pattern type
                if signal.get('pattern_type') in ['DOUBLE_BOTTOM', 'ROUNDING_BOTTOM', 'ASCENDING_TRIANGLE', 'FALLING_WEDGE']:
                    color = 'green'
                else:
                    color = 'red'

                # Add entry point
                fig.add_trace(
                    go.Scatter(
                        x=[signal_date],
                        y=[signal_data['close']],
                        mode='markers',
                        name=f"{signal.get('pattern_type', 'Unknown')} Entry",
                        marker=dict(color=color, size=10, symbol='triangle-up'),
                        text=f"{signal.get('pattern_type', 'Unknown')}<br>Conf: {signal.get('confidence', 0):.2f}",
                        textposition="top center"
                    ),
                    row=1, col=1
                )

                # Add stop loss line
                fig.add_hline(
                    y=signal.get('stop_loss', 0),
                    line=dict(color='red', width=1, dash='dash'),
                    annotation_text=f"Stop Loss: ${signal.get('stop_loss', 0):.2f}",
                    row=1, col=1
                )

                # Add target line
                fig.add_hline(
                    y=signal.get('target_price', 0),
                    line=dict(color='green', width=1, dash='dash'),
                    annotation_text=f"Target: ${signal.get('target_price', 0):.2f}",
                    row=1, col=1
                )

        # Update layout
        fig.update_layout(
            title=f"Price Chart - {results.get('symbol', 'Unknown')}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )

        fig.update_yaxes(title_text="Volume", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    def _display_pattern_performance(self, results: Dict[str, Any]):
        """Display pattern performance analysis"""
        signals = results['signals']

        # Performance by pattern type
        performance_by_pattern = {}
        for signal in signals:
            pattern_type = signal.get('pattern_type', 'Unknown')
            if pattern_type not in performance_by_pattern:
                performance_by_pattern[pattern_type] = []
            performance_by_pattern[pattern_type].append(signal)

        # Create performance metrics
        pattern_stats = []
        for pattern_type, pattern_signals in performance_by_pattern.items():
            if pattern_signals:
                avg_confidence = np.mean([s.get('confidence', 0) for s in pattern_signals])
                avg_return = np.mean([s.get('potential_return', 0) for s in pattern_signals])
                count = len(pattern_signals)

                pattern_stats.append({
                    'Pattern': pattern_type,
                    'Count': count,
                    'Avg Confidence': f"{avg_confidence:.2f}",
                    'Avg Return': f"{avg_return:.1%}",
                    'Total Signals': count
                })

        if pattern_stats:
            df_pattern = pd.DataFrame(pattern_stats)
            st.dataframe(df_pattern, use_container_width=True, hide_index=True)

            # Performance visualization
            fig = px.scatter(
                df_pattern,
                x='Avg Confidence',
                y='Avg Return',
                size='Count',
                text='Pattern',
                title="Pattern Performance Analysis",
                hover_data=['Count'],
                color='Count',
                color_continuous_scale='Plasma'
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(width=800, height=500)
            st.plotly_chart(fig, use_container_width=True)

    def _display_export_options(self, results: Dict[str, Any]):
        """Display export options for results"""
        signals = results['signals']
        data = results['data']

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export Signals as CSV", use_container_width=True):
                self._export_signals_csv(signals)

        with col2:
            if st.button("📈 Export Full Analysis as CSV", use_container_width=True):
                self._export_full_analysis_csv(signals, data)

        # JSON export
        if st.button("💾 Export as JSON", use_container_width=True):
            self._export_json_results(results)

        st.success("Results exported successfully!")

    def _export_signals_csv(self, signals: List[Dict[str, Any]]):
        """Export signals as CSV"""
        try:
            df = pd.DataFrame(signals)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Signals CSV",
                data=csv_data,
                file_name=f"pattern_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error exporting CSV: {e}")

    def _export_full_analysis_csv(self, signals: List[Dict[str, Any]], data: pd.DataFrame):
        """Export full analysis as CSV"""
        try:
            # Create analysis summary
            summary = {
                'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total Signals': len(signals),
                'Data Points': len(data),
                'Date Range': f"{data.index.min()} to {data.index.max()}"
            }

            # Add summary to signals
            export_data = []
            for signal in signals:
                export_data.append({
                    'Analysis Date': summary['Analysis Date'],
                    'Total Signals': summary['Total Signals'],
                    'Data Points': summary['Data Points'],
                    'Date Range': summary['Date Range'],
                    **signal
                })

            df = pd.DataFrame(export_data)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download Full Analysis CSV",
                data=csv_data,
                file_name=f"full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error exporting CSV: {e}")

    def _export_json_results(self, results: Dict[str, Any]):
        """Export results as JSON"""
        try:
            import json
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download Results JSON",
                data=json_data,
                file_name=f"pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Error exporting JSON: {e}")

    def _show_no_patterns_found(self):
        """Show message when no patterns are found"""
        st.info("🔍 No patterns detected in the specified date range.")

        # Possible reasons
        st.markdown("**Possible reasons:**")
        reasons = [
            "Market conditions may not favor the selected patterns",
            "Date range might be too short for pattern formation",
            "Confidence threshold might be too high",
            "Selected patterns may not be present in this market data"
        ]

        for i, reason in enumerate(reasons, 1):
            st.write(f"{i}. {reason}")

        # Recommendations
        st.markdown("**Recommendations:**")
        recommendations = [
            "Try adjusting the confidence threshold",
            "Extend the date range for analysis",
            "Select different pattern types",
            "Check if market data is available for the selected period"
        ]

        for i, recommendation in enumerate(recommendations, 1):
            st.write(f"{i}. {recommendation}")

        # Pattern formation information
        with st.expander("📚 About Pattern Formation", expanded=False):
            st.markdown("""
            ### Pattern Formation Requirements:

            **Double Bottom:** Typically forms over several weeks, requires clear support level testing
            **Head and Shoulders:** Forms over 3-6 months, requires distinct peaks and troughs
            **Cup and Handle:** Forms over several months, requires consolidation phase
            **Flag Patterns:** Forms over 1-4 weeks, requires preceding strong move

            Most patterns require sufficient price movement and trading volume to be detectable.
            """)