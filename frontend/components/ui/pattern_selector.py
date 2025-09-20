"""
Pattern Selector Component - Handles pattern selection interface
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import pandas as pd

from ...config import settings

class PatternSelector:
    """Handles pattern selection and configuration"""

    def __init__(self):
        self.logger = st.logger.get_logger(__name__)

    def render_pattern_selection(self) -> List[str]:
        """Render pattern selection interface"""
        st.subheader("🔍 Pattern Selection")

        # Pattern categories with icons
        pattern_categories = {
            "🔄 Reversal Patterns": {
                "patterns": [
                    "DOUBLE_BOTTOM",
                    "HEAD_AND_SHOULDERS",
                    "ROUNDING_BOTTOM"
                ],
                "description": "Patterns that indicate potential trend reversals",
                "icon": "🔄"
            },
            "➡️ Continuation Patterns": {
                "patterns": [
                    "FLAG_PATTERN",
                    "CUP_HANDLE",
                    "ASCENDING_TRIANGLE",
                    "DESCENDING_TRIANGLE",
                    "RISING_WEDGE",
                    "FALLING_WEDGE"
                ],
                "description": "Patterns that suggest trend continuation",
                "icon": "➡️"
            },
            "🚀 Breakout Patterns": {
                "patterns": [
                    "VCP_BREAKOUT"
                ],
                "description": "Patterns that indicate potential breakouts",
                "icon": "🚀"
            }
        }

        selected_patterns = []

        # Render each category
        for category_name, category_info in pattern_categories.items():
            with st.expander(f"{category_name} ({len(category_info['patterns'])} patterns)", expanded=True):
                st.markdown(f"**{category_info['description']}**")

                # Create pattern selection grid
                patterns = category_info["patterns"]
                cols = 2  # Number of columns for pattern selection

                for i in range(0, len(patterns), cols):
                    cols_container = st.columns(cols)

                    for j in range(cols):
                        if i + j < len(patterns):
                            pattern = patterns[i + j]
                            pattern_info = self._get_pattern_info(pattern)

                            with cols_container[j]:
                                # Pattern card
                                card_style = """
                                <div style="
                                    background: #f0f2f6;
                                    padding: 15px;
                                    border-radius: 10px;
                                    border: 2px solid transparent;
                                    transition: all 0.3s ease;
                                    cursor: pointer;
                                " onmouseover="this.style.borderColor='#007bff'; this.style.transform='translateY(-2px)';"
                                onmouseout="this.style.borderColor='transparent'; this.style.transform='translateY(0)';">
                                    <h4 style="margin: 0; color: #333; font-size: 16px;">{name}</h4>
                                    <p style="margin: 5px 0 0 0; color: #666; font-size: 12px;">{type}</p>
                                    <p style="margin: 5px 0 0 0; color: #28a745; font-size: 12px;">{trend}</p>
                                </div>
                                """.format(
                                    name=pattern_info['name'],
                                    type=pattern_info['type'],
                                    trend=pattern_info['trend']
                                )

                                # Pattern image placeholder
                                pattern_image = pattern_info.get('image', '📊')

                                # Selection container
                                selected_key = f"pattern_{pattern}"
                                is_selected = st.checkbox(
                                    pattern_image,
                                    key=selected_key,
                                    help=pattern_info['description']
                                )

                                if is_selected:
                                    selected_patterns.append(pattern)

                                # Display card
                                st.markdown(card_style, unsafe_allow_html=True)

                                # Add pattern details on hover
                                with st.expander(f"Details for {pattern}", expanded=False):
                                    st.markdown(f"**Name:** {pattern_info['name']}")
                                    st.markdown(f"**Type:** {pattern_info['type']}")
                                    st.markdown(f"**Trend:** {pattern_info['trend']}")
                                    st.markdown(f"**Description:** {pattern_info['description']}")
                                    st.markdown(f" **Formation:** {pattern_info['formation']}")
                                    st.markdown(f" **Entry:** {pattern_info['entry']}")
                                    st.markdown(f" **Stop Loss:** {pattern_info['stop_loss']}")
                                    st.markdown(f" **Target:** {pattern_info['target']}")

        # Bulk selection options
        st.subheader("⚡ Quick Selection")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Select All Reversal", use_container_width=True):
                selected_patterns.extend([p for p in pattern_categories["🔄 Reversal Patterns"]["patterns"] if p not in selected_patterns])

        with col2:
            if st.button("➡️ Select All Continuation", use_container_width=True):
                selected_patterns.extend([p for p in pattern_categories["➡️ Continuation Patterns"]["patterns"] if p not in selected_patterns])

        with col3:
            if st.button("🚀 Select All Breakout", use_container_width=True):
                selected_patterns.extend([p for p in pattern_categories["🚀 Breakout Patterns"]["patterns"] if p not in selected_patterns])

        # Clear selection option
        if selected_patterns:
            if st.button("🗑️ Clear All Selections", use_container_width=True):
                selected_patterns.clear()

        return selected_patterns

    def _get_pattern_info(self, pattern_type: str) -> Dict[str, Any]:
        """Get detailed information about a pattern type"""
        pattern_database = {
            "DOUBLE_BOTTOM": {
                "name": "Double Bottom",
                "type": "Reversal",
                "trend": "Bullish",
                "description": "A bullish reversal pattern formed by two consecutive lows at approximately the same level, followed by a breakout above the resistance level.",
                "formation": "Forms over several weeks, requires clear support level testing and increased volume on breakout",
                "entry": "Breakout above resistance level",
                "stop_loss": "Below the low point of the double bottom",
                "target": "Height of the pattern from the breakout point",
                "image": "📈"
            },
            "HEAD_AND_SHOULDERS": {
                "name": "Head and Shoulders",
                "type": "Reversal",
                "trend": "Bearish",
                "description": "A bearish reversal pattern consisting of three peaks (left shoulder, head, right shoulder) with a neckline that connects the two troughs.",
                "formation": "Forms over 3-6 months, requires distinct peaks and troughs, volume should decrease on right shoulder",
                "entry": "Breakout below neckline",
                "stop_loss": "Above the head peak",
                "target": "Height of the head from the neckline",
                "image": "📉"
            },
            "ROUNDING_BOTTOM": {
                "name": "Rounding Bottom",
                "type": "Reversal",
                "trend": "Bullish",
                "description": "A long-term bullish reversal pattern characterized by a gradual decline to a low, followed by a gradual increase back up.",
                "formation": "Forms over several months, requires gradual price movement and increasing volume",
                "entry": "Breakout above the rounding bottom",
                "stop_loss": "Below the lowest point of the bottom",
                "target": "Height of the pattern from breakout point",
                "image": "🔄"
            },
            "FLAG_PATTERN": {
                "name": "Flag Pattern",
                "type": "Continuation",
                "trend": "Bullish/Bearish",
                "description": "A short consolidation pattern that occurs after a strong price move, followed by continuation of the original trend.",
                "formation": "Forms over 1-4 weeks, requires preceding strong move and decreasing volume during consolidation",
                "entry": "Breakout from flag pattern",
                "stop_loss": "Opposite end of the flag",
                "target": "Length of the flagpole from breakout point",
                "image": "🚩"
            },
            "CUP_HANDLE": {
                "name": "Cup and Handle",
                "type": "Continuation",
                "trend": "Bullish",
                "description": "A bullish continuation pattern consisting of a cup-shaped consolidation followed by a handle-shaped consolidation before breakout.",
                "formation": "Forms over several months, requires gradual cup formation and tight handle consolidation",
                "entry": "Breakout above the handle",
                "stop_loss": "Below the cup bottom",
                "target": "Height of the cup from breakout point",
                "image": "☕"
            },
            "ASCENDING_TRIANGLE": {
                "name": "Ascending Triangle",
                "type": "Continuation",
                "trend": "Bullish",
                "description": "A bullish continuation pattern characterized by horizontal resistance and ascending support trendlines.",
                "formation": "Forms over several weeks to months, requires test of resistance level and higher lows",
                "entry": "Breakout above resistance",
                "stop_loss": "Below the ascending trendline",
                "target": "Height of the triangle from breakout point",
                "image": "📊"
            },
            "DESCENDING_TRIANGLE": {
                "name": "Descending Triangle",
                "type": "Continuation",
                "trend": "Bearish",
                "description": "A bearish continuation pattern characterized by descending resistance and horizontal support trendlines.",
                "formation": "Forms over several weeks to months, requires test of support level and lower highs",
                "entry": "Breakout below support",
                "stop_loss": "Above the descending trendline",
                "target": "Height of the triangle from breakout point",
                "image": "📉"
            },
            "RISING_WEDGE": {
                "name": "Rising Wedge",
                "type": "Continuation",
                "trend": "Bearish",
                "description": "A bearish continuation pattern characterized by converging trendlines with ascending highs and lows.",
                "formation": "Forms over 1-3 months, requires narrowing price range and decreasing volume",
                "entry": "Breakout below the lower trendline",
                "stop_loss": "Above the highest point of the wedge",
                "target": "Height of the wedge from breakout point",
                "image": "⬆️"
            },
            "FALLING_WEDGE": {
                "name": "Falling Wedge",
                "type": "Continuation",
                "trend": "Bullish",
                "description": "A bullish continuation pattern characterized by converging trendlines with descending highs and lows.",
                "formation": "Forms over 1-3 months, requires narrowing price range and decreasing volume",
                "entry": "Breakout above the upper trendline",
                "stop_loss": "Below the lowest point of the wedge",
                "target": "Height of the wedge from breakout point",
                "image": "⬇️"
            },
            "VCP_BREAKOUT": {
                "name": "VCP Breakout",
                "type": "Breakout",
                "trend": "Bullish",
                "description": "A bullish breakout pattern from a volatility contraction pattern, indicating accumulation and potential upward momentum.",
                "formation": "Forms over several months, requires decreasing volatility and base building",
                "entry": "Breakout above resistance of VCP",
                "stop_loss": "Below the VCP base",
                "target": "Height of the base from breakout point",
                "image": "🚀"
            }
        }

        return pattern_database.get(pattern_type, {
            "name": pattern_type,
            "type": "Unknown",
            "trend": "Unknown",
            "description": "Pattern information not available",
            "formation": "Formation details not available",
            "entry": "Entry strategy not available",
            "stop_loss": "Stop loss not available",
            "target": "Target not available",
            "image": "📊"
        })

    def display_pattern_statistics(self, selected_patterns: List[str], all_patterns: List[str]):
        """Display pattern selection statistics"""
        if not selected_patterns:
            return

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Selected Patterns",
                len(selected_patterns),
                f"{len(selected_patterns)}/{len(all_patterns)} total"
            )

        with col2:
            st.metric(
                "Coverage",
                f"{len(selected_patterns)/len(all_patterns)*100:.1f}%",
                f"{len(selected_patterns)}/{len(all_patterns)} patterns"
            )

        # Pattern distribution by category
        category_counts = {}
        for pattern in selected_patterns:
            for category_name, category_info in {
                "🔄 Reversal Patterns": ["DOUBLE_BOTTOM", "HEAD_AND_SHOULDERS", "ROUNDING_BOTTOM"],
                "➡️ Continuation Patterns": ["FLAG_PATTERN", "CUP_HANDLE", "ASCENDING_TRIANGLE", "DESCENDING_TRIANGLE", "RISING_WEDGE", "FALLING_WEDGE"],
                "🚀 Breakout Patterns": ["VCP_BREAKOUT"]
            }.items():
                if pattern in category_info:
                    category_counts[category_name] = category_counts.get(category_name, 0) + 1
                    break

        if category_counts:
            st.subheader("📊 Pattern Distribution")
            for category, count in category_counts.items():
                st.write(f"{category}: {count} patterns")

    def validate_pattern_selection(self, selected_patterns: List[str]) -> List[str]:
        """Validate pattern selection and return any errors"""
        errors = []

        if not selected_patterns:
            errors.append("At least one pattern must be selected")

        if len(selected_patterns) > 10:
            errors.append("Maximum 10 patterns can be selected at once")

        # Check for invalid patterns
        valid_patterns = set(settings.SUPPORTED_PATTERNS)
        invalid_patterns = [p for p in selected_patterns if p not in valid_patterns]
        if invalid_patterns:
            errors.append(f"Invalid patterns selected: {', '.join(invalid_patterns)}")

        return errors

    def get_pattern_recommendations(self, market_conditions: str = "normal") -> List[str]:
        """Get pattern recommendations based on market conditions"""
        recommendations = []

        if market_conditions == "trending":
            recommendations.extend(["FLAG_PATTERN", "CUP_HANDLE", "ASCENDING_TRIANGLE"])
        elif market_conditions == "ranging":
            recommendations.extend(["DOUBLE_BOTTOM", "HEAD_AND_SHOULDERS", "ROUNDING_BOTTOM"])
        elif market_conditions == "volatile":
            recommendations.extend(["VCP_BREAKOUT", "RISING_WEDGE", "FALLING_WEDGE"])
        else:  # normal conditions
            recommendations.extend([
                "DOUBLE_BOTTOM",
                "FLAG_PATTERN",
                "CUP_HANDLE",
                "ASCENDING_TRIANGLE",
                "VCP_BREAKOUT"
            ])

        return recommendations

    def create_pattern_help_section(self):
        """Create comprehensive pattern help section"""
        with st.expander("📚 Pattern Knowledge Base", expanded=False):
            st.markdown("""
            ### Pattern Detection Guide

            **Understanding Technical Patterns:**

            Technical analysis patterns are formations that appear on price charts and are used to predict future price movements. They can be categorized into three main types:

            **1. Reversal Patterns**
            - Indicate potential trend changes
            - High reliability when confirmed
            - Require confirmation through volume and price action

            **2. Continuation Patterns**
            - Suggest the trend will continue
   - Occur during market consolidation
   - Often result in continuation moves

            **3. Breakout Patterns**
            - Indicate potential breakouts from consolidation
            - Often associated with volume increases
            - Can lead to significant price movements

            **Pattern Detection Tips:**
            - Always use proper risk management
            - Confirm patterns with volume analysis
            - Consider multiple timeframes
            - Use stop-loss orders for risk control
            - Monitor market context for better accuracy

            **Risk Management:**
            - Always use stop-loss orders
            - Position sizing based on risk tolerance
            - Consider multiple confirmation signals
            - Be aware of false breakouts
            """)

        return recommendations if 'recommendations' in locals() else []