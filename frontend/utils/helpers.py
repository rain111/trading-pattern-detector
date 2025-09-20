"""
Utility Functions - Common helper functions for the frontend
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import streamlit as st
import hashlib
import json
from pathlib import Path

from ..config import settings

def format_currency(value: Union[int, float]) -> str:
    """Format currency value with appropriate formatting"""
    if value is None or pd.isna(value):
        return "N/A"

    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value: Union[int, float]) -> str:
    """Format percentage value"""
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.1%}"

def format_date(date: datetime) -> str:
    """Format date for display"""
    if date is None:
        return "N/A"

    return date.strftime("%Y-%m-%d")

def format_datetime(dt: datetime) -> str:
    """Format datetime for display"""
    if dt is None:
        return "N/A"

    return dt.strftime("%Y-%m-%d %H:%M")

def format_confidence(confidence: float) -> str:
    """Format confidence score"""
    if confidence is None or pd.isna(confidence):
        return "N/A"

    if confidence >= 0.8:
        return f"🟢 {confidence:.2f} (High)"
    elif confidence >= 0.5:
        return f"🟡 {confidence:.2f} (Medium)"
    else:
        return f"🔴 {confidence:.2f} (Low)"

def format_risk_level(risk_level: str) -> str:
    """Format risk level for display"""
    if risk_level is None:
        return "N/A"

    risk_map = {
        'low': '🟢 Low',
        'medium': '🟡 Medium',
        'high': '🔴 High'
    }

    return risk_map.get(risk_level.lower(), risk_level.lower())

def calculate_timeframe_dates(start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Calculate timeframe statistics"""
    if not start_date or not end_date:
        return {}

    duration = end_date - start_date
    days = duration.days
    weeks = days / 7
    months = days / 30.44  # Average days per month

    return {
        'days': days,
        'weeks': weeks,
        'months': months,
        'years': days / 365.25,
        'business_days': int(days * 0.7),  # Approximate business days
        'trading_days': max(0, int(days * 0.6))  # Approximate trading days
    }

def create_cache_key(*args, **kwargs) -> str:
    """Create a cache key from input parameters"""
    key_data = {
        'args': args,
        'kwargs': kwargs,
        'timestamp': datetime.now().isoformat()
    }

    # Create hash
    key_string = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()

    return key_hash

def validate_stock_symbol(symbol: str) -> List[str]:
    """Validate stock symbol format"""
    errors = []

    if not symbol:
        errors.append("Stock symbol is required")
    elif not symbol.isalnum():
        errors.append("Stock symbol must be alphanumeric")
    elif len(symbol) > 10:
        errors.append("Stock symbol must be 10 characters or less")

    return errors

def validate_date_range(start_date: datetime, end_date: datetime) -> List[str]:
    """Validate date range"""
    errors = []

    if not start_date or not end_date:
        errors.append("Both start and end dates are required")
    elif start_date >= end_date:
        errors.append("Start date must be before end date")

    # Check if date range is too long (max 10 years)
    if start_date and end_date:
        duration = (end_date - start_date).days
        if duration > 365 * 10:
            errors.append("Date range cannot exceed 10 years")
        elif duration < 30:
            errors.append("Date range must be at least 30 days")

    return errors

def validate_pattern_selection(patterns: List[str]) -> List[str]:
    """Validate pattern selection"""
    errors = []

    if not patterns:
        errors.append("At least one pattern must be selected")

    if len(patterns) > 10:
        errors.append("Maximum 10 patterns can be selected at once")

    # Check for invalid patterns
    valid_patterns = set(settings.SUPPORTED_PATTERNS)
    invalid_patterns = [p for p in patterns if p not in valid_patterns]
    if invalid_patterns:
        errors.append(f"Invalid patterns selected: {', '.join(invalid_patterns)}")

    return errors

def validate_confidence_threshold(threshold: float) -> List[str]:
    """Validate confidence threshold"""
    errors = []

    if not 0.0 <= threshold <= 1.0:
        errors.append("Confidence threshold must be between 0.0 and 1.0")

    return errors

def get_pattern_category(pattern: str) -> str:
    """Get the category of a pattern"""
    pattern_categories = {
        "🔄 Reversal Patterns": ["DOUBLE_BOTTOM", "HEAD_AND_SHOULDERS", "ROUNDING_BOTTOM"],
        "➡️ Continuation Patterns": ["FLAG_PATTERN", "CUP_HANDLE", "ASCENDING_TRIANGLE", "DESCENDING_TRIANGLE", "RISING_WEDGE", "FALLING_WEDGE"],
        "🚀 Breakout Patterns": ["VCP_BREAKOUT"]
    }

    for category, patterns in pattern_categories.items():
        if pattern in patterns:
            return category

    return "Unknown"

def get_pattern_icon(pattern: str) -> str:
    """Get icon for a pattern"""
    pattern_icons = {
        "DOUBLE_BOTTOM": "📈",
        "HEAD_AND_SHOULDERS": "📉",
        "ROUNDING_BOTTOM": "🔄",
        "FLAG_PATTERN": "🚩",
        "CUP_HANDLE": "☕",
        "ASCENDING_TRIANGLE": "📊",
        "DESCENDING_TRIANGLE": "📉",
        "RISING_WEDGE": "⬆️",
        "FALLING_WEDGE": "⬇️",
        "VCP_BREAKOUT": "🚀"
    }

    return pattern_icons.get(pattern, "📊")

def create_download_button(data: str, filename: str, mime_type: str, label: str):
    """Create a download button for data"""
    return st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )

def create_file_download_button(filename: str, label: str):
    """Create a download button for a file"""
    if not Path(filename).exists():
        return None

    with open(filename, 'r') as f:
        data = f.read()

    return create_download_button(
        data=data,
        filename=filename,
        mime="text/plain",
        label=label
    )

def create_progress_bar(progress: float, text: str = ""):
    """Create a progress bar with optional text"""
    progress_bar = st.progress(progress)
    if text:
        st.write(text)

    return progress_bar

def create_columns_with_ratios(ratios: List[int]):
    """Create columns with specified width ratios"""
    total = sum(ratios)
    return st.columns([r/total for r in ratios])

def create_expander(title: str, expanded: bool = False):
    """Create an expander with title"""
    return st.expander(title, expanded=expanded)

def create_metric_cards(metrics: Dict[str, Any]):
    """Create metric cards in a grid"""
    cols = st.columns(len(metrics))

    for i, (key, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, (int, float)):
                st.metric(key, f"{value:.2f}")
            else:
                st.metric(key, str(value))

def create_data_table(data: pd.DataFrame, max_rows: int = 100):
    """Create a data table with row limit"""
    if len(data) > max_rows:
        st.write(f"Showing first {max_rows} rows of {len(data)} total rows")
        return st.dataframe(data.head(max_rows))
    else:
        return st.dataframe(data)

def create_line_chart(data: pd.DataFrame, x_column: str, y_column: str, title: str = ""):
    """Create a line chart"""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_column],
        y=data[y_column],
        mode='lines',
        name=y_column
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        showlegend=True
    )

    return st.plotly_chart(fig, use_container_width=True)

def create_candlestick_chart(data: pd.DataFrame, title: str = ""):
    """Create a candlestick chart"""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True
    )

    return st.plotly_chart(fig, use_container_width=True)

def get_available_memory():
    """Get available system memory"""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None

def get_cpu_usage():
    """Get current CPU usage"""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return None

def log_system_info():
    """Log system information"""
    import platform
    import psutil

    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent
    }

    return system_info

def create_error_message(error: Exception, context: str = "") -> str:
    """Create a user-friendly error message"""
    if context:
        return f"❌ {context}: {str(error)}"
    else:
        return f"❌ Error: {str(error)}"

def create_success_message(message: str) -> str:
    """Create a success message"""
    return f"✅ {message}"

def create_warning_message(message: str) -> str:
    """Create a warning message"""
    return f"⚠️ {message}"

def create_info_message(message: str) -> str:
    """Create an info message"""
    return f"ℹ️ {message}"

def validate_dataframe(data: pd.DataFrame) -> List[str]:
    """Validate DataFrame structure and content"""
    errors = []

    if data.empty:
        errors.append("DataFrame is empty")
        return errors

    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    # Check for NaN values
    nan_counts = data.isnull().sum()
    for col, count in nan_counts.items():
        if count > 0:
            errors.append(f"Column '{col}' has {count} NaN values")

    # Check for infinite values
    for col in data.columns:
        if np.isinf(data[col]).any():
            errors.append(f"Column '{col}' contains infinite values")

    # Check OHLC relationships
    if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        invalid_high_low = (data['high'] < data['low']).sum()
        invalid_high_open = (data['high'] < data['open']).sum()
        invalid_high_close = (data['high'] < data['close']).sum()
        invalid_low_open = (data['low'] > data['open']).sum()
        invalid_low_close = (data['low'] > data['close']).sum()

        total_invalid = (invalid_high_low + invalid_high_open + invalid_high_close +
                        invalid_low_open + invalid_low_close)

        if total_invalid > 0:
            errors.append(f"Found {total_invalid} invalid OHLC relationships")

    return errors