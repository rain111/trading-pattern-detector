"""
Error Handler - Centralized error handling and user feedback
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import streamlit as st
from dataclasses import dataclass, asdict
from enum import Enum

class ErrorType(Enum):
    """Error type categories"""
    VALIDATION = "validation"
    NETWORK = "network"
    DATA = "data"
    PATTERN_DETECTION = "pattern_detection"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"

@dataclass
class ErrorInfo:
    """Error information structure"""
    error_type: ErrorType
    message: str
    details: str
    timestamp: datetime
    context: Dict[str, Any]
    suggestions: List[str]
    severity: str = "medium"  # low, medium, high, critical

class ErrorHandler:
    """Centralized error handling for the application"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.errors: List[ErrorInfo] = []
        self.error_callbacks: List[Callable] = []
        self._setup_error_logging()

    def _setup_error_logging(self):
        """Setup error logging"""
        # Create logs directory if it doesn't exist
        from ..config import settings
        import os
        os.makedirs(settings.LOGS_DIR, exist_ok=True)

        # Setup file logging
        log_file = settings.LOGS_DIR / "errors.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def handle_error(
        self,
        error: Exception,
        error_type: ErrorType = ErrorType.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        severity: str = "medium"
    ) -> ErrorInfo:
        """Handle an error and provide user-friendly feedback"""
        try:
            # Create error info
            error_info = ErrorInfo(
                error_type=error_type,
                message=user_message or str(error),
                details=str(error),
                timestamp=datetime.now(),
                context=context or {},
                suggestions=suggestions or self._get_default_suggestions(error_type),
                severity=severity
            )

            # Log the error
            self._log_error(error_info)

            # Add to error list
            self.errors.append(error_info)

            # Call callbacks
            for callback in self.error_callbacks:
                try:
                    callback(error_info)
                except Exception as e:
                    self.logger.error(f"Error in error callback: {e}")

            return error_info

        except Exception as e:
            self.logger.error(f"Error in error handling: {e}")
            return ErrorInfo(
                error_type=ErrorType.UNKNOWN,
                message="Internal error occurred",
                details=str(e),
                timestamp=datetime.now(),
                context={},
                suggestions=["Please try again later"],
                severity="critical"
            )

    def _log_error(self, error_info: ErrorInfo):
        """Log error to file and console"""
        log_message = f"[{error_info.error_type.value}] {error_info.message}"

        if error_info.severity == "critical":
            self.logger.critical(log_message, exc_info=True)
        elif error_info.severity == "high":
            self.logger.error(log_message, exc_info=True)
        elif error_info.severity == "medium":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _get_default_suggestions(self, error_type: ErrorType) -> List[str]:
        """Get default suggestions for error types"""
        suggestions_map = {
            ErrorType.VALIDATION: [
                "Check your input values",
                "Ensure all required fields are filled",
                "Verify date format and range"
            ],
            ErrorType.NETWORK: [
                "Check your internet connection",
                "Try again in a few moments",
                "Verify Yahoo Finance API availability"
            ],
            ErrorType.DATA: [
                "Check if symbol exists and is valid",
                "Try a different date range",
                "Verify data availability for the selected period"
            ],
            ErrorType.PATTERN_DETECTION: [
                "Try adjusting confidence threshold",
                "Select different pattern types",
                "Extend the date range for analysis"
            ],
            ErrorType.CONFIGURATION: [
                "Check configuration settings",
                "Verify required dependencies are installed",
                "Restart the application"
            ],
            ErrorType.UNKNOWN: [
                "Try again later",
                "Check your input parameters",
                "Contact support if the problem persists"
            ]
        }

        return suggestions_map.get(error_type, [
            "Try again later",
            "Check your input parameters"
        ])

    def display_error(self, error_info: ErrorInfo, show_details: bool = False):
        """Display error information to user"""
        # Choose appropriate Streamlit component based on severity
        if error_info.severity == "critical":
            st.error(f"🚨 {error_info.message}")
        elif error_info.severity == "high":
            st.error(f"❌ {error_info.message}")
        elif error_info.severity == "medium":
            st.warning(f"⚠️ {error_info.message}")
        else:
            st.info(f"ℹ️ {error_info.message}")

        # Show suggestions
        if error_info.suggestions:
            st.write("**Suggestions:**")
            for i, suggestion in enumerate(error_info.suggestions, 1):
                st.write(f"{i}. {suggestion}")

        # Show details if requested
        if show_details:
            with st.expander("🔍 Error Details", expanded=False):
                st.write(f"**Error Type:** {error_info.error_type.value}")
                st.write(f"**Severity:** {error_info.severity}")
                st.write(f"**Timestamp:** {error_info.timestamp}")

                if error_info.context:
                    st.write("**Context:**")
                    for key, value in error_info.context.items():
                        st.write(f"- {key}: {value}")

                st.write(f"**Details:** {error_info.details}")

    def display_errors_summary(self):
        """Display summary of all errors"""
        if not self.errors:
            return

        st.subheader("📋 Error Summary")

        # Group errors by type
        error_types = {}
        for error in self.errors:
            error_type = error.error_type.value
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)

        # Display error counts
        col1, col2, col3 = st.columns(3)

        with col1:
            total_errors = len(self.errors)
            critical_errors = len([e for e in self.errors if e.severity == "critical"])
            st.metric("Total Errors", total_errors, f"{critical_errors} Critical")

        with col2:
            error_types_count = len(error_types)
            st.metric("Error Types", error_types_count)

        with col3:
            recent_errors = len([e for e in self.errors if
                               (datetime.now() - e.timestamp).total_seconds() < 3600])
            st.metric("Last Hour", recent_errors)

        # Display errors by type
        for error_type, errors in error_types.items():
            with st.expander(f"📄 {error_type.title()} ({len(errors)} errors)"):
                for i, error in enumerate(errors[-5:], 1):  # Show last 5 errors
                    self.display_error(error, show_details=False)

    def add_error_callback(self, callback: Callable):
        """Add a callback function to be called on errors"""
        self.error_callbacks.append(callback)

    def get_errors_by_type(self, error_type: ErrorType) -> List[ErrorInfo]:
        """Get all errors of a specific type"""
        return [error for error in self.errors if error.error_type == error_type]

    def get_errors_by_severity(self, severity: str) -> List[ErrorInfo]:
        """Get all errors of a specific severity"""
        return [error for error in self.errors if error.severity == severity]

    def clear_errors(self, error_type: Optional[ErrorType] = None):
        """Clear errors"""
        if error_type:
            self.errors = [error for error in self.errors if error.error_type != error_type]
        else:
            self.errors.clear()

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.errors:
            return {
                'total_errors': 0,
                'error_types': {},
                'severity_distribution': {},
                'errors_last_hour': 0
            }

        # Error type distribution
        error_types = {}
        for error in self.errors:
            error_type = error.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Severity distribution
        severity_distribution = {}
        for error in self.errors:
            severity = error.severity
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1

        # Errors in last hour
        errors_last_hour = len([error for error in self.errors if
                              (datetime.now() - error.timestamp).total_seconds() < 3600])

        return {
            'total_errors': len(self.errors),
            'error_types': error_types,
            'severity_distribution': severity_distribution,
            'errors_last_hour': errors_last_hour,
            'first_error': self.errors[0].timestamp if self.errors else None,
            'last_error': self.errors[-1].timestamp if self.errors else None
        }

    def clear_old_errors(self, max_age_hours: int = 24):
        """Clear errors older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        old_count = len(self.errors)
        self.errors = [error for error in self.errors if error.timestamp > cutoff_time]
        cleared_count = old_count - len(self.errors)

        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} old errors")

# Global error handler instance
error_handler = ErrorHandler()

# Convenience functions for common error scenarios
def handle_validation_error(message: str, context: Optional[Dict[str, Any]] = None):
    """Handle validation errors"""
    return error_handler.handle_error(
        Exception(message),
        ErrorType.VALIDATION,
        context,
        message,
        [
            "Check your input values",
            "Ensure all required fields are filled",
            "Verify data format and range"
        ],
        "medium"
    )

def handle_network_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Handle network-related errors"""
    return error_handler.handle_error(
        error,
        ErrorType.NETWORK,
        context,
        "Network error occurred while fetching data",
        [
            "Check your internet connection",
            "Try again in a few moments",
            "Verify Yahoo Finance API availability"
        ],
        "high"
    )

def handle_data_error(error: Exception, symbol: str, context: Optional[Dict[str, Any]] = None):
    """Handle data-related errors"""
    return error_handler.handle_error(
        error,
        ErrorType.DATA,
        {**context or {}, 'symbol': symbol},
        f"Data error occurred for symbol {symbol}",
        [
            "Check if symbol exists and is valid",
            "Try a different date range",
            "Verify data availability for the selected period"
        ],
        "medium"
    )

def handle_pattern_detection_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """Handle pattern detection errors"""
    return error_handler.handle_error(
        error,
        ErrorType.PATTERN_DETECTION,
        context,
        "Pattern detection failed",
        [
            "Try adjusting confidence threshold",
            "Select different pattern types",
            "Extend the date range for analysis"
        ],
        "medium"
    )