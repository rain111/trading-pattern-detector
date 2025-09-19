"""
Logging configuration for the trading pattern detection system.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    console_output: bool = True,
    file_output: bool = True,
) -> None:
    """
    Setup comprehensive logging configuration for the trading pattern detection system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default path)
        console_output: Whether to log to console
        file_output: Whether to log to file
    """

    # Determine log file path
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "trading_pattern_detector.log"

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure handlers
    handlers = {}

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(formatter)
        handlers["console"] = console_handler

    # File handler
    if file_output:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        handlers["file"] = file_handler

    # Configure loggers
    loggers_config = {
        "root": {
            "level": "DEBUG",
            "handlers": list(handlers.keys()),
            "propagate": False,
        },
        "trading_pattern_detector": {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": False,
            "qualname": "trading_pattern_detector",
        },
        "trading_pattern_detector.core": {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": False,
            "qualname": "trading_pattern_detector.core",
        },
        "trading_pattern_detector.detectors": {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": False,
            "qualname": "trading_pattern_detector.detectors",
        },
        "trading_pattern_detector.analysis": {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": False,
            "qualname": "trading_pattern_detector.analysis",
        },
        "trading_pattern_detector.utils": {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": False,
            "qualname": "trading_pattern_detector.utils",
        },
        "trading_pattern_detector.config": {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": False,
            "qualname": "trading_pattern_detector.config",
        },
        "trading_pattern_detector.plugins": {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": False,
            "qualname": "trading_pattern_detector.plugins",
        },
        "tests": {
            "level": "WARNING",
            "handlers": ["console"] if console_output else [],
            "propagate": False,
        },
    }

    # Configure logging
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": handlers,
        "loggers": loggers_config,
        "root": {"level": "DEBUG", "handlers": list(handlers.keys())},
    }

    try:
        logging.config.dictConfig(config)

        # Log successful setup
        logger = logging.getLogger("trading_pattern_detector")
        logger.info(f"Logging setup completed successfully - Level: {level}")
        logger.info(f"Log file: {log_file.absolute()}")

        if console_output:
            logger.info("Console logging enabled")
        if file_output:
            logger.info("File logging enabled")

    except Exception as e:
        # Fallback to basic logging if dictConfig fails
        logging.basicConfig(
            level=getattr(logging, level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers.values() if handlers else [logging.StreamHandler()],
        )
        logger = logging.getLogger("trading_pattern_detector")
        logger.error(f"Advanced logging setup failed, using basic configuration: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TradingPatternLogger:
    """
    Specialized logger for trading pattern detection operations.
    Provides structured logging for specific trading operations.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_pattern_detection(
        self,
        symbol: str,
        pattern_type: str,
        confidence: float,
        entry_price: float,
        timeframe: str,
    ) -> None:
        """Log pattern detection results."""
        self.logger.info(
            f"Pattern detected - Symbol: {symbol}, Type: {pattern_type}, "
            f"Confidence: {confidence:.2f}, Entry: ${entry_price:.2f}, "
            f"Timeframe: {timeframe}"
        )

    def log_signal_generation(
        self,
        symbol: str,
        action: str,
        price: float,
        stop_loss: float,
        target_price: float,
        risk_level: str,
    ) -> None:
        """Log signal generation."""
        self.logger.info(
            f"Signal generated - Symbol: {symbol}, Action: {action}, "
            f"Price: ${price:.2f}, Stop: ${stop_loss:.2f}, "
            f"Target: ${target_price:.2f}, Risk: {risk_level}"
        )

    def log_data_quality_issue(self, symbol: str, issue: str) -> None:
        """Log data quality issues."""
        self.logger.warning(f"Data quality issue - Symbol: {symbol}, Issue: {issue}")

    def log_detector_performance(
        self, detector_name: str, processing_time: float, signals_found: int
    ) -> None:
        """Log detector performance metrics."""
        self.logger.info(
            f"Detector performance - {detector_name}: "
            f"Time: {processing_time:.3f}s, Signals: {signals_found}"
        )

    def log_error(self, operation: str, error: Exception, context: str = "") -> None:
        """Log errors with context."""
        error_msg = f"Error in {operation}: {str(error)}"
        if context:
            error_msg += f" | Context: {context}"
        self.logger.error(error_msg, exc_info=True)

    def log_config_change(self, component: str, change: str) -> None:
        """Log configuration changes."""
        self.logger.info(f"Config changed - {component}: {change}")

    def log_market_data_update(
        self, symbol: str, data_points: int, timeframe: str
    ) -> None:
        """Log market data updates."""
        self.logger.debug(
            f"Market data updated - Symbol: {symbol}, "
            f"Data points: {data_points}, Timeframe: {timeframe}"
        )
