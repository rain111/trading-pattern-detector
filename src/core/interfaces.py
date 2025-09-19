from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import logging
from datetime import datetime
import numpy as np


class PatternType(Enum):
    VCP_BREAKOUT = "vcp_breakout"
    FLAG_PATTERN = "flag_pattern"
    CUP_HANDLE = "cup_handle"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DOUBLE_BOTTOM = "double_bottom"
    WEDGE_PATTERN = "wedge_pattern"
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    ROUNDING_BOTTOM = "rounding_bottom"
    DESCENDING_TRIANGLE = "descending_triangle"
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"


@dataclass
class PatternConfig:
    """Base configuration for pattern detection"""

    min_confidence: float = 0.6
    max_lookback: int = 100
    timeframe: str = "1d"
    volume_threshold: float = 1000000.0
    volatility_threshold: float = 0.001
    reward_ratio: float = 2.0


@dataclass
class PatternSignal:
    """Enhanced trading signal with comprehensive metadata"""

    symbol: str
    pattern_type: PatternType
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    timeframe: str
    timestamp: pd.Timestamp
    metadata: Dict[str, Any]
    signal_strength: float = 0.0
    risk_level: str = "medium"
    expected_duration: Optional[str] = None
    probability_target: Optional[float] = None


class EnhancedPatternDetector(ABC):
    """Enhanced base class with common functionality"""

    def __init__(self, config: PatternConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_components()

    def _setup_components(self):
        """Setup analysis components"""
        try:
            from ..analysis.volatility_analyzer import VolatilityAnalyzer
            from ..analysis.volume_analyzer import VolumeAnalyzer
            from ..analysis.trend_analyzer import TrendAnalyzer

            self.volatility_analyzer = VolatilityAnalyzer()
            self.volume_analyzer = VolumeAnalyzer()
            self.trend_analyzer = TrendAnalyzer()
        except ImportError:
            # Fallback for testing
            self.volatility_analyzer = None
            self.volume_analyzer = None
            self.trend_analyzer = None

    @abstractmethod
    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Main pattern detection method"""
        pass

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return required columns for the detector"""
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Enhanced data validation with comprehensive error handling"""
        try:
            # Check for None or invalid data
            if data is None:
                self.logger.error("Data is None")
                return False

            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                try:
                    data = pd.DataFrame(data)
                except Exception as e:
                    self.logger.error(f"Cannot convert data to DataFrame: {e}")
                    return False

            # Check for empty data
            if data.empty:
                self.logger.error("Data is empty")
                return False

            # Check required columns
            required_columns = self.get_required_columns()
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]

            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False

            # Check data quality
            for col in ["open", "high", "low", "close", "volume"]:
                if col in data.columns:
                    if data[col].isna().any():
                        self.logger.error(f"Column '{col}' contains NaN values")
                        return False

                    # Check for invalid price values
                    if col in ["open", "high", "low", "close"]:
                        if (data[col] <= 0).any():
                            self.logger.error(
                                f"Column '{col}' contains non-positive values"
                            )
                            return False

            # Check price consistency
            if "high" in data.columns and "low" in data.columns:
                if (data["high"] < data["low"]).any():
                    self.logger.error("High prices cannot be lower than low prices")
                    return False

                if "open" in data.columns and "close" in data.columns:
                    # Check if open/close are within high/low bounds
                    invalid_open = (data["open"] > data["high"]) | (
                        data["open"] < data["low"]
                    )
                    invalid_close = (data["close"] > data["high"]) | (
                        data["close"] < data["low"]
                    )

                    if invalid_open.any():
                        self.logger.error("Open prices outside high/low bounds")
                        return False

                    if invalid_close.any():
                        self.logger.error("Close prices outside high/low bounds")
                        return False

            # Check minimum data length
            if len(data) < self.config.max_lookback:
                self.logger.warning(
                    f"Insufficient data: {len(data)} rows, "
                    f"minimum required: {self.config.max_lookback}"
                )
                return False

            # Check for data timeliness (if timestamp index exists)
            if hasattr(data, "index") and isinstance(data.index, pd.DatetimeIndex):
                latest_date = data.index.max()
                # Ensure both timestamps have the same timezone handling
                current_date = pd.Timestamp.now(tz=latest_date.tz)
                time_diff = current_date - latest_date

                if time_diff.days > 7:  # More than a week old
                    self.logger.warning(f"Data may be stale: latest date {latest_date}")

            self.logger.debug(
                f"Data validation passed: {len(data)} rows, {len(data.columns)} columns"
            )
            return True

        except Exception as e:
            self.logger.error(
                f"Unexpected error during data validation: {e}", exc_info=True
            )
            return False

    def calculate_confidence(self, pattern_data: dict) -> float:
        """Calculate pattern confidence score"""
        confidence = 0.5  # Base confidence

        # Volume confidence
        if "volume_ratio" in pattern_data:
            confidence += min(pattern_data["volume_ratio"] * 0.1, 0.2)

        # Volatility confidence
        if "volatility_score" in pattern_data:
            confidence += min(abs(pattern_data["volatility_score"]) * 0.2, 0.2)

        # Trend confidence
        if "trend_strength" in pattern_data:
            confidence += min(pattern_data["trend_strength"] * 0.1, 0.1)

        return min(confidence, 1.0)

    def generate_signals(self, patterns: List[dict]) -> List[PatternSignal]:
        """Convert pattern detections to trading signals"""
        signals = []

        for pattern in patterns:
            try:
                confidence = self.calculate_confidence(pattern)

                if confidence >= self.config.min_confidence:
                    signal = PatternSignal(
                        symbol=pattern["symbol"],
                        pattern_type=pattern["pattern_type"],
                        confidence=confidence,
                        entry_price=pattern["entry_price"],
                        stop_loss=pattern["stop_loss"],
                        target_price=pattern["target_price"],
                        timeframe=self.config.timeframe,
                        timestamp=pattern["timestamp"],
                        metadata=pattern.get("metadata", {}),
                        signal_strength=self._calculate_signal_strength(pattern),
                        risk_level=self._determine_risk_level(pattern),
                    )
                    signals.append(signal)

            except Exception as e:
                self.logger.error(f"Error generating signal: {e}")
                continue

        return signals

    def _calculate_signal_strength(self, pattern: dict) -> float:
        """Calculate overall signal strength"""
        strength = pattern.get("confidence", 0.5)
        volume_multiplier = pattern.get("volume_ratio", 1.0)
        strength *= volume_multiplier

        return min(strength, 1.0)

    def _determine_risk_level(self, pattern: dict) -> str:
        """Determine risk level based on pattern characteristics"""
        volatility = pattern.get("volatility_score", 0)

        if volatility > 0.05:
            return "high"
        elif volatility > 0.02:
            return "medium"
        else:
            return "low"


class BaseDetector(EnhancedPatternDetector):
    """Base detector with common functionality"""

    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Main pattern detection method"""
        pass

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for pattern detection"""
        # Remove NaN values
        data = data.dropna()

        # Sort by timestamp
        data = data.sort_index()

        # Calculate additional indicators if needed
        if "returns" not in data.columns:
            data["returns"] = data["close"].pct_change()

        return data

    def validate_signals(self, signals: List[PatternSignal]) -> List[PatternSignal]:
        """Validate and filter signals"""
        valid_signals = []

        for signal in signals:
            # Check confidence threshold
            if signal.confidence < self.config.min_confidence:
                continue

            # Check price validity
            if (
                signal.entry_price <= 0
                or signal.stop_loss <= 0
                or signal.target_price <= 0
            ):
                continue

            # Check logical relationships
            if (
                signal.stop_loss == signal.entry_price
                or signal.target_price == signal.entry_price
            ):
                continue

            valid_signals.append(signal)

        return valid_signals


class DataValidator:
    """Validates input data for pattern detection"""

    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> None:
        """Validate price data contains required columns and proper format"""
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for NaN values
        for col in required_columns:
            if data[col].isna().any():
                raise ValueError(f"Column '{col}' contains NaN values")

        # Check price logic consistency
        if (data["high"] < data["low"]).any():
            raise ValueError("High prices cannot be lower than low prices")

        # Allow for minor deviations in real market data (like stock splits, dividends)
        tolerance = 0.001  # 0.1% tolerance

        # Check if high is significantly less than open or close
        high_violations = (data["high"] < data["open"] * (1 - tolerance)) | (data["high"] < data["close"] * (1 - tolerance))
        if high_violations.any():
            high_violation_count = high_violations.sum()
            raise ValueError(f"Found {high_violation_count} rows where high prices are significantly less than open/close prices")

        # Check if low is significantly greater than open or close
        low_violations = (data["low"] > data["open"] * (1 + tolerance)) | (data["low"] > data["close"] * (1 + tolerance))
        if low_violations.any():
            low_violation_count = low_violations.sum()
            raise ValueError(f"Found {low_violation_count} rows where low prices are significantly greater than open/close prices")

    @staticmethod
    def clean_ohlc_data(data: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLC data by fixing obvious inconsistencies"""
        cleaned_data = data.copy()

        for idx in cleaned_data.index:
            open_price = cleaned_data.loc[idx, 'open']
            close_price = cleaned_data.loc[idx, 'close']
            current_high = cleaned_data.loc[idx, 'high']
            current_low = cleaned_data.loc[idx, 'low']

            # Ensure high is at least the maximum of open and close
            expected_high = max(open_price, close_price)
            if current_high < expected_high:
                cleaned_data.loc[idx, 'high'] = expected_high
                print(f"Fixed high price for {idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else idx}: {current_high:.6f} -> {expected_high:.6f}")

            # Ensure low is at most the minimum of open and close
            expected_low = min(open_price, close_price)
            if current_low > expected_low:
                cleaned_data.loc[idx, 'low'] = expected_low
                print(f"Fixed low price for {idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else idx}: {current_low:.6f} -> {expected_low:.6f}")

        return cleaned_data

    @staticmethod
    def validate_price_data_safe(data: pd.DataFrame) -> bool:
        """Safe validation that returns boolean instead of raising exceptions"""
        try:
            DataValidator.validate_price_data(data)
            return True
        except ValueError:
            return False


class PatternEngine:
    """Main engine for pattern detection coordination"""

    def __init__(
        self,
        detectors: List[EnhancedPatternDetector],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.detectors = detectors
        self.config = config or {}
        self.validator = DataValidator()
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_patterns(self, data: pd.DataFrame, symbol: str) -> List[PatternSignal]:
        """Run all detectors on the given data with comprehensive error handling"""
        try:
            # Validate input data
            if not DataValidator.validate_price_data_safe(data):
                self.logger.error("Data validation failed, aborting pattern detection")
                return []

            if not self.detectors:
                self.logger.warning("No detectors configured for pattern detection")
                return []

            self.logger.info(
                f"Starting pattern detection for {symbol} with {len(self.detectors)} detectors"
            )
            all_signals = []
            successful_detectors = 0

            # Run each detector with error handling
            for i, detector in enumerate(self.detectors):
                try:
                    detector_name = detector.__class__.__name__
                    self.logger.debug(
                        f"Running detector {i+1}/{len(self.detectors)}: {detector_name}"
                    )

                    # Validate data for this detector
                    if not detector.validate_data(data):
                        self.logger.warning(
                            f"Data validation failed for detector {detector_name}"
                        )
                        continue

                    # Run pattern detection
                    signals = detector.detect_pattern(data)
                    successful_detectors += 1

                    # Validate and update signals
                    if signals:
                        valid_signals = []
                        for signal in signals:
                            try:
                                # Ensure signal has the required symbol
                                if (
                                    not hasattr(signal, "symbol")
                                    or signal.symbol is None
                                ):
                                    signal.symbol = symbol
                                elif signal.symbol != symbol:
                                    signal.symbol = symbol

                                # Validate signal data
                                if (
                                    signal.entry_price > 0
                                    and signal.stop_loss > 0
                                    and signal.target_price > 0
                                    and signal.confidence > 0
                                ):
                                    valid_signals.append(signal)
                                else:
                                    self.logger.warning(
                                        f"Invalid signal data for {symbol}"
                                    )
                            except Exception as e:
                                self.logger.warning(f"Signal validation error: {e}")
                                continue

                        # Update symbol and add to results
                        for signal in valid_signals:
                            if not hasattr(signal, "symbol") or signal.symbol is None:
                                signal.symbol = symbol

                        all_signals.extend(valid_signals)
                        self.logger.debug(
                            f"Detector {detector_name} found {len(valid_signals)} signals"
                        )
                    else:
                        self.logger.debug(f"Detector {detector_name} found no signals")

                except Exception as e:
                    self.logger.error(
                        f"Error in detector {detector.__class__.__name__}: {e}",
                        exc_info=True,
                    )
                    continue

            # Filter signals by confidence threshold
            min_confidence = self.config.get("min_confidence", 0.6)
            filtered_signals = [
                signal for signal in all_signals if signal.confidence >= min_confidence
            ]

            self.logger.info(
                f"Pattern detection completed for {symbol}: "
                f"{successful_detectors}/{len(self.detectors)} detectors succeeded, "
                f"found {len(filtered_signals)} valid signals"
            )

            return filtered_signals

        except Exception as e:
            self.logger.error(
                f"Critical error during pattern detection for {symbol}: {e}",
                exc_info=True,
            )
            return []

    def add_detector(self, detector: EnhancedPatternDetector) -> None:
        """Add a new detector to the engine"""
        self.detectors.append(detector)

    def remove_detector(self, detector_type: type) -> None:
        """Remove a detector by type"""
        self.detectors = [d for d in self.detectors if not isinstance(d, detector_type)]
