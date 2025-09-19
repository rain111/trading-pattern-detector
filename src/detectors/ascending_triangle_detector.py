import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class AscendingTriangleDetector(BaseDetector):
    """Detector for Ascending Triangle patterns"""

    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ascending Triangle parameters
        self.min_pattern_length = 40  # Minimum days for pattern
        self.max_trendline_deviation = 0.015  # Maximum deviation from trendline
        self.max_horizontal_deviation = 0.008  # Maximum deviation from horizontal line
        self.min_volume_spike = 1.2  # Minimum volume spike for breakout
        self.trendline_points = 3  # Minimum points to define trendline
        self.horizontal_points = 3  # Minimum points to define horizontal line
        self.breakout_threshold = 0.01  # Price movement threshold for breakout

    def get_required_columns(self) -> List[str]:
        """Get required columns for pattern detection"""
        return ["open", "high", "low", "close", "volume"]

    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect Ascending Triangle patterns in the data"""
        signals = []

        if len(data) < self.min_pattern_length:
            return signals

        try:
            # Look for potential ascending triangle patterns
            for i in range(self.min_pattern_length, len(data)):
                pattern_signals = self._analyze_ascending_triangle(data, i)
                signals.extend(pattern_signals)

        except Exception as e:
            self.logger.error(f"Error in ascending triangle detection: {e}")

        return signals

    def _analyze_ascending_triangle(
        self, data: pd.DataFrame, end_idx: int
    ) -> List[PatternSignal]:
        """Analyze potential ascending triangle pattern ending at end_idx"""
        signals = []

        try:
            # Define search window
            start_idx = max(0, end_idx - self.min_pattern_length - 20)

            # Check for ascending triangle formation
            pattern_info = self._identify_ascending_triangle(data, start_idx, end_idx)

            if pattern_info:
                signal = self._create_ascending_triangle_signal(
                    pattern_info, data, end_idx
                )
                if signal:
                    signals.append(signal)

        except Exception as e:
            self.logger.debug(f"Error analyzing ascending triangle: {e}")

        return signals

    def _identify_ascending_triangle(
        self, data: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Identify ascending triangle pattern"""
        try:
            window_data = data.iloc[start_idx:end_idx]

            # Find potential horizontal resistance line
            horizontal_resistance = self._find_horizontal_resistance(window_data)
            if not horizontal_resistance:
                return None

            # Find ascending trendline
            ascending_trendline = self._find_ascending_trendline(
                window_data, horizontal_resistance["price"]
            )
            if not ascending_trendline:
                return None

            # Check if both lines intersect properly
            if not self._check_triangle_formation(
                horizontal_resistance, ascending_trendline, window_data
            ):
                return None

            # Check for breakout above horizontal resistance
            current_price = data.iloc[end_idx]["close"]
            if current_price > horizontal_resistance["price"] * (
                1 + self.breakout_threshold
            ):
                return {
                    "horizontal_resistance": horizontal_resistance,
                    "ascending_trendline": ascending_trendline,
                    "pattern_start_idx": start_idx,
                    "pattern_end_idx": end_idx,
                    "resistance_price": horizontal_resistance["price"],
                    "trendline_slope": ascending_trendline["slope"],
                    "trendline_intercept": ascending_trendline["intercept"],
                    "pattern_length": end_idx - start_idx,
                }

            return None

        except Exception as e:
            self.logger.debug(f"Error identifying ascending triangle: {e}")
            return None

    def _find_horizontal_resistance(
        self, data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Find horizontal resistance line"""
        try:
            # Find local maxima
            peaks = []
            for i in range(1, len(data) - 1):
                if (
                    data.iloc[i]["high"] >= data.iloc[i - 1]["high"]
                    and data.iloc[i]["high"] >= data.iloc[i + 1]["high"]
                    and data.iloc[i]["high"] > data.iloc[i - 1]["close"]
                    and data.iloc[i]["high"] > data.iloc[i + 1]["close"]
                ):
                    peaks.append(
                        {
                            "idx": i,
                            "price": data.iloc[i]["high"],
                            "timestamp": data.index[i],
                        }
                    )

            if len(peaks) < self.horizontal_points:
                return None

            # Check for horizontal resistance (peaks at similar levels)
            prices = [p["price"] for p in peaks]
            mean_price = np.mean(prices)
            std_price = np.std(prices)

            # Select peaks that are close to mean price
            resistance_peaks = [
                p
                for p in peaks
                if abs(p["price"] - mean_price) / mean_price
                <= self.max_horizontal_deviation
            ]

            if len(resistance_peaks) < self.horizontal_points:
                return None

            # Use the most recent peak as resistance level
            resistance_price = resistance_peaks[-1]["price"]

            return {
                "price": resistance_price,
                "peaks": resistance_peaks,
                "mean_price": mean_price,
                "std_price": std_price,
            }

        except Exception as e:
            self.logger.debug(f"Error finding horizontal resistance: {e}")
            return None

    def _find_ascending_trendline(
        self, data: pd.DataFrame, resistance_price: float
    ) -> Optional[Dict[str, Any]]:
        """Find ascending trendline"""
        try:
            # Find local minima for trendline
            valleys = []
            for i in range(1, len(data) - 1):
                if (
                    data.iloc[i]["low"] <= data.iloc[i - 1]["low"]
                    and data.iloc[i]["low"] <= data.iloc[i + 1]["low"]
                    and data.iloc[i]["low"] < data.iloc[i - 1]["close"]
                    and data.iloc[i]["low"] < data.iloc[i + 1]["close"]
                ):
                    valleys.append(
                        {
                            "idx": i,
                            "price": data.iloc[i]["low"],
                            "timestamp": data.index[i],
                        }
                    )

            if len(valleys) < self.trendline_points:
                return None

            # Sort valleys by index
            valleys_sorted = sorted(valleys, key=lambda x: x["idx"])

            # Try different combinations of valley points
            best_fit = None
            best_score = float("inf")

            for i in range(len(valleys_sorted) - 1):
                for j in range(i + 1, len(valleys_sorted)):
                    for k in range(j + 1, min(j + 3, len(valleys_sorted))):
                        points = valleys_sorted[i : j + 2]
                        if len(points) >= self.trendline_points:
                            fit = self._fit_trendline(points)
                            score = fit["deviation_score"]

                            if score < best_score:
                                best_score = score
                                best_fit = fit

            if best_fit and best_fit["deviation_score"] <= self.max_trendline_deviation:
                return best_fit

            return None

        except Exception as e:
            self.logger.debug(f"Error finding ascending trendline: {e}")
            return None

    def _fit_trendline(self, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fit trendline through points"""
        try:
            x = np.array([p["idx"] for p in points])
            y = np.array([p["price"] for p in points])

            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept

            # Calculate deviation score
            deviations = np.abs(y - y_pred)
            deviation_score = np.mean(deviations) / np.mean(y)

            return {
                "slope": slope,
                "intercept": intercept,
                "deviation_score": deviation_score,
                "points": points,
            }

        except Exception as e:
            self.logger.debug(f"Error fitting trendline: {e}")
            return {
                "slope": 0,
                "intercept": 0,
                "deviation_score": float("inf"),
                "points": [],
            }

    def _check_triangle_formation(
        self,
        horizontal_resistance: Dict[str, Any],
        ascending_trendline: Dict[str, Any],
        data: pd.DataFrame,
    ) -> bool:
        """Check if both lines form a proper triangle"""
        try:
            # Check if trendline is ascending (positive slope)
            if ascending_trendline["slope"] <= 0:
                return False

            # Check if resistance is above trendline start
            trendline_start_price = ascending_trendline["intercept"]
            resistance_price = horizontal_resistance["price"]

            if resistance_price <= trendline_start_price:
                return False

            # Check if triangle is converging
            trendline_end_price = (
                ascending_trendline["slope"] * (len(data) - 1)
                + ascending_trendline["intercept"]
            )
            convergence_ratio = (
                abs(trendline_end_price - resistance_price) / resistance_price
            )

            if convergence_ratio > 0.05:  # More than 5% convergence
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Error checking triangle formation: {e}")
            return False

    def _create_ascending_triangle_signal(
        self, pattern_info: Dict[str, Any], data: pd.DataFrame, end_idx: int
    ) -> Optional[PatternSignal]:
        """Create trading signal for ascending triangle pattern"""
        try:
            # Entry price at breakout above resistance
            entry_price = data.iloc[end_idx]["close"]
            resistance_price = pattern_info["resistance_price"]

            # Stop loss below the trendline
            trendline_x = end_idx - pattern_info["pattern_start_idx"]
            trendline_price = (
                pattern_info["trendline_slope"] * trendline_x
                + pattern_info["trendline_intercept"]
            )
            stop_loss = trendline_price * 0.98  # 2% buffer below trendline

            # Target price based on triangle height
            triangle_height = resistance_price - pattern_info["trendline_intercept"]
            target_price = resistance_price + triangle_height

            # Calculate confidence
            confidence = self._calculate_ascending_triangle_confidence(
                pattern_info, data, end_idx
            )

            # Calculate volume validation
            volume_data = data.iloc[max(0, end_idx - 20) : end_idx + 1]
            avg_volume = volume_data["volume"].mean()
            current_volume = data.iloc[end_idx]["volume"]
            volume_ratio = current_volume / avg_volume

            signal = PatternSignal(
                symbol="UNKNOWN",  # Will be set by PatternEngine
                pattern_type=PatternType.ASCENDING_TRIANGLE,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                timeframe=self.config.timeframe,
                timestamp=data.index[end_idx],
                metadata={
                    "resistance_price": resistance_price,
                    "trendline_slope": pattern_info["trendline_slope"],
                    "trendline_intercept": pattern_info["trendline_intercept"],
                    "triangle_height": triangle_height,
                    "volume_ratio": volume_ratio,
                    "pattern_length": pattern_info["pattern_length"],
                },
                signal_strength=min(volume_ratio / self.min_volume_spike, 1.0),
                risk_level="medium",
            )
            return signal

        except Exception as e:
            self.logger.error(f"Error creating ascending triangle signal: {e}")
            return None

    def _calculate_ascending_triangle_confidence(
        self, pattern_info: Dict[str, Any], data: pd.DataFrame, end_idx: int
    ) -> float:
        """Calculate confidence score for ascending triangle pattern"""
        confidence = 0.7  # Base confidence

        # Triangle formation quality
        deviation_score = pattern_info["ascending_trendline"]["deviation_score"]
        confidence -= deviation_score * 0.2

        # Pattern length
        pattern_length = pattern_info["pattern_length"]
        if pattern_length > 60:
            confidence += 0.1
        elif pattern_length > 40:
            confidence += 0.05

        # Horizontal resistance strength
        resistance_std = pattern_info["horizontal_resistance"]["std_price"]
        confidence -= resistance_std * 0.5

        # Volume confirmation
        volume_data = data.iloc[max(0, end_idx - 20) : end_idx + 1]
        recent_volume = volume_data["volume"].iloc[-5:].mean()
        avg_volume = volume_data["volume"].mean()
        volume_strength = recent_volume / avg_volume
        confidence += min(volume_strength * 0.1, 0.1)

        return min(max(confidence, 0.3), 1.0)
