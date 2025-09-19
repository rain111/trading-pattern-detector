import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class DescendingTriangleDetector(BaseDetector):
    """Detector for Descending Triangle patterns"""

    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Descending Triangle parameters
        self.min_pattern_length = 40  # Minimum days for pattern
        self.max_trendline_deviation = 0.015  # Maximum deviation from trendline
        self.max_horizontal_deviation = 0.008  # Maximum deviation from horizontal line
        self.min_volume_spike = 1.2  # Minimum volume spike for breakdown
        self.trendline_points = 3  # Minimum points to define trendline
        self.horizontal_points = 3  # Minimum points to define horizontal line
        self.breakdown_threshold = 0.01  # Price movement threshold for breakdown

    def get_required_columns(self) -> List[str]:
        """Get required columns for pattern detection"""
        return ["open", "high", "low", "close", "volume"]

    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect Descending Triangle patterns in the data"""
        signals = []

        if len(data) < self.min_pattern_length:
            return signals

        try:
            # Look for potential descending triangle patterns
            for i in range(self.min_pattern_length, len(data)):
                pattern_signals = self._analyze_descending_triangle(data, i)
                signals.extend(pattern_signals)

        except Exception as e:
            self.logger.error(f"Error in descending triangle detection: {e}")

        return signals

    def _analyze_descending_triangle(
        self, data: pd.DataFrame, end_idx: int
    ) -> List[PatternSignal]:
        """Analyze potential descending triangle pattern ending at end_idx"""
        signals = []

        try:
            # Define search window
            start_idx = max(0, end_idx - self.min_pattern_length - 20)

            # Check for descending triangle formation
            pattern_info = self._identify_descending_triangle(data, start_idx, end_idx)

            if pattern_info:
                signal = self._create_descending_triangle_signal(
                    pattern_info, data, end_idx
                )
                if signal:
                    signals.append(signal)

        except Exception as e:
            self.logger.debug(f"Error analyzing descending triangle: {e}")

        return signals

    def _identify_descending_triangle(
        self, data: pd.DataFrame, start_idx: int, end_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Identify descending triangle pattern"""
        try:
            window_data = data.iloc[start_idx:end_idx]

            # Find potential horizontal support line
            horizontal_support = self._find_horizontal_support(window_data)
            if not horizontal_support:
                return None

            # Find descending trendline
            descending_trendline = self._find_descending_trendline(
                window_data, horizontal_support["price"]
            )
            if not descending_trendline:
                return None

            # Check if both lines intersect properly
            if not self._check_triangle_formation(
                horizontal_support, descending_trendline, window_data
            ):
                return None

            # Check for breakdown below horizontal support
            current_price = data.iloc[end_idx]["close"]
            if current_price < horizontal_support["price"] * (
                1 - self.breakdown_threshold
            ):
                return {
                    "horizontal_support": horizontal_support,
                    "descending_trendline": descending_trendline,
                    "pattern_start_idx": start_idx,
                    "pattern_end_idx": end_idx,
                    "support_price": horizontal_support["price"],
                    "trendline_slope": descending_trendline["slope"],
                    "trendline_intercept": descending_trendline["intercept"],
                    "pattern_length": end_idx - start_idx,
                }

            return None

        except Exception as e:
            self.logger.debug(f"Error identifying descending triangle: {e}")
            return None

    def _find_horizontal_support(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find horizontal support line"""
        try:
            # Find local minima
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

            if len(valleys) < self.horizontal_points:
                return None

            # Check for horizontal support (valleys at similar levels)
            prices = [v["price"] for v in valleys]
            mean_price = np.mean(prices)
            std_price = np.std(prices)

            # Select valleys that are close to mean price
            support_valleys = [
                v
                for v in valleys
                if abs(v["price"] - mean_price) / mean_price
                <= self.max_horizontal_deviation
            ]

            if len(support_valleys) < self.horizontal_points:
                return None

            # Use the most recent valley as support level
            support_price = support_valleys[-1]["price"]

            return {
                "price": support_price,
                "valleys": support_valleys,
                "mean_price": mean_price,
                "std_price": std_price,
            }

        except Exception as e:
            self.logger.debug(f"Error finding horizontal support: {e}")
            return None

    def _find_descending_trendline(
        self, data: pd.DataFrame, support_price: float
    ) -> Optional[Dict[str, Any]]:
        """Find descending trendline"""
        try:
            # Find local maxima for trendline
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

            if len(peaks) < self.trendline_points:
                return None

            # Sort peaks by index
            peaks_sorted = sorted(peaks, key=lambda x: x["idx"])

            # Try different combinations of peak points
            best_fit = None
            best_score = float("inf")

            for i in range(len(peaks_sorted) - 1):
                for j in range(i + 1, len(peaks_sorted)):
                    for k in range(j + 1, min(j + 3, len(peaks_sorted))):
                        points = peaks_sorted[i : j + 2]
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
            self.logger.debug(f"Error finding descending trendline: {e}")
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
        horizontal_support: Dict[str, Any],
        descending_trendline: Dict[str, Any],
        data: pd.DataFrame,
    ) -> bool:
        """Check if both lines form a proper triangle"""
        try:
            # Check if trendline is descending (negative slope)
            if descending_trendline["slope"] >= 0:
                return False

            # Check if support is below trendline start
            trendline_start_price = descending_trendline["intercept"]
            support_price = horizontal_support["price"]

            if support_price >= trendline_start_price:
                return False

            # Check if triangle is converging
            trendline_end_price = (
                descending_trendline["slope"] * (len(data) - 1)
                + descending_trendline["intercept"]
            )
            convergence_ratio = abs(trendline_end_price - support_price) / support_price

            if convergence_ratio > 0.05:  # More than 5% convergence
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Error checking triangle formation: {e}")
            return False

    def _create_descending_triangle_signal(
        self, pattern_info: Dict[str, Any], data: pd.DataFrame, end_idx: int
    ) -> Optional[PatternSignal]:
        """Create trading signal for descending triangle pattern"""
        try:
            # Entry price at breakdown below support
            entry_price = data.iloc[end_idx]["close"]
            support_price = pattern_info["support_price"]

            # Stop loss above the trendline
            trendline_x = end_idx - pattern_info["pattern_start_idx"]
            trendline_price = (
                pattern_info["trendline_slope"] * trendline_x
                + pattern_info["trendline_intercept"]
            )
            stop_loss = trendline_price * 1.02  # 2% buffer above trendline

            # Target price based on triangle height
            triangle_height = pattern_info["trendline_intercept"] - support_price
            target_price = support_price - triangle_height

            # Calculate confidence
            confidence = self._calculate_descending_triangle_confidence(
                pattern_info, data, end_idx
            )

            # Calculate volume validation
            volume_data = data.iloc[max(0, end_idx - 20) : end_idx + 1]
            avg_volume = volume_data["volume"].mean()
            current_volume = data.iloc[end_idx]["volume"]
            volume_ratio = current_volume / avg_volume

            signal = PatternSignal(
                symbol="UNKNOWN",  # Will be set by PatternEngine
                pattern_type=PatternType.DESCENDING_TRIANGLE,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                timeframe=self.config.timeframe,
                timestamp=data.index[end_idx],
                metadata={
                    "support_price": support_price,
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
            self.logger.error(f"Error creating descending triangle signal: {e}")
            return None

    def _calculate_descending_triangle_confidence(
        self, pattern_info: Dict[str, Any], data: pd.DataFrame, end_idx: int
    ) -> float:
        """Calculate confidence score for descending triangle pattern"""
        confidence = 0.7  # Base confidence

        # Triangle formation quality
        deviation_score = pattern_info["descending_trendline"]["deviation_score"]
        confidence -= deviation_score * 0.2

        # Pattern length
        pattern_length = pattern_info["pattern_length"]
        if pattern_length > 60:
            confidence += 0.1
        elif pattern_length > 40:
            confidence += 0.05

        # Horizontal support strength
        support_std = pattern_info["horizontal_support"]["std_price"]
        confidence -= support_std * 0.5

        # Volume confirmation
        volume_data = data.iloc[max(0, end_idx - 20) : end_idx + 1]
        recent_volume = volume_data["volume"].iloc[-5:].mean()
        avg_volume = volume_data["volume"].mean()
        volume_strength = recent_volume / avg_volume
        confidence += min(volume_strength * 0.1, 0.1)

        return min(max(confidence, 0.3), 1.0)
