import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class RoundingBottomDetector(BaseDetector):
    """Detector for Rounding Bottom patterns"""

    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Rounding Bottom parameters
        self.min_pattern_length = 50  # Minimum days for pattern
        self.max_decline_range = 0.15  # Maximum decline during pattern formation
        self.min_bottom_width = 15  # Minimum width of bottom in days
        self.min_volume_spike = 1.3  # Minimum volume spike for breakout
        self.curvature_threshold = 0.02  # Threshold for detecting curvature
        self.neckline_distance = 0.005  # Maximum distance from neckline for breakout

    def get_required_columns(self) -> List[str]:
        """Get required columns for pattern detection"""
        return ['open', 'high', 'low', 'close', 'volume']

    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect Rounding Bottom patterns in the data"""
        signals = []

        if len(data) < self.min_pattern_length:
            return signals

        try:
            # Look for potential rounding bottom patterns
            for i in range(self.min_pattern_length, len(data)):
                pattern_signals = self._analyze_rounding_bottom(data, i)
                signals.extend(pattern_signals)

        except Exception as e:
            self.logger.error(f"Error in rounding bottom detection: {e}")

        return signals

    def _analyze_rounding_bottom(self, data: pd.DataFrame, end_idx: int) -> List[PatternSignal]:
        """Analyze potential rounding bottom pattern ending at end_idx"""
        signals = []

        try:
            # Define search window
            start_idx = max(0, end_idx - self.min_pattern_length)

            # Check for rounding bottom formation
            pattern_info = self._identify_rounding_bottom(data, start_idx, end_idx)

            if pattern_info:
                signal = self._create_rounding_bottom_signal(pattern_info, data, end_idx)
                if signal:
                    signals.append(signal)

        except Exception as e:
            self.logger.debug(f"Error analyzing rounding bottom: {e}")

        return signals

    def _identify_rounding_bottom(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[Dict[str, Any]]:
        """Identify rounding bottom pattern"""
        try:
            window_data = data.iloc[start_idx:end_idx]

            # Find the lowest point (bottom)
            bottom_idx = window_data['low'].idxmin()
            bottom_price = window_data['low'].min()

            # Check if bottom is not at the edge of the window
            if bottom_idx <= start_idx + self.min_bottom_width or bottom_idx >= end_idx - self.min_bottom_width:
                return None

            # Find the highest point before bottom (decline start)
            pre_bottom = window_data.loc[:bottom_idx]
            decline_start_idx = pre_bottom['high'].idxmax()
            decline_start_price = pre_bottom['high'].max()

            # Find the highest point after bottom (potential breakout)
            post_bottom = window_data.loc[bottom_idx:]
            breakout_idx = post_bottom['high'].idxmax()
            breakout_price = post_bottom['high'].max()
            breakout_absolute_idx = start_idx + breakout_idx

            # Check decline magnitude
            decline_percentage = (decline_start_price - bottom_price) / decline_start_price
            if decline_percentage > self.max_decline_range:
                return None

            # Calculate curvature score
            curvature_score = self._calculate_curvature_score(window_data, bottom_idx)

            # Check for smooth rounding bottom
            if curvature_score < self.curvature_threshold:
                return None

            # Calculate neckline (horizontal line at decline start level)
            neckline = decline_start_price

            # Check if current price is approaching or breaking neckline
            current_price = data.iloc[end_idx]['close']
            price_distance_from_neckline = abs(current_price - neckline) / neckline

            return {
                'decline_start_idx': decline_start_idx + start_idx,
                'decline_start_price': decline_start_price,
                'bottom_idx': bottom_idx + start_idx,
                'bottom_price': bottom_price,
                'breakout_idx': breakout_absolute_idx,
                'breakout_price': breakout_price,
                'neckline': neckline,
                'decline_percentage': decline_percentage,
                'curvature_score': curvature_score,
                'price_distance_from_neckline': price_distance_from_neckline,
                'pattern_length': end_idx - start_idx
            }

        except Exception as e:
            self.logger.debug(f"Error identifying rounding bottom: {e}")
            return None

    def _calculate_curvature_score(self, data: pd.DataFrame, bottom_idx: int) -> float:
        """Calculate curvature score for the bottom pattern"""
        try:
            # Select points around the bottom
            left_range = min(20, bottom_idx)
            right_range = min(20, len(data) - bottom_idx - 1)

            left_data = data.iloc[bottom_idx - left_range:bottom_idx]
            right_data = data.iloc[bottom_idx + 1:bottom_idx + right_range + 1]

            if len(left_data) < 5 or len(right_data) < 5:
                return float('inf')

            # Calculate slopes
            left_slope = (left_data['close'].iloc[-1] - left_data['close'].iloc[0]) / len(left_data)
            right_slope = (right_data['close'].iloc[-1] - right_data['close'].iloc[0]) / len(right_data)

            # Check for U-shape (negative slope on left, positive slope on right)
            if left_slope < 0 and right_slope > 0:
                # Calculate curvature using polynomial fit
                x_left = np.arange(len(left_data))
                x_right = np.arange(len(right_data))

                # Fit quadratic to left side
                left_coeffs = np.polyfit(x_left, left_data['close'].values, 2)
                right_coeffs = np.polyfit(x_right, right_data['close'].values, 2)

                # Calculate curvature (second derivative)
                left_curvature = abs(2 * left_coeffs[0])
                right_curvature = abs(2 * right_coeffs[0])

                return max(left_curvature, right_curvature)
            else:
                return float('inf')

        except Exception as e:
            self.logger.debug(f"Error calculating curvature score: {e}")
            return float('inf')

    def _create_rounding_bottom_signal(self, pattern_info: Dict[str, Any], data: pd.DataFrame, end_idx: int) -> Optional[PatternSignal]:
        """Create trading signal for rounding bottom pattern"""
        try:
            # Check for breakout above neckline
            current_price = data.iloc[end_idx]['close']
            neckline = pattern_info['neckline']

            if current_price > neckline:
                # Entry price at breakout
                entry_price = current_price

                # Stop loss below bottom
                stop_loss = pattern_info['bottom_price'] * 0.98  # 2% buffer

                # Target price based on decline magnitude
                decline_range = pattern_info['decline_start_price'] - pattern_info['bottom_price']
                target_price = neckline + decline_range * 1.0  # 1:1 risk/reward

                # Calculate confidence
                confidence = self._calculate_rounding_bottom_confidence(pattern_info)

                # Calculate volume validation
                volume_data = data.iloc[max(0, end_idx-20):end_idx+1]
                avg_volume = volume_data['volume'].mean()
                current_volume = data.iloc[end_idx]['volume']
                volume_ratio = current_volume / avg_volume

                signal = PatternSignal(
                    symbol="UNKNOWN",  # Will be set by PatternEngine
                    pattern_type=PatternType.ROUNDING_BOTTOM,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    timeframe=self.config.timeframe,
                    timestamp=data.index[end_idx],
                    metadata={
                        'decline_start_price': pattern_info['decline_start_price'],
                        'bottom_price': pattern_info['bottom_price'],
                        'neckline': neckline,
                        'decline_percentage': pattern_info['decline_percentage'],
                        'curvature_score': pattern_info['curvature_score'],
                        'volume_ratio': volume_ratio,
                        'pattern_length': pattern_info['pattern_length']
                    },
                    signal_strength=min(volume_ratio / self.min_volume_spike, 1.0),
                    risk_level="low"
                )
                return signal
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error creating rounding bottom signal: {e}")
            return None

    def _calculate_rounding_bottom_confidence(self, pattern_info: Dict[str, Any]) -> float:
        """Calculate confidence score for rounding bottom pattern"""
        confidence = 0.7  # Base confidence

        # Decline magnitude (moderate declines are more reliable)
        decline_percentage = pattern_info['decline_percentage']
        if 0.05 < decline_percentage < 0.12:
            confidence += 0.1
        elif 0.03 < decline_percentage <= 0.05:
            confidence += 0.05

        # Curvature score (smoother bottoms are better)
        curvature_score = pattern_info['curvature_score']
        if curvature_score < 0.01:
            confidence += 0.1
        elif curvature_score < 0.02:
            confidence += 0.05

        # Pattern length (longer patterns are more reliable)
        pattern_length = pattern_info['pattern_length']
        if pattern_length > 80:
            confidence += 0.1
        elif pattern_length > 60:
            confidence += 0.05

        # Distance from bottom (patterns that are well-formed)
        time_from_bottom = pattern_info['breakout_idx'] - pattern_info['bottom_idx']
        if time_from_bottom > 20:
            confidence += 0.05

        return min(confidence, 1.0)