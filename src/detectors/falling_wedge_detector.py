import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class FallingWedgeDetector(BaseDetector):
    """Detector for Falling Wedge patterns"""

    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Falling Wedge parameters
        self.min_pattern_length = 45  # Minimum days for pattern
        self.max_trendline_deviation = 0.02  # Maximum deviation from trendline
        self.wedge_angle_threshold = 0.15  # Maximum wedge angle in degrees
        self.min_volume_spike = 1.1  # Minimum volume spike for breakout
        self.trendline_points = 5  # Minimum points to define trendlines
        self.breakout_threshold = 0.01  # Price movement threshold for breakout
        self.convergence_threshold = 0.05  # Maximum convergence as percentage

    def get_required_columns(self) -> List[str]:
        """Get required columns for pattern detection"""
        return ['open', 'high', 'low', 'close', 'volume']

    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect Falling Wedge patterns in the data"""
        signals = []

        if len(data) < self.min_pattern_length:
            return signals

        try:
            # Look for potential falling wedge patterns
            for i in range(self.min_pattern_length, len(data)):
                pattern_signals = self._analyze_falling_wedge(data, i)
                signals.extend(pattern_signals)

        except Exception as e:
            self.logger.error(f"Error in falling wedge detection: {e}")

        return signals

    def _analyze_falling_wedge(self, data: pd.DataFrame, end_idx: int) -> List[PatternSignal]:
        """Analyze potential falling wedge pattern ending at end_idx"""
        signals = []

        try:
            # Define search window
            start_idx = max(0, end_idx - self.min_pattern_length - 30)

            # Check for falling wedge formation
            pattern_info = self._identify_falling_wedge(data, start_idx, end_idx)

            if pattern_info:
                signal = self._create_falling_wedge_signal(pattern_info, data, end_idx)
                if signal:
                    signals.append(signal)

        except Exception as e:
            self.logger.debug(f"Error analyzing falling wedge: {e}")

        return signals

    def _identify_falling_wedge(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[Dict[str, Any]]:
        """Identify falling wedge pattern"""
        try:
            window_data = data.iloc[start_idx:end_idx]

            # Find upper trendline (connecting lower highs)
            upper_trendline = self._find_upper_trendline(window_data)
            if not upper_trendline:
                return None

            # Find lower trendline (connecting lower lows)
            lower_trendline = self._find_lower_trendline(window_data)
            if not lower_trendline:
                return None

            # Check if both lines form a falling wedge
            if not self._check_wedge_formation(upper_trendline, lower_trendline, window_data):
                return None

            # Check for breakout above upper trendline
            current_price = data.iloc[end_idx]['close']
            trendline_price_at_end = self._calculate_trendline_price(upper_trendline, end_idx - start_idx)

            if current_price > trendline_price_at_end * (1 + self.breakout_threshold):
                return {
                    'upper_trendline': upper_trendline,
                    'lower_trendline': lower_trendline,
                    'pattern_start_idx': start_idx,
                    'pattern_end_idx': end_idx,
                    'upper_slope': upper_trendline['slope'],
                    'upper_intercept': upper_trendline['intercept'],
                    'lower_slope': lower_trendline['slope'],
                    'lower_intercept': lower_trendline['intercept'],
                    'pattern_length': end_idx - start_idx,
                    'wedge_angle': self._calculate_wedge_angle(upper_trendline, lower_trendline)
                }

            return None

        except Exception as e:
            self.logger.debug(f"Error identifying falling wedge: {e}")
            return None

    def _find_upper_trendline(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find upper trendline (connecting lower highs)"""
        try:
            # Find local maxima
            peaks = []
            for i in range(1, len(data) - 1):
                if (data.iloc[i]['high'] >= data.iloc[i-1]['high'] and
                    data.iloc[i]['high'] >= data.iloc[i+1]['high'] and
                    data.iloc[i]['high'] > data.iloc[i-1]['close'] and
                    data.iloc[i]['high'] > data.iloc[i+1]['close']):
                    peaks.append({
                        'idx': i,
                        'price': data.iloc[i]['high'],
                        'timestamp': data.index[i]
                    })

            if len(peaks) < self.trendline_points:
                return None

            # Sort peaks by index
            peaks_sorted = sorted(peaks, key=lambda x: x['idx'])

            # Try different combinations for best fit
            best_fit = None
            best_score = float('inf')

            # Use last N points for trendline
            for start in range(0, len(peaks_sorted) - 3):
                points = peaks_sorted[start:start+4]
                fit = self._fit_trendline(points)
                score = fit['deviation_score']

                if score < best_score:
                    best_score = score
                    best_fit = fit

            if best_fit and best_fit['deviation_score'] <= self.max_trendline_deviation:
                return best_fit

            return None

        except Exception as e:
            self.logger.debug(f"Error finding upper trendline: {e}")
            return None

    def _find_lower_trendline(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Find lower trendline (connecting lower lows)"""
        try:
            # Find local minima
            valleys = []
            for i in range(1, len(data) - 1):
                if (data.iloc[i]['low'] <= data.iloc[i-1]['low'] and
                    data.iloc[i]['low'] <= data.iloc[i+1]['low'] and
                    data.iloc[i]['low'] < data.iloc[i-1]['close'] and
                    data.iloc[i]['low'] < data.iloc[i+1]['close']):
                    valleys.append({
                        'idx': i,
                        'price': data.iloc[i]['low'],
                        'timestamp': data.index[i]
                    })

            if len(valleys) < self.trendline_points:
                return None

            # Sort valleys by index
            valleys_sorted = sorted(valleys, key=lambda x: x['idx'])

            # Try different combinations for best fit
            best_fit = None
            best_score = float('inf')

            # Use last N points for trendline
            for start in range(0, len(valleys_sorted) - 3):
                points = valleys_sorted[start:start+4]
                fit = self._fit_trendline(points)
                score = fit['deviation_score']

                if score < best_score:
                    best_score = score
                    best_fit = fit

            if best_fit and best_fit['deviation_score'] <= self.max_trendline_deviation:
                return best_fit

            return None

        except Exception as e:
            self.logger.debug(f"Error finding lower trendline: {e}")
            return None

    def _fit_trendline(self, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fit trendline through points"""
        try:
            x = np.array([p['idx'] for p in points])
            y = np.array([p['price'] for p in points])

            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept

            # Calculate deviation score
            deviations = np.abs(y - y_pred)
            deviation_score = np.mean(deviations) / np.mean(y)

            return {
                'slope': slope,
                'intercept': intercept,
                'deviation_score': deviation_score,
                'points': points
            }

        except Exception as e:
            self.logger.debug(f"Error fitting trendline: {e}")
            return {'slope': 0, 'intercept': 0, 'deviation_score': float('inf'), 'points': []}

    def _check_wedge_formation(self, upper_trendline: Dict[str, Any], lower_trendline: Dict[str, Any], data: pd.DataFrame) -> bool:
        """Check if both lines form a valid falling wedge"""
        try:
            # Check if both trendlines are descending
            if upper_trendline['slope'] >= 0 or lower_trendline['slope'] >= 0:
                return False

            # Check if lower trendline is steeper than upper trendline
            if lower_trendline['slope'] >= upper_trendline['slope']:
                return False

            # Check wedge angle
            wedge_angle = self._calculate_wedge_angle(upper_trendline, lower_trendline)
            if wedge_angle > self.wedge_angle_threshold:
                return False

            # Check convergence (lines should be converging)
            upper_end_price = upper_trendline['slope'] * (len(data) - 1) + upper_trendline['intercept']
            lower_end_price = lower_trendline['slope'] * (len(data) - 1) + lower_trendline['intercept']

            convergence_ratio = abs(upper_end_price - lower_end_price) / max(upper_end_price, lower_end_price)
            if convergence_ratio > self.convergence_threshold:
                return False

            # Check if wedge is formed (upper line above lower line)
            for i in range(0, len(data), 5):  # Check every 5th point
                upper_price = upper_trendline['slope'] * i + upper_trendline['intercept']
                lower_price = lower_trendline['slope'] * i + lower_trendline['intercept']
                if upper_price <= lower_price:
                    return False

            return True

        except Exception as e:
            self.logger.debug(f"Error checking wedge formation: {e}")
            return False

    def _calculate_wedge_angle(self, upper_trendline: Dict[str, Any], lower_trendline: Dict[str, Any]) -> float:
        """Calculate wedge angle between two trendlines"""
        try:
            # Calculate angle between two lines
            slope1 = upper_trendline['slope']
            slope2 = lower_trendline['slope']

            # Angle in radians
            angle_rad = np.arctan(abs(slope1 - slope2) / (1 + slope1 * slope2))

            # Convert to degrees
            angle_deg = np.degrees(angle_rad)

            return angle_deg

        except Exception as e:
            self.logger.debug(f"Error calculating wedge angle: {e}")
            return float('inf')

    def _calculate_trendline_price(self, trendline: Dict[str, Any], x: int) -> float:
        """Calculate trendline price at given x position"""
        try:
            return trendline['slope'] * x + trendline['intercept']
        except Exception as e:
            self.logger.debug(f"Error calculating trendline price: {e}")
            return 0.0

    def _create_falling_wedge_signal(self, pattern_info: Dict[str, Any], data: pd.DataFrame, end_idx: int) -> Optional[PatternSignal]:
        """Create trading signal for falling wedge pattern"""
        try:
            # Entry price at breakout above upper trendline
            entry_price = data.iloc[end_idx]['close']
            trendline_price = self._calculate_trendline_price(pattern_info['upper_trendline'], end_idx - pattern_info['pattern_start_idx'])

            # Stop loss below the lower trendline
            lower_trendline_price = self._calculate_trendline_price(pattern_info['lower_trendline'], end_idx - pattern_info['pattern_start_idx'])
            stop_loss = lower_trendline_price * 0.98  # 2% buffer below lower trendline

            # Target price based on wedge height
            wedge_height = trendline_price - lower_trendline_price
            target_price = trendline_price + wedge_height * 0.5  # Conservative target

            # Calculate confidence
            confidence = self._calculate_falling_wedge_confidence(pattern_info, data, end_idx)

            # Calculate volume validation
            volume_data = data.iloc[max(0, end_idx-20):end_idx+1]
            avg_volume = volume_data['volume'].mean()
            current_volume = data.iloc[end_idx]['volume']
            volume_ratio = current_volume / avg_volume

            signal = PatternSignal(
                symbol="UNKNOWN",  # Will be set by PatternEngine
                pattern_type=PatternType.FALLING_WEDGE,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                timeframe=self.config.timeframe,
                timestamp=data.index[end_idx],
                metadata={
                    'upper_trendline_slope': pattern_info['upper_slope'],
                    'upper_trendline_intercept': pattern_info['upper_intercept'],
                    'lower_trendline_slope': pattern_info['lower_slope'],
                    'lower_trendline_intercept': pattern_info['lower_intercept'],
                    'wedge_angle': pattern_info['wedge_angle'],
                    'volume_ratio': volume_ratio,
                    'pattern_length': pattern_info['pattern_length']
                },
                signal_strength=min(volume_ratio / self.min_volume_spike, 1.0),
                risk_level="medium"
            )
            return signal

        except Exception as e:
            self.logger.error(f"Error creating falling wedge signal: {e}")
            return None

    def _calculate_falling_wedge_confidence(self, pattern_info: Dict[str, Any], data: pd.DataFrame, end_idx: int) -> float:
        """Calculate confidence score for falling wedge pattern"""
        confidence = 0.7  # Base confidence

        # Trendline quality
        upper_deviation = pattern_info['upper_trendline']['deviation_score']
        lower_deviation = pattern_info['lower_trendline']['deviation_score']
        confidence -= (upper_deviation + lower_deviation) * 0.1

        # Wedge angle (smaller angles are more reliable)
        wedge_angle = pattern_info['wedge_angle']
        if wedge_angle < 0.1:
            confidence += 0.1
        elif wedge_angle < 0.15:
            confidence += 0.05

        # Pattern length
        pattern_length = pattern_info['pattern_length']
        if pattern_length > 60:
            confidence += 0.1
        elif pattern_length > 45:
            confidence += 0.05

        # Convergence (better convergence = more reliable)
        upper_end_price = pattern_info['upper_slope'] * (pattern_length - 1) + pattern_info['upper_intercept']
        lower_end_price = pattern_info['lower_slope'] * (pattern_length - 1) + pattern_info['lower_intercept']
        convergence = abs(upper_end_price - lower_end_price) / max(upper_end_price, lower_end_price)
        confidence -= convergence * 0.5

        # Volume confirmation
        volume_data = data.iloc[max(0, end_idx-20):end_idx+1]
        recent_volume = volume_data['volume'].iloc[-5:].mean()
        avg_volume = volume_data['volume'].mean()
        volume_strength = recent_volume / avg_volume
        confidence += min(volume_strength * 0.1, 0.1)

        return min(max(confidence, 0.3), 1.0)