import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class HeadAndShouldersDetector(BaseDetector):
    """Detector for Head and Shoulders patterns"""

    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Head and Shoulders parameters
        self.min_pattern_length = 60  # Minimum days for pattern
        self.max_peak_distance = 0.04  # Maximum distance between peaks as percentage
        self.min_valley_depth = 0.02  # Minimum valley depth between peaks
        self.neckline_lookback = 20  # Lookback period for neckline
        self.max_neckline_deviation = 0.01  # Maximum deviation from neckline
        self.volume_threshold = 1.2  # Volume spike multiplier for breakout
        self.left_shoulder_min = 15  # Minimum days for left shoulder
        self.right_shoulder_min = 10  # Minimum days for right shoulder

    def get_required_columns(self) -> List[str]:
        """Get required columns for pattern detection"""
        return ['open', 'high', 'low', 'close', 'volume']

    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect Head and Shoulders patterns in the data"""
        signals = []

        if len(data) < self.min_pattern_length:
            return signals

        try:
            # Look for potential head and shoulders patterns
            for i in range(self.min_pattern_length, len(data)):
                pattern_signals = self._analyze_head_and_shoulders(data, i)
                signals.extend(pattern_signals)

        except Exception as e:
            self.logger.error(f"Error in head and shoulders detection: {e}")

        return signals

    def _analyze_head_and_shoulders(self, data: pd.DataFrame, end_idx: int) -> List[PatternSignal]:
        """Analyze potential head and shoulders pattern ending at end_idx"""
        signals = []

        try:
            # Define search window
            start_idx = max(0, end_idx - self.neckline_lookback - 40)

            # Find peaks and valleys
            peaks, valleys = self._find_peaks_and_valleys(data, start_idx, end_idx)

            # Look for head and shoulders formation
            if len(peaks) >= 3 and len(valleys) >= 2:
                patterns = self._identify_head_and_shoulders(peaks, valleys, data, end_idx)
                signals.extend(patterns)

        except Exception as e:
            self.logger.debug(f"Error analyzing head and shoulders: {e}")

        return signals

    def _find_peaks_and_valleys(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> tuple:
        """Find peaks and valleys in the data"""
        try:
            window_data = data.iloc[start_idx:end_idx]

            # Find peaks (local maxima)
            peaks = []
            for i in range(1, len(window_data) - 1):
                if (window_data.iloc[i]['high'] > window_data.iloc[i-1]['high'] and
                    window_data.iloc[i]['high'] > window_data.iloc[i+1]['high']):
                    peaks.append((start_idx + i, window_data.iloc[i]['high']))

            # Find valleys (local minima)
            valleys = []
            for i in range(1, len(window_data) - 1):
                if (window_data.iloc[i]['low'] < window_data.iloc[i-1]['low'] and
                    window_data.iloc[i]['low'] < window_data.iloc[i+1]['low']):
                    valleys.append((start_idx + i, window_data.iloc[i]['low']))

            return peaks, valleys

        except Exception as e:
            self.logger.debug(f"Error finding peaks and valleys: {e}")
            return [], []

    def _identify_head_and_shoulders(self, peaks: list, valleys: list, data: pd.DataFrame, end_idx: int) -> List[PatternSignal]:
        """Identify head and shoulders pattern from peaks and valleys"""
        signals = []

        try:
            # Sort peaks and valleys by index
            peaks_sorted = sorted(peaks, key=lambda x: x[0])
            valleys_sorted = sorted(valleys, key=lambda x: x[0])

            # Look for the pattern: left shoulder, head, right shoulder
            for i in range(len(peaks_sorted) - 2):
                left_shoulder = peaks_sorted[i]
                head = peaks_sorted[i + 1]
                right_shoulder = peaks_sorted[i + 2]

                # Check pattern structure
                if self._is_valid_head_and_shoulders(left_shoulder, head, right_shoulder, valleys_sorted, data, end_idx):
                    signals = self._create_head_and_shoulders_signal(left_shoulder, head, right_shoulder, valleys_sorted, data, end_idx)
                    break

        except Exception as e:
            self.logger.debug(f"Error identifying head and shoulders pattern: {e}")

        return signals

    def _is_valid_head_and_shoulders(self, left_shoulder: tuple, head: tuple, right_shoulder: tuple,
                                   valleys: list, data: pd.DataFrame, end_idx: int) -> bool:
        """Check if peak formation is a valid head and shoulders pattern"""
        try:
            # Check peak heights: head should be highest, shoulders should be roughly equal
            head_height = head[1]
            left_height = left_shoulder[1]
            right_height = right_shoulder[1]

            # Head must be higher than both shoulders
            if head_height <= left_height or head_height <= right_height:
                return False

            # Shoulder heights should be reasonably close
            shoulder_height_diff = abs(left_height - right_height)
            if shoulder_height_diff > head_height * 0.15:  # More than 15% difference
                return False

            # Find valleys between peaks
            left_valley = None
            right_valley = None

            for valley in valleys:
                if left_shoulder[0] < valley[0] < head[0]:
                    left_valley = valley
                elif head[0] < valley[0] < right_shoulder[0]:
                    right_valley = valley
                    break

            # Need both valleys
            if not left_valley or not right_valley:
                return False

            # Check valley depths
            left_depth = head_height - left_valley[1]
            right_depth = head_height - right_valley[1]

            if left_depth < self.min_valley_depth or right_depth < self.min_valley_depth:
                return False

            # Check pattern symmetry
            depth_diff = abs(left_depth - right_depth)
            if depth_diff > max(left_depth, right_depth) * 0.3:  # More than 30% asymmetry
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Error validating head and shoulders: {e}")
            return False

    def _create_head_and_shoulders_signal(self, left_shoulder: tuple, head: tuple, right_shoulder: tuple,
                                        valleys: list, data: pd.DataFrame, end_idx: int) -> List[PatternSignal]:
        """Create trading signal for head and shoulders pattern"""
        signals = []

        try:
            # Find neckline connecting the valleys
            left_valley = None
            right_valley = None

            for valley in valleys:
                if left_shoulder[0] < valley[0] < head[0]:
                    left_valley = valley
                elif head[0] < valley[0] < right_shoulder[0]:
                    right_valley = valley
                    break

            if not left_valley or not right_valley:
                return signals

            # Calculate neckline
            neckline_start = left_valley[1]
            neckline_end = right_valley[1]
            neckline_slope = (neckline_end - neckline_start) / (right_valley[0] - left_valley[0])

            # Entry price (neckline breakout)
            entry_price = neckline_end
            current_high = data.iloc[end_idx]['high']

            # Check for breakout above neckline
            if current_high > neckline_end:
                # Stop loss below right shoulder
                stop_loss = right_shoulder[1]

                # Target price based on head height to neckline distance
                head_to_neckline = head[1] - neckline_end
                target_price = neckline_end + head_to_neckline * 1.0  # 1:1 risk/reward

                # Calculate confidence
                confidence = self._calculate_head_and_shoulders_confidence(
                    left_shoulder, head, right_shoulder, left_valley, right_valley
                )

                # Calculate volume validation
                volume_data = data.iloc[max(0, right_valley[0]-10):right_valley[0]+10]
                avg_volume = volume_data['volume'].mean()
                current_volume = data.iloc[end_idx]['volume']
                volume_ratio = current_volume / avg_volume

                signal = PatternSignal(
                    symbol="UNKNOWN",  # Will be set by PatternEngine
                    pattern_type=PatternType.HEAD_AND_SHOULDERS,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    timeframe=self.config.timeframe,
                    timestamp=data.index[end_idx],
                    metadata={
                        'left_shoulder_price': left_shoulder[1],
                        'head_price': head[1],
                        'right_shoulder_price': right_shoulder[1],
                        'left_valley_price': left_valley[1],
                        'right_valley_price': right_valley[1],
                        'neckline_slope': neckline_slope,
                        'volume_ratio': volume_ratio,
                        'pattern_length': right_shoulder[0] - left_shoulder[0]
                    },
                    signal_strength=min(volume_ratio / self.volume_threshold, 1.0),
                    risk_level="medium"
                )
                signals.append(signal)

        except Exception as e:
            self.logger.error(f"Error creating head and shoulders signal: {e}")

        return signals

    def _calculate_head_and_shoulders_confidence(self, left_shoulder: tuple, head: tuple, right_shoulder: tuple,
                                               left_valley: tuple, right_valley: tuple) -> float:
        """Calculate confidence score for head and shoulders pattern"""
        confidence = 0.6  # Base confidence

        # Shoulder symmetry
        shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
        head_height = head[1]
        shoulder_symmetry = 1.0 - (shoulder_height_diff / head_height)
        confidence += shoulder_symmetry * 0.15

        # Valley symmetry
        valley_depth_left = head[1] - left_valley[1]
        valley_depth_right = head[1] - right_valley[1]
        valley_depth_diff = abs(valley_depth_left - valley_depth_right)
        valley_symmetry = 1.0 - (valley_depth_diff / max(valley_depth_left, valley_depth_right))
        confidence += valley_symmetry * 0.15

        # Pattern length (longer patterns are more reliable)
        pattern_length = right_shoulder[0] - left_shoulder[0]
        if pattern_length > 80:
            confidence += 0.1
        elif pattern_length > 60:
            confidence += 0.05

        return min(confidence, 1.0)