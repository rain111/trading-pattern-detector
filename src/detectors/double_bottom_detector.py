import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class DoubleBottomDetector(BaseDetector):
    """Detector for Double Bottom patterns"""

    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Double bottom parameters
        self.min_pattern_length = 40  # Minimum days for pattern
        self.max_bottom_distance = 0.03  # Maximum distance between bottoms as percentage
        self.min_retracement_height = 0.02  # Minimum retracement height between bottoms
        self.max_neckline_distance = 0.015  # Maximum distance from neckline as percentage
        self.min_volume_spike = 1.5  # Minimum volume spike for breakout
        self.neckline_lookback = 10  # Lookback period for neckline

    def get_required_columns(self) -> List[str]:
        """Get required columns for pattern detection"""
        return ['open', 'high', 'low', 'close', 'volume']

    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect Double Bottom patterns in the data"""
        signals = []

        if len(data) < self.min_pattern_length:
            return signals

        try:
            # Look for potential double bottom patterns
            for i in range(self.min_pattern_length, len(data)):
                bottom_signals = self._analyze_double_bottom(data, i)
                signals.extend(bottom_signals)

        except Exception as e:
            self.logger.error(f"Error in double bottom detection: {e}")

        return signals

    def _analyze_double_bottom(self, data: pd.DataFrame, end_idx: int) -> List[PatternSignal]:
        """Analyze potential double bottom pattern ending at end_idx"""
        signals = []

        try:
            # Define search window for double bottom
            search_start = max(0, end_idx - self.min_pattern_length * 2)
            search_data = data.iloc[search_start:end_idx]

            # Find potential bottom points
            bottom_candidates = self._find_bottom_candidates(search_data)

            # Check for double bottom patterns
            for i in range(len(bottom_candidates)):
                for j in range(i + 1, len(bottom_candidates)):
                    bottom1_idx, bottom1_price = bottom_candidates[i]
                    bottom2_idx, bottom2_price = bottom_candidates[j]

                    # Check bottom distance requirement
                    price_distance = abs(bottom1_price - bottom2_price) / bottom1_price
                    if price_distance > self.max_bottom_distance:
                        continue

                    # Check pattern timing
                    pattern_duration = bottom2_idx - bottom1_idx
                    if pattern_duration < self.min_pattern_length // 2:
                        continue

                    # Check for neckline formation
                    neckline_signals = self._analyze_neckline_and_breakout(
                        data, bottom1_idx, bottom2_idx, end_idx, bottom1_price, bottom2_price
                    )

                    signals.extend(neckline_signals)

        except Exception as e:
            self.logger.debug(f"Error analyzing double bottom: {e}")

        return signals

    def _find_bottom_candidates(self, data: pd.DataFrame) -> List[tuple]:
        """Find potential bottom points in the data"""
        bottom_candidates = []

        try:
            # Find local minima
            for i in range(1, len(data) - 1):
                current_low = data['low'].iloc[i]
                prev_low = data['low'].iloc[i - 1]
                next_low = data['low'].iloc[i + 1]

                # Check if it's a local minimum
                if current_low < prev_low and current_low < next_low:
                    # Check if it's significant (not too small a dip)
                    recent_high = data.iloc[max(0, i-5):i+5]['high'].max()
                    dip_ratio = (recent_high - current_low) / recent_high

                    if dip_ratio > self.min_retracement_height:
                        bottom_candidates.append((i, current_low))

        except Exception as e:
            self.logger.debug(f"Error finding bottom candidates: {e}")

        return bottom_candidates

    def _analyze_neckline_and_breakout(self, data: pd.DataFrame, bottom1_idx: int,
                                     bottom2_idx: int, current_idx: int,
                                     bottom1_price: float, bottom2_price: float) -> List[PatternSignal]:
        """Analyze neckline and breakout signal"""
        signals = []

        try:
            # Determine which bottom is higher (neckline should be above both)
            higher_bottom = max(bottom1_price, bottom2_price)
            lower_bottom = min(bottom1_price, bottom2_price)

            # Calculate neckline level (slightly above higher bottom)
            neckline = higher_bottom * (1 + self.max_neckline_distance)

            # Check if price has broken above neckline
            breakout_idx = None
            breakout_price = None
            breakout_volume = None

            # Look for breakout in recent data
            breakout_search_start = bottom2_idx + 5  # Some time after second bottom
            if current_idx > breakout_search_start:
                breakout_search_data = data.iloc[breakout_search_start:current_idx]

                for i, row in breakout_search_data.iterrows():
                    if row['high'] > neckline:
                        breakout_idx = i
                        breakout_price = row['high']
                        breakout_volume = row['volume']
                        break

            if breakout_idx is None:
                return signals

            # Check volume surge on breakout
            avg_volume = data.iloc[breakout_search_start:breakout_idx]['volume'].mean()
            if breakout_volume < avg_volume * self.min_volume_spike:
                return signals

            # Calculate pattern measurements
            pattern_height = neckline - lower_bottom
            target_price = neckline + pattern_height  # Measured move

            # Calculate stop loss (below lower bottom)
            stop_loss = lower_bottom * 0.98

            # Calculate confidence based on pattern quality
            confidence = self._calculate_double_bottom_confidence(
                abs(bottom1_price - bottom2_price) / lower_bottom,
                pattern_height / neckline,
                breakout_volume / avg_volume,
                (breakout_idx - bottom1_idx) / self.min_pattern_length
            )

            if confidence >= self.config.min_confidence:
                signal = PatternSignal(
                    symbol="",  # Will be set by PatternEngine
                    pattern_type=PatternType.DOUBLE_BOTTOM,
                    confidence=confidence,
                    entry_price=breakout_price,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    timeframe=self.config.timeframe,
                    timestamp=data.index[breakout_idx],
                    metadata={
                        'bottom1_price': bottom1_price,
                        'bottom2_price': bottom2_price,
                        'neckline': neckline,
                        'pattern_height': pattern_height,
                        'volume_ratio': breakout_volume / avg_volume,
                        'pattern_duration': breakout_idx - bottom1_idx,
                        'pattern_quality': 'high' if confidence > 0.8 else 'medium'
                    }
                )
                signals.append(signal)

        except Exception as e:
            self.logger.debug(f"Error analyzing neckline and breakout: {e}")

        return signals

    def _calculate_double_bottom_confidence(self, bottom_distance_ratio: float,
                                          height_ratio: float, volume_ratio: float,
                                          duration_ratio: float) -> float:
        """Calculate confidence score for double bottom pattern"""

        # Base confidence factors
        distance_score = 1.0 - (bottom_distance_ratio / self.max_bottom_distance)
        height_score = min(1.0, height_ratio / 0.1)  # Higher pattern = better score
        volume_score = min(1.0, volume_ratio / 3.0)  # Higher volume = better score
        duration_score = min(1.0, duration_ratio)  # Longer pattern = better score

        # Weighted average
        confidence = (0.2 * distance_score + 0.2 * height_score +
                    0.4 * volume_score + 0.2 * duration_score)

        return max(0.0, min(1.0, confidence))