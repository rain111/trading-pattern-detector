import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class CupHandleDetector(BaseDetector):
    """Detector for Cup and Handle patterns"""

    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Cup pattern parameters
        self.min_cup_length = 60  # Minimum days for cup formation
        self.max_cup_depth = 0.15  # Maximum cup depth as percentage of cup high
        self.min_handle_length = 10  # Minimum days for handle formation
        self.max_handle_drop = 0.05  # Maximum handle drop as percentage
        self.min_volume_surge = 1.2  # Minimum volume surge for breakout

        # Handle parameters
        self.handle_trend_support = 0.98  # Support level trend (descending support)

    def get_required_columns(self) -> List[str]:
        """Get required columns for pattern detection"""
        return ["open", "high", "low", "close", "volume"]

    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect Cup and Handle patterns in the data"""
        signals = []

        if len(data) < self.min_cup_length + self.min_handle_length:
            return signals

        try:
            # Look for potential cup patterns
            for i in range(self.min_cup_length, len(data) - self.min_handle_length):
                cup_signals = self._analyze_cup_pattern(data, i)
                signals.extend(cup_signals)

        except Exception as e:
            self.logger.error(f"Error in cup handle detection: {e}")

        return signals

    def _analyze_cup_pattern(
        self, data: pd.DataFrame, end_idx: int
    ) -> List[PatternSignal]:
        """Analyze potential cup pattern ending at end_idx"""
        signals = []

        try:
            # Define cup region
            cup_start = max(0, end_idx - self.min_cup_length)
            cup_data = data.iloc[cup_start:end_idx]

            # Find cup high and low points
            cup_high = cup_data["high"].max()
            cup_low = cup_data["low"].min()

            # Calculate cup depth
            cup_depth = (cup_high - cup_low) / cup_high

            # Check cup depth requirement
            if cup_depth > self.max_cup_depth:
                return signals

            # Check if cup has proper U-shape
            cup_mid_idx = cup_start + len(cup_data) // 2
            first_half_high = cup_data.iloc[: len(cup_data) // 2]["high"].mean()
            second_half_high = cup_data.iloc[len(cup_data) // 2 :]["high"].mean()

            # Cup should show rounding bottom pattern
            if second_half_high > first_half_high * 1.02:  # Right side higher
                # Check for handle formation
                handle_signals = self._analyze_handle_pattern(
                    data, end_idx, cup_high, cup_low, cup_depth
                )
                signals.extend(handle_signals)

        except Exception as e:
            self.logger.debug(f"Error analyzing cup pattern: {e}")

        return signals

    def _analyze_handle_pattern(
        self,
        data: pd.DataFrame,
        cup_end_idx: int,
        cup_high: float,
        cup_low: float,
        cup_depth: float,
    ) -> List[PatternSignal]:
        """Analyze handle pattern after cup formation"""
        signals = []

        try:
            # Define handle region (after cup formation)
            handle_start = cup_end_idx
            handle_end = min(
                len(data), handle_start + self.min_handle_length + 20
            )  # Look ahead

            if handle_start >= handle_end:
                return signals

            handle_data = data.iloc[handle_start:handle_end]

            # Check handle characteristics
            handle_high = handle_data["high"].max()
            handle_low = handle_data["low"].min()

            # Handle should not exceed cup high significantly
            if handle_high > cup_high * 1.02:
                return signals

            # Check handle depth (should be shallow)
            handle_depth = (cup_high - handle_low) / cup_high
            if handle_depth > self.max_handle_drop:
                return signals

            # Check for descending support trend
            support_points = []
            for i in range(0, len(handle_data) - 1, 3):  # Sample every 3 days
                if i + 1 < len(handle_data):
                    # Find local low in this window
                    window_low = handle_data.iloc[i : i + 3]["low"].min()
                    support_points.append(window_low)

            # Check if support points show descending trend
            if len(support_points) >= 2:
                trend = np.polyfit(range(len(support_points)), support_points, 1)[0]
                if trend > -0.001:  # Should be descending or flat
                    return signals

            # Check volume during handle (should be declining)
            handle_volume = handle_data["volume"].mean()
            cup_volume = data.iloc[handle_start - self.min_cup_length : handle_start][
                "volume"
            ].mean()

            if handle_volume > cup_volume * 0.8:  # Handle volume should be lower
                return signals

            # Check for breakout signal
            if len(data) > handle_end:
                breakout_data = data.iloc[handle_end : handle_end + 5]  # Next 5 days
                if len(breakout_data) > 0:
                    breakout_high = breakout_data["high"].iloc[0]
                    breakout_volume = breakout_data["volume"].iloc[0]

                    # Breakout above cup high with volume surge
                    if (
                        breakout_high > cup_high
                        and breakout_volume > handle_volume * self.min_volume_surge
                    ):
                        # Calculate target price
                        cup_height = cup_high - cup_low
                        target_price = (
                            cup_high + cup_height * 0.3
                        )  # 30% cup height target

                        # Calculate stop loss (below handle low)
                        stop_loss = handle_low * 0.98

                        # Calculate confidence based on pattern quality
                        confidence = self._calculate_cup_handle_confidence(
                            cup_depth,
                            handle_depth,
                            handle_volume / cup_volume,
                            breakout_volume / handle_volume,
                        )

                        if confidence >= self.config.min_confidence:
                            signal = PatternSignal(
                                symbol="",  # Will be set by PatternEngine
                                pattern_type=PatternType.CUP_HANDLE,
                                confidence=confidence,
                                entry_price=breakout_high,
                                stop_loss=stop_loss,
                                target_price=target_price,
                                timeframe=self.config.timeframe,
                                timestamp=(
                                    data.index[handle_end]
                                    if handle_end < len(data.index)
                                    else data.index[-1]
                                ),
                                metadata={
                                    "cup_high": cup_high,
                                    "cup_low": cup_low,
                                    "handle_low": handle_low,
                                    "cup_depth": cup_depth,
                                    "handle_depth": handle_depth,
                                    "volume_ratio": breakout_volume / handle_volume,
                                    "pattern_quality": (
                                        "high" if confidence > 0.8 else "medium"
                                    ),
                                },
                            )
                            signals.append(signal)

        except Exception as e:
            self.logger.debug(f"Error analyzing handle pattern: {e}")

        return signals

    def _calculate_cup_handle_confidence(
        self,
        cup_depth: float,
        handle_depth: float,
        volume_ratio: float,
        breakout_volume: float,
    ) -> float:
        """Calculate confidence score for cup and handle pattern"""

        # Base confidence factors
        depth_score = 1.0 - (cup_depth / self.max_cup_depth)  # Deeper cup = lower score
        handle_score = 1.0 - (
            handle_depth / self.max_handle_drop
        )  # Deeper handle = lower score
        volume_score = min(
            1.0, breakout_volume / 2.0
        )  # Higher volume surge = better score

        # Weighted average
        confidence = 0.3 * depth_score + 0.2 * handle_score + 0.5 * volume_score

        return max(0.0, min(1.0, confidence))
