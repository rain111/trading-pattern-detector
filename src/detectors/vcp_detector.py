import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class StageAnalyzer:
    """Analyzer for VCP pattern stages"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_volatility_contraction(
        self, data: pd.DataFrame, start_idx: int, period: int = 30
    ) -> bool:
        """Detect volatility contraction stage"""
        try:
            if start_idx + period >= len(data):
                return False

            # Calculate ATR for the period
            atr_data = data.iloc[start_idx : start_idx + period]
            high, low, close = atr_data["high"], atr_data["low"], atr_data["close"]

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.mean()

            # Check if volatility is contracting
            first_half_atr = true_range.iloc[: period // 2].mean()
            second_half_atr = true_range.iloc[period // 2 :].mean()

            return second_half_atr < first_half_atr * 0.8
        except Exception as e:
            self.logger.error(f"Error detecting volatility contraction: {e}")
            return False

    def detect_consolidation(
        self, data: pd.DataFrame, start_idx: int, max_range: float = 0.05
    ) -> Dict[str, Any]:
        """Detect consolidation stage"""
        try:
            if start_idx + 40 >= len(data):
                return {"is_consolidation": False}

            consolidation_data = data.iloc[start_idx : start_idx + 40]
            high_range = (
                consolidation_data["high"].max() - consolidation_data["high"].min()
            )
            low_range = (
                consolidation_data["low"].max() - consolidation_data["low"].min()
            )
            price_range = (
                consolidation_data["close"].max() - consolidation_data["close"].min()
            )
            avg_price = consolidation_data["close"].mean()

            # Check if consolidation range is acceptable
            is_consolidation = (price_range / avg_price) < max_range

            return {
                "is_consolidation": is_consolidation,
                "price_range": price_range,
                "price_range_pct": price_range / avg_price,
                "support": consolidation_data["low"].min(),
                "resistance": consolidation_data["high"].max(),
                "consolidation_length": len(consolidation_data),
            }
        except Exception as e:
            self.logger.error(f"Error detecting consolidation: {e}")
            return {"is_consolidation": False}

    def detect_breakout(
        self,
        data: pd.DataFrame,
        consolidation_info: Dict,
        breakout_threshold: float = 0.02,
    ) -> Dict[str, Any]:
        """Detect breakout stage"""
        try:
            consolidation_start = consolidation_info.get("start_idx", 0)
            consolidation_end = consolidation_start + consolidation_info.get(
                "consolidation_length", 0
            )

            if consolidation_end + 5 >= len(data):
                return {"breakout_confirmed": False}

            breakout_data = data.iloc[consolidation_end : consolidation_end + 5]
            resistance = consolidation_info["resistance"]

            # Check for breakout above resistance
            breakout_confirmed = breakout_data["high"].max() > resistance * (
                1 + breakout_threshold
            )

            if breakout_confirmed:
                breakout_price = breakout_data[
                    breakout_data["high"] > resistance * (1 + breakout_threshold)
                ].iloc[0]["high"]
                volume_spike = self._check_volume_spike(data, consolidation_end)

                return {
                    "breakout_confirmed": True,
                    "breakout_price": breakout_price,
                    "breakout_strength": (breakout_price - resistance) / resistance,
                    "volume_spike": volume_spike,
                    "confidence": self._calculate_breakout_confidence(
                        breakout_data, consolidation_info
                    ),
                }
            else:
                return {"breakout_confirmed": False}
        except Exception as e:
            self.logger.error(f"Error detecting breakout: {e}")
            return {"breakout_confirmed": False}

    def _check_volume_spike(self, data: pd.DataFrame, breakout_idx: int) -> bool:
        """Check for volume spike at breakout"""
        try:
            if breakout_idx >= len(data):
                return False

            breakout_volume = data["volume"].iloc[breakout_idx]
            avg_volume = data["volume"].iloc[:breakout_idx].mean()

            return breakout_volume > avg_volume * 1.5
        except Exception as e:
            self.logger.error(f"Error checking volume spike: {e}")
            return False

    def _calculate_breakout_confidence(
        self, breakout_data: pd.DataFrame, consolidation_info: Dict
    ) -> float:
        """Calculate breakout confidence"""
        try:
            confidence = 0.5

            # Volume confidence
            volume_ratio = breakout_data["volume"].mean() / consolidation_info.get(
                "avg_volume", 1
            )
            confidence += min(volume_ratio * 0.2, 0.2)

            # Price momentum confidence
            momentum = (
                breakout_data["close"].iloc[-1] - breakout_data["close"].iloc[0]
            ) / breakout_data["close"].iloc[0]
            confidence += min(abs(momentum) * 3, 0.3)

            return min(confidence, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating breakout confidence: {e}")
            return 0.5


class VCPBreakoutDetector(BaseDetector):
    """VCP (Volatility Contraction Pattern) Breakout Detector"""

    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.stage_analyzer = StageAnalyzer()
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect VCP patterns"""
        if not self.validate_data(data):
            return []

        data = self.preprocess_data(data)
        signals = []

        # Scan for potential VCP patterns
        for i in range(len(data) - 100):  # Minimum pattern length
            if self._is_potential_vcp_start(data, i):
                vcp_signals = self._analyze_vcp_pattern(data, i)
                signals.extend(vcp_signals)

        return self.validate_signals(signals)

    def get_required_columns(self) -> List[str]:
        return ["open", "high", "low", "close", "volume"]

    def _is_potential_vcp_start(self, data: pd.DataFrame, start_idx: int) -> bool:
        """Check if position could be start of VCP pattern"""
        try:
            if start_idx + 20 >= len(data):
                return False

            # Check for initial decline (Stage 1)
            decline_data = data.iloc[start_idx : start_idx + 20]
            decline = (
                decline_data["close"].iloc[-1] - decline_data["close"].iloc[0]
            ) / decline_data["close"].iloc[0]

            # Check for sufficient volume during decline
            avg_decline_volume = decline_data["volume"].mean()
            avg_volume = data["volume"].iloc[:start_idx].mean()
            volume_during_decline = avg_decline_volume > avg_volume

            return decline < -0.10 and volume_during_decline  # 10% decline with volume
        except Exception as e:
            self.logger.error(f"Error checking potential VCP start: {e}")
            return False

    def _analyze_vcp_pattern(
        self, data: pd.DataFrame, start_idx: int
    ) -> List[PatternSignal]:
        """Complete VCP pattern analysis"""
        try:
            # Stage 1: Initial decline (already confirmed)

            # Stage 2: Volatility contraction
            if not self.stage_analyzer.detect_volatility_contraction(data, start_idx):
                return []

            # Stage 3: Consolidation
            consolidation_info = self._detect_consolidation_with_support_resistance(
                data, start_idx
            )

            if not consolidation_info["is_consolidation"]:
                return []

            # Store start index for breakout detection
            consolidation_info["start_idx"] = start_idx

            # Stage 4: Breakout
            breakout_info = self.stage_analyzer.detect_breakout(
                data, consolidation_info
            )

            if not breakout_info["breakout_confirmed"]:
                return []

            # Generate signal
            return [
                self._generate_vcp_signal(
                    data, start_idx, consolidation_info, breakout_info
                )
            ]

        except Exception as e:
            self.logger.error(f"Error analyzing VCP pattern: {e}")
            return []

    def _detect_consolidation_with_support_resistance(
        self, data: pd.DataFrame, start_idx: int
    ) -> Dict[str, Any]:
        """Enhanced consolidation detection with support/resistance"""
        try:
            consolidation_data = data.iloc[start_idx : start_idx + 40]
            close_prices = consolidation_data["close"]

            # Find support and resistance levels
            support = consolidation_data["low"].min()
            resistance = consolidation_data["high"].max()

            # Check for proper consolidation characteristics
            price_range = resistance - support
            avg_price = close_prices.mean()
            range_ratio = price_range / avg_price

            # Check for multiple touches of support/resistance
            support_touches = (consolidation_data["low"] <= support * 1.005).sum()
            resistance_touches = (
                consolidation_data["high"] >= resistance * 0.995
            ).sum()

            return {
                "is_consolidation": range_ratio < 0.05
                and support_touches >= 2
                and resistance_touches >= 2,
                "price_range": price_range,
                "range_ratio": range_ratio,
                "support": support,
                "resistance": resistance,
                "avg_volume": consolidation_data["volume"].mean(),
                "support_touches": support_touches,
                "resistance_touches": resistance_touches,
                "consolidation_length": len(consolidation_data),
            }
        except Exception as e:
            self.logger.error(f"Error detecting consolidation with SR: {e}")
            return {"is_consolidation": False}

    def _generate_vcp_signal(
        self,
        data: pd.DataFrame,
        start_idx: int,
        consolidation_info: dict,
        breakout_info: dict,
    ) -> PatternSignal:
        """Generate VCP trading signal"""
        try:
            current_price = data["close"].iloc[-1]
            breakout_price = breakout_info["breakout_price"]

            # Calculate risk parameters
            risk_distance = current_price - consolidation_info["support"]
            target_distance = risk_distance * self.config.reward_ratio
            target_price = current_price + target_distance

            # Calculate stop loss (below support or breakout failure)
            stop_loss = min(
                consolidation_info["support"] * 0.98, breakout_price * 0.995
            )

            # Pattern metadata
            pattern_data = {
                "consolidation_range": consolidation_info["price_range"],
                "consolidation_ratio": consolidation_info["range_ratio"],
                "support_level": consolidation_info["support"],
                "resistance_level": consolidation_info["resistance"],
                "breakout_strength": breakout_info["breakout_strength"],
                "volume_spike": breakout_info["volume_spike"],
                "support_touches": consolidation_info["support_touches"],
                "resistance_touches": consolidation_info["resistance_touches"],
                "volatility_contraction": self._calculate_volatility_contraction_score(
                    data, start_idx
                ),
            }

            return PatternSignal(
                symbol="UNKNOWN",  # Will be set by caller
                pattern_type=PatternType.VCP_BREAKOUT,
                confidence=breakout_info["confidence"],
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                timeframe=self.config.timeframe,
                timestamp=data.index[-1],
                metadata=pattern_data,
                signal_strength=self._calculate_signal_strength(pattern_data),
                risk_level=self._determine_risk_level(pattern_data),
                expected_duration="2-4 weeks",
                probability_target=0.65,
            )
        except Exception as e:
            self.logger.error(f"Error generating VCP signal: {e}")
            raise

    def _calculate_volatility_contraction_score(
        self, data: pd.DataFrame, start_idx: int
    ) -> float:
        """Calculate volatility contraction score"""
        try:
            decline_period = 20
            consolidation_period = 30

            # Decline period volatility
            decline_data = data.iloc[start_idx : start_idx + decline_period]
            decline_volatility = decline_data["close"].pct_change().std()

            # Consolidation period volatility
            consolidation_start = start_idx + decline_period
            consolidation_data = data.iloc[
                consolidation_start : consolidation_start + consolidation_period
            ]
            consolidation_volatility = consolidation_data["close"].pct_change().std()

            # Contraction score (higher is better)
            if decline_volatility > 0:
                return max(0, 1 - (consolidation_volatility / decline_volatility))
            else:
                return 0
        except Exception as e:
            self.logger.error(f"Error calculating volatility contraction score: {e}")
            return 0

    def _calculate_signal_strength(self, pattern_data: dict) -> float:
        """Calculate VCP signal strength"""
        try:
            strength = 0.5

            # Volume strength
            if pattern_data.get("volume_spike", False):
                strength += 0.2

            # Support/resistance touches strength
            touches = pattern_data.get("support_touches", 0) + pattern_data.get(
                "resistance_touches", 0
            )
            strength += min(touches * 0.05, 0.2)

            # Volatility contraction strength
            volatility_score = pattern_data.get("volatility_contraction", 0)
            strength += volatility_score * 0.2

            # Breakout strength
            breakout_strength = pattern_data.get("breakout_strength", 0)
            strength += min(breakout_strength * 2, 0.2)

            return min(strength, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.5

    def _determine_risk_level(self, pattern_data: dict) -> str:
        """Determine VCP risk level"""
        try:
            risk_score = 0

            # Range ratio risk
            range_ratio = pattern_data.get("consolidation_ratio", 0)
            risk_score += range_ratio * 5

            # Breakout strength risk
            breakout_strength = pattern_data.get("breakout_strength", 0)
            risk_score += breakout_strength * 2

            # Touch count risk (fewer touches = higher risk)
            touches = pattern_data.get("support_touches", 0) + pattern_data.get(
                "resistance_touches", 0
            )
            risk_score += (5 - min(touches, 5)) * 0.1

            if risk_score > 0.6:
                return "high"
            elif risk_score > 0.3:
                return "medium"
            else:
                return "low"
        except Exception as e:
            self.logger.error(f"Error determining risk level: {e}")
            return "medium"
