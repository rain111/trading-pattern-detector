import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging


class VolumeAnalyzer:
    """Volume pattern analysis for confirmation signals"""

    def __init__(self, volume_window: int = 20):
        self.volume_window = volume_window
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns in the data"""
        try:
            avg_volume = data["volume"].rolling(window=self.volume_window).mean()
            volume_ratio = data["volume"] / avg_volume

            # Volume trends
            volume_trend = np.polyfit(range(len(data)), data["volume"].values, 1)[0]
            recent_volume_trend = np.polyfit(
                range(min(10, len(data))), data["volume"].tail(10).values, 1
            )[0]

            # Volume spikes
            volume_spikes = volume_ratio > 2.0
            volume_spike_count = volume_spikes.sum()

            # Volume accumulation/distribution
            volume_price_trend = self._calculate_volume_price_trend(data)

            # OBV (On Balance Volume)
            obv = self._calculate_obv(data)

            return {
                "volume_ratio": volume_ratio.iloc[-1] if len(volume_ratio) > 0 else 1.0,
                "volume_trend": volume_trend,
                "recent_volume_trend": recent_volume_trend,
                "volume_spike_count": volume_spike_count,
                "volume_spike_ratio": volume_spike_count / len(data),
                "volume_spike_active": (
                    volume_spikes.iloc[-1] if len(volume_spikes) > 0 else False
                ),
                "volume_price_trend": volume_price_trend,
                "obv": obv,
                "avg_volume": avg_volume.iloc[-1] if len(avg_volume) > 0 else 0,
                "volume_std": (
                    data["volume"].rolling(window=self.volume_window).std().iloc[-1]
                    if len(data) > 0
                    else 0
                ),
            }
        except Exception as e:
            self.logger.error(f"Error analyzing volume patterns: {e}")
            return {}

    def confirm_volume_breakout(self, data: pd.DataFrame, breakout_index: int) -> bool:
        """Confirm breakout with volume analysis"""
        try:
            if breakout_index >= len(data):
                return False

            breakout_volume = data["volume"].iloc[breakout_index]
            avg_volume = data["volume"].iloc[:breakout_index].mean()

            return breakout_volume > avg_volume * 1.5
        except Exception as e:
            self.logger.error(f"Error confirming volume breakout: {e}")
            return False

    def detect_volume_surge(
        self, data: pd.DataFrame, threshold: float = 2.0, lookback: int = 5
    ) -> bool:
        """Detect volume surge above threshold"""
        try:
            recent_avg = data["volume"].tail(lookback).mean()
            historical_avg = data["volume"].iloc[:-lookback].mean()

            return recent_avg > historical_avg * threshold
        except Exception as e:
            self.logger.error(f"Error detecting volume surge: {e}")
            return False

    def calculate_money_flow_index(
        self, data: pd.DataFrame, period: int = 14
    ) -> pd.Series:
        """Calculate Money Flow Index"""
        try:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            money_flow = typical_price * data["volume"]

            positive_flow = money_flow.copy()
            negative_flow = money_flow.copy()

            positive_flow[typical_price < typical_price.shift(1)] = 0
            negative_flow[typical_price > typical_price.shift(1)] = 0

            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()

            money_ratio = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + money_ratio))

            return mfi
        except Exception as e:
            self.logger.error(f"Error calculating Money Flow Index: {e}")
            return pd.Series()

    def calculate_volume_weighted_average_price(
        self, data: pd.DataFrame, period: int = 20
    ) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)"""
        try:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).rolling(window=period).sum() / data[
                "volume"
            ].rolling(window=period).sum()

            return vwap
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return pd.Series()

    def _calculate_volume_price_trend(self, data: pd.DataFrame) -> float:
        """Calculate volume-price correlation"""
        try:
            returns = data["close"].pct_change()
            volume_changes = data["volume"].pct_change()

            # Remove NaN values
            valid_data = pd.concat([returns, volume_changes], axis=1).dropna()

            if len(valid_data) < 2:
                return 0.0

            correlation = valid_data["close"].corr(valid_data["volume"])

            return correlation if not np.isnan(correlation) else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating volume-price trend: {e}")
            return 0.0

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        try:
            obv = pd.Series(index=data.index, dtype=float)
            obv.iloc[0] = data["volume"].iloc[0]

            for i in range(1, len(data)):
                if data["close"].iloc[i] > data["close"].iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] + data["volume"].iloc[i]
                elif data["close"].iloc[i] < data["close"].iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] - data["volume"].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i - 1]

            return obv
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return pd.Series()

    def detect_volume_divergence(
        self, data: pd.DataFrame, lookback: int = 20
    ) -> Dict[str, Any]:
        """Detect volume divergence with price"""
        try:
            price_change = data["close"].iloc[-1] - data["close"].iloc[-lookback]
            volume_change = data["volume"].iloc[-1] - data["volume"].iloc[-lookback]

            # Bullish divergence: price down, volume up
            bullish_divergence = price_change < 0 and volume_change > 0

            # Bearish divergence: price up, volume down
            bearish_divergence = price_change > 0 and volume_change < 0

            return {
                "bullish_divergence": bullish_divergence,
                "bearish_divergence": bearish_divergence,
                "price_change_pct": (price_change / data["close"].iloc[-lookback])
                * 100,
                "volume_change_pct": (volume_change / data["volume"].iloc[-lookback])
                * 100,
            }
        except Exception as e:
            self.logger.error(f"Error detecting volume divergence: {e}")
            return {
                "bullish_divergence": False,
                "bearish_divergence": False,
                "price_change_pct": 0,
                "volume_change_pct": 0,
            }
