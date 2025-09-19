import pandas as pd
import numpy as np
from typing import List, Dict, Any
from scipy import stats
import logging


class TrendAnalyzer:
    """Trend detection and analysis"""

    def __init__(self, trend_period: int = 50):
        self.trend_period = trend_period
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze price trends"""
        try:
            close_prices = data["close"]

            if len(close_prices) < 2:
                return {
                    "slope": 0,
                    "trend_direction": "neutral",
                    "trend_strength": 0,
                    "r_squared": 0,
                    "ma_convergence": 0,
                    "is_strong_trend": False,
                }

            # Linear trend analysis
            x = np.arange(len(close_prices))
            y = close_prices.values

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # ADX-like trend strength calculation
            trend_strength = abs(r_value)

            # Trend direction
            trend_direction = "upward" if slope > 0 else "downward"

            # Moving average analysis
            sma_short = close_prices.rolling(window=20).mean()
            sma_long = close_prices.rolling(window=50).mean()

            # Trend strength based on moving average convergence
            ma_convergence = (
                (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
                if len(sma_long) > 0 and not pd.isna(sma_long.iloc[-1])
                else 0
            )

            return {
                "slope": slope,
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "r_squared": r_value**2,
                "ma_convergence": ma_convergence,
                "is_strong_trend": trend_strength > 0.7,
                "price_position": self._determine_price_position(data),
                "trend_robustness": self._calculate_trend_robustness(close_prices),
            }
        except Exception as e:
            self.logger.error(f"Error detecting trend: {e}")
            return {
                "slope": 0,
                "trend_direction": "neutral",
                "trend_strength": 0,
                "r_squared": 0,
                "ma_convergence": 0,
                "is_strong_trend": False,
            }

    def find_swings(self, data: pd.DataFrame) -> Dict[str, List]:
        """Find price swings for pattern detection"""
        try:
            highs = data["high"]
            lows = data["low"]

            swing_highs = []
            swing_lows = []

            # Enhanced swing detection logic
            for i in range(2, len(highs) - 2):
                # Swing high
                if (
                    highs.iloc[i] > highs.iloc[i - 1]
                    and highs.iloc[i] > highs.iloc[i + 1]
                    and highs.iloc[i] > highs.iloc[i - 2]
                    and highs.iloc[i] > highs.iloc[i + 2]
                    and highs.iloc[i] > highs.mean() * 1.02
                ):  # Significant high
                    swing_highs.append(
                        {
                            "index": i,
                            "price": highs.iloc[i],
                            "timestamp": data.index[i],
                            "strength": self._calculate_swing_strength(highs, i),
                        }
                    )

                # Swing low
                if (
                    lows.iloc[i] < lows.iloc[i - 1]
                    and lows.iloc[i] < lows.iloc[i + 1]
                    and lows.iloc[i] < lows.iloc[i - 2]
                    and lows.iloc[i] < lows.iloc[i + 2]
                    and lows.iloc[i] < lows.mean() * 0.98
                ):  # Significant low
                    swing_lows.append(
                        {
                            "index": i,
                            "price": lows.iloc[i],
                            "timestamp": data.index[i],
                            "strength": self._calculate_swing_strength(lows, i),
                        }
                    )

            return {"swing_highs": swing_highs, "swing_lows": swing_lows}
        except Exception as e:
            self.logger.error(f"Error finding swings: {e}")
            return {"swing_highs": [], "swing_lows": []}

    def calculate_momentum(
        self, data: pd.DataFrame, period: int = 10
    ) -> Dict[str, float]:
        """Calculate momentum indicators"""
        try:
            close_prices = data["close"]

            # Rate of Change (ROC)
            roc = (
                (close_prices.iloc[-1] - close_prices.iloc[-period])
                / close_prices.iloc[-period]
                * 100
            )

            # Momentum
            momentum = close_prices.iloc[-1] - close_prices.iloc[-period]

            # Relative Strength
            if len(close_prices) > period * 2:
                strength = np.polyfit(
                    range(period), close_prices.iloc[-period:].values, 1
                )[0]
                reference_strength = np.polyfit(
                    range(period), close_prices.iloc[-period * 2 : -period].values, 1
                )[0]
                relative_strength = (
                    strength / reference_strength if reference_strength != 0 else 1.0
                )
            else:
                relative_strength = 1.0

            return {
                "roc": roc,
                "momentum": momentum,
                "relative_strength": relative_strength,
                "is_momentum_strong": abs(roc) > 5,  # 5% threshold
            }
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {e}")
            return {
                "roc": 0,
                "momentum": 0,
                "relative_strength": 1.0,
                "is_momentum_strong": False,
            }

    def find_trendlines(
        self, data: pd.DataFrame, swing_points: Dict[str, List]
    ) -> List[Dict]:
        """Draw trendlines from swing points"""
        try:
            trendlines = []
            swing_highs = swing_points["swing_highs"]
            swing_lows = swing_points["swing_lows"]

            # Uptrend lines (connecting higher lows)
            for i in range(len(swing_lows) - 1):
                for j in range(i + 1, min(i + 4, len(swing_lows))):
                    point1 = swing_lows[i]
                    point2 = swing_lows[j]

                    # Check if trend has positive slope and proper alignment
                    slope = (point2["price"] - point1["price"]) / (
                        point2["index"] - point1["index"]
                    )

                    if slope > 0 and abs(slope) < 0.1:  # Valid uptrend
                        trendlines.append(
                            {
                                "start_point": point1,
                                "end_point": point2,
                                "slope": slope,
                                "type": "uptrend",
                                "strength": self._calculate_trendline_strength(
                                    point1, point2, swing_highs
                                ),
                            }
                        )

            # Downtrend lines (connecting lower highs)
            for i in range(len(swing_highs) - 1):
                for j in range(i + 1, min(i + 4, len(swing_highs))):
                    point1 = swing_highs[i]
                    point2 = swing_highs[j]

                    # Check if trend has negative slope and proper alignment
                    slope = (point2["price"] - point1["price"]) / (
                        point2["index"] - point1["index"]
                    )

                    if slope < 0 and abs(slope) < 0.1:  # Valid downtrend
                        trendlines.append(
                            {
                                "start_point": point1,
                                "end_point": point2,
                                "slope": slope,
                                "type": "downtrend",
                                "strength": self._calculate_trendline_strength(
                                    point1, point2, swing_lows
                                ),
                            }
                        )

            return trendlines
        except Exception as e:
            self.logger.error(f"Error finding trendlines: {e}")
            return []

    def _determine_price_position(self, data: pd.DataFrame) -> str:
        """Determine current price position relative to moving averages"""
        try:
            close_prices = data["close"]

            if len(close_prices) < 50:
                return "neutral"

            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()

            current_price = close_prices.iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]

            if pd.isna(current_sma_20) or pd.isna(current_sma_50):
                return "neutral"

            if current_price > current_sma_20 > current_sma_50:
                return "above_all"
            elif current_price < current_sma_20 < current_sma_50:
                return "below_all"
            elif current_price > current_sma_20:
                return "above_short"
            else:
                return "below_short"
        except Exception as e:
            self.logger.error(f"Error determining price position: {e}")
            return "neutral"

    def _calculate_trend_robustness(self, prices: pd.Series) -> float:
        """Calculate trend robustness score"""
        try:
            if len(prices) < 10:
                return 0.0

            # Calculate trend consistency over multiple periods
            short_trend = np.polyfit(range(10), prices.tail(10).values, 1)[0]
            medium_trend = np.polyfit(range(20), prices.tail(20).values, 1)[0]
            long_trend = np.polyfit(range(50), prices.tail(50).values, 1)[0]

            # Calculate trend consistency
            trend_consistency = (
                1 - abs(short_trend - medium_trend) / abs(long_trend)
                if long_trend != 0
                else 1
            )
            trend_consistency = max(0, min(1, trend_consistency))

            return trend_consistency
        except Exception as e:
            self.logger.error(f"Error calculating trend robustness: {e}")
            return 0.0

    def _calculate_swing_strength(self, prices: pd.Series, index: int) -> float:
        """Calculate swing strength"""
        try:
            lookback = min(5, index)
            lookforward = min(5, len(prices) - index)

            # Strength based on price deviation from local average
            local_high = prices.iloc[index - lookback : index + lookforward + 1].max()
            local_low = prices.iloc[index - lookback : index + lookforward + 1].min()

            if local_high > local_low:
                return (prices.iloc[index] - local_low) / (local_high - local_low)
            else:
                return 0.5
        except Exception as e:
            self.logger.error(f"Error calculating swing strength: {e}")
            return 0.5

    def _calculate_trendline_strength(
        self, point1: dict, point2: dict, reference_points: List[dict]
    ) -> float:
        """Calculate trendline strength based on how well it holds"""
        try:
            trendline_price = point1["price"] + (point2["price"] - point1["price"]) * (
                point2["index"] - point1["index"]
            )

            # Check how many reference points touch or respect the trendline
            touches = 0
            for point in reference_points:
                if abs(point["price"] - trendline_price) < 0.01 * trendline_price:
                    touches += 1

            return touches / len(reference_points) if reference_points else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating trendline strength: {e}")
            return 0.0
