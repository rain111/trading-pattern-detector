import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging


class SupportResistanceDetector:
    """Support and resistance level detection"""

    def __init__(self, tolerance: float = 0.02):
        self.tolerance = tolerance
        self.logger = logging.getLogger(self.__class__.__name__)

    def find_support(
        self, data: pd.DataFrame, tolerance: float = None
    ) -> List[Dict[str, Any]]:
        """Find support levels"""
        try:
            tolerance = tolerance or self.tolerance
            lows = data["low"]
            support_levels = []

            # Find local minima with significance threshold
            for i in range(1, len(lows) - 1):
                if (
                    lows.iloc[i] < lows.iloc[i - 1]
                    and lows.iloc[i] < lows.iloc[i + 1]
                    and lows.iloc[i] < lows.mean() * (1 - tolerance)
                ):

                    # Check how many times this level has been tested
                    test_count = self._count_level_tests(
                        data, lows.iloc[i], is_support=True
                    )

                    support_levels.append(
                        {
                            "price": lows.iloc[i],
                            "index": i,
                            "timestamp": data.index[i],
                            "test_count": test_count,
                            "strength": self._calculate_level_strength(
                                data, lows.iloc[i], is_support=True
                            ),
                            "zone": self._create_price_zone(lows.iloc[i], tolerance),
                        }
                    )

            # Sort by strength and combine nearby levels
            support_levels = sorted(
                support_levels, key=lambda x: x["strength"], reverse=True
            )
            support_levels = self._combine_nearby_levels(support_levels)

            return support_levels
        except Exception as e:
            self.logger.error(f"Error finding support levels: {e}")
            return []

    def find_resistance(
        self, data: pd.DataFrame, tolerance: float = None
    ) -> List[Dict[str, Any]]:
        """Find resistance levels"""
        try:
            tolerance = tolerance or self.tolerance
            highs = data["high"]
            resistance_levels = []

            # Find local maxima with significance threshold
            for i in range(1, len(highs) - 1):
                if (
                    highs.iloc[i] > highs.iloc[i - 1]
                    and highs.iloc[i] > highs.iloc[i + 1]
                    and highs.iloc[i] > highs.mean() * (1 + tolerance)
                ):

                    # Check how many times this level has been tested
                    test_count = self._count_level_tests(
                        data, highs.iloc[i], is_support=False
                    )

                    resistance_levels.append(
                        {
                            "price": highs.iloc[i],
                            "index": i,
                            "timestamp": data.index[i],
                            "test_count": test_count,
                            "strength": self._calculate_level_strength(
                                data, highs.iloc[i], is_support=False
                            ),
                            "zone": self._create_price_zone(highs.iloc[i], tolerance),
                        }
                    )

            # Sort by strength and combine nearby levels
            resistance_levels = sorted(
                resistance_levels, key=lambda x: x["strength"], reverse=True
            )
            resistance_levels = self._combine_nearby_levels(resistance_levels)

            return resistance_levels
        except Exception as e:
            self.logger.error(f"Error finding resistance levels: {e}")
            return []

    def find_dynamic_support_resistance(
        self, data: pd.DataFrame, period: int = 20
    ) -> Dict[str, pd.Series]:
        """Find dynamic support and resistance using moving averages"""
        try:
            # Use moving averages as dynamic support/resistance
            sma_20 = data["close"].rolling(window=period).mean()
            sma_50 = data["close"].rolling(window=period * 2).mean()

            # Dynamic levels based on price action
            dynamic_support = data["low"].rolling(window=period).min()
            dynamic_resistance = data["high"].rolling(window=period).max()

            return {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "dynamic_support": dynamic_support,
                "dynamic_resistance": dynamic_resistance,
                "support_resistance_zone": self._calculate_sr_zone(
                    dynamic_support, dynamic_resistance
                ),
            }
        except Exception as e:
            self.logger.error(f"Error finding dynamic support/resistance: {e}")
            return {}

    def draw_trendlines(
        self, data: pd.DataFrame, swing_points: Dict[str, List]
    ) -> List[Dict]:
        """Draw trendlines from swing points"""
        try:
            trendlines = []
            swing_highs = swing_points["swing_highs"]
            swing_lows = swing_points["swing_lows"]

            # Trendline parameters
            min_points = 2
            max_distance = 10  # Maximum index distance between points

            # Uptrend lines (connecting higher lows)
            self._add_trendline_type(
                trendlines, swing_lows, "uptrend", min_points, max_distance
            )

            # Downtrend lines (connecting lower highs)
            self._add_trendline_type(
                trendlines, swing_highs, "downtrend", min_points, max_distance
            )

            # Horizontal lines (significant levels)
            significant_levels = self._find_significant_levels(swing_highs, swing_lows)
            for level in significant_levels:
                trendlines.append(
                    {
                        "start_point": level,
                        "end_point": level,
                        "slope": 0,
                        "type": "horizontal",
                        "strength": level["strength"],
                        "is_broken": self._is_level_broken(
                            data, level["price"], level["is_support"]
                        ),
                    }
                )

            return sorted(trendlines, key=lambda x: x["strength"], reverse=True)
        except Exception as e:
            self.logger.error(f"Error drawing trendlines: {e}")
            return []

    def get_current_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get current market context regarding support/resistance"""
        try:
            current_price = data["close"].iloc[-1]
            current_index = len(data) - 1

            support_levels = self.find_support(data)
            resistance_levels = self.find_resistance(data)

            # Find nearest levels
            nearest_support = self._find_nearest_level(
                current_price, support_levels, is_support=True
            )
            nearest_resistance = self._find_nearest_level(
                current_price, resistance_levels, is_support=False
            )

            # Calculate distance to levels
            support_distance = (
                (current_price - nearest_support["price"])
                / nearest_support["price"]
                * 100
                if nearest_support
                else float("inf")
            )
            resistance_distance = (
                (nearest_resistance["price"] - current_price)
                / nearest_resistance["price"]
                * 100
                if nearest_resistance
                else float("inf")
            )

            # Determine market position
            market_position = self._determine_market_position(
                current_price, support_levels, resistance_levels
            )

            return {
                "current_price": current_price,
                "market_position": market_position,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_distance_pct": support_distance,
                "resistance_distance_pct": resistance_distance,
                "support_resistance_ratio": (
                    resistance_distance / support_distance
                    if support_distance != 0 and resistance_distance != 0
                    else 1.0
                ),
                "key_levels": {
                    "strong_support": sorted(
                        [s for s in support_levels if s["strength"] > 0.7],
                        key=lambda x: x["strength"],
                        reverse=True,
                    )[:3],
                    "strong_resistance": sorted(
                        [r for r in resistance_levels if r["strength"] > 0.7],
                        key=lambda x: x["strength"],
                        reverse=True,
                    )[:3],
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting market context: {e}")
            return {}

    def _count_level_tests(
        self, data: pd.DataFrame, level_price: float, is_support: bool
    ) -> int:
        """Count how many times a level has been tested"""
        try:
            if is_support:
                # Check how many times price has touched the support level from below
                test_points = data[data["low"] <= level_price * (1 + self.tolerance)]
            else:
                # Check how many times price has touched the resistance level from above
                test_points = data[data["high"] >= level_price * (1 - self.tolerance)]

            return len(test_points)
        except Exception as e:
            self.logger.error(f"Error counting level tests: {e}")
            return 0

    def _calculate_level_strength(
        self, data: pd.DataFrame, level_price: float, is_support: bool
    ) -> float:
        """Calculate level strength based on multiple factors"""
        try:
            test_count = self._count_level_tests(data, level_price, is_support)

            # Calculate strength based on test count, recency, and significance
            base_strength = min(test_count * 0.2, 1.0)

            # Factor in recent activity
            recent_tests = test_count if len(data) > 50 else test_count * 1.5

            # Factor in price significance (distance from average)
            avg_price = data["close"].mean()
            significance = abs(level_price - avg_price) / avg_price
            significance_factor = min(significance * 2, 0.5)

            return base_strength + significance_factor
        except Exception as e:
            self.logger.error(f"Error calculating level strength: {e}")
            return 0.0

    def _create_price_zone(
        self, level_price: float, tolerance: float
    ) -> Dict[str, float]:
        """Create support/resistance price zone"""
        return {
            "upper": level_price * (1 + tolerance),
            "lower": level_price * (1 - tolerance),
            "center": level_price,
        }

    def _combine_nearby_levels(
        self, levels: List[Dict], distance_threshold: float = 0.01
    ) -> List[Dict]:
        """Combine nearby levels to avoid duplication"""
        if not levels:
            return []

        combined = []
        processed = set()

        for i, level in enumerate(levels):
            if i in processed:
                continue

            # Find nearby levels
            nearby = [level]
            for j, other in enumerate(levels):
                if i != j and j not in processed:
                    distance = abs(level["price"] - other["price"]) / level["price"]
                    if distance < distance_threshold:
                        nearby.append(other)
                        processed.add(j)

            # Combine levels
            avg_price = np.mean([n["price"] for n in nearby])
            avg_strength = np.mean([n["strength"] for n in nearby])
            avg_test_count = np.mean([n["test_count"] for n in nearby])

            combined.append(
                {
                    "price": avg_price,
                    "zone": self._create_price_zone(avg_price, self.tolerance),
                    "strength": avg_strength,
                    "test_count": int(avg_test_count),
                    "original_levels": len(nearby),
                }
            )

            processed.add(i)

        return sorted(combined, key=lambda x: x["strength"], reverse=True)

    def _add_trendline_type(
        self,
        trendlines: List[Dict],
        points: List[str],
        trend_type: str,
        min_points: int,
        max_distance: int,
    ):
        """Add specific type of trendline"""
        for i in range(len(points)):
            for j in range(i + 1, min(i + max_distance, len(points))):
                point1 = points[i]
                point2 = points[j]

                # Calculate trendline slope
                slope = (point2["price"] - point1["price"]) / (
                    point2["index"] - point1["index"]
                )

                # Validate slope based on trend type
                if trend_type == "uptrend" and slope <= 0:
                    continue
                elif trend_type == "downtrend" and slope >= 0:
                    continue

                # Calculate strength
                strength = self._calculate_trendline_strength(point1, point2, points)

                if strength > 0.3:  # Minimum strength threshold
                    trendlines.append(
                        {
                            "start_point": point1,
                            "end_point": point2,
                            "slope": slope,
                            "type": trend_type,
                            "strength": strength,
                            "is_broken": False,
                        }
                    )

    def _find_significant_levels(
        self, swing_highs: List[Dict], swing_lows: List[Dict]
    ) -> List[Dict]:
        """Find significant horizontal levels"""
        significant_levels = []

        # Combine highs and lows and sort by price
        all_levels = []
        for high in swing_highs:
            if high["strength"] > 0.5:
                all_levels.append(
                    {
                        "price": high["price"],
                        "is_support": False,
                        "strength": high["strength"],
                    }
                )

        for low in swing_lows:
            if low["strength"] > 0.5:
                all_levels.append(
                    {
                        "price": low["price"],
                        "is_support": True,
                        "strength": low["strength"],
                    }
                )

        return sorted(all_levels, key=lambda x: x["strength"], reverse=True)

    def _find_nearest_level(
        self, price: float, levels: List[Dict], is_support: bool
    ) -> Dict:
        """Find the nearest support/resistance level"""
        if not levels:
            return {}

        # Filter by type
        filtered_levels = [
            level for level in levels if level.get("is_support", None) == is_support
        ]

        if not filtered_levels:
            return {}

        # Find nearest level
        nearest = min(filtered_levels, key=lambda x: abs(x["price"] - price))

        return nearest

    def _determine_market_position(
        self, price: float, support_levels: List[Dict], resistance_levels: List[Dict]
    ) -> str:
        """Determine current market position"""
        try:
            if not support_levels or not resistance_levels:
                return "neutral"

            nearest_support = self._find_nearest_level(
                price, support_levels, is_support=True
            )
            nearest_resistance = self._find_nearest_level(
                price, resistance_levels, is_support=False
            )

            support_distance = (
                (price - nearest_support["price"]) / nearest_support["price"] * 100
            )
            resistance_distance = (
                (nearest_resistance["price"] - price)
                / nearest_resistance["price"]
                * 100
            )

            if support_distance < resistance_distance:
                return "near_support"
            else:
                return "near_resistance"
        except Exception as e:
            self.logger.error(f"Error determining market position: {e}")
            return "neutral"

    def _is_level_broken(
        self, data: pd.DataFrame, level_price: float, is_support: bool
    ) -> bool:
        """Check if a support/resistance level has been broken"""
        try:
            if is_support:
                # Support is broken if price moves significantly above
                return (data["high"] > level_price * 1.02).any()
            else:
                # Resistance is broken if price moves significantly below
                return (data["low"] < level_price * 0.98).any()
        except Exception as e:
            self.logger.error(f"Error checking level break: {e}")
            return False

    def _calculate_sr_zone(
        self, support: pd.Series, resistance: pd.Series
    ) -> pd.Series:
        """Calculate support/resistance zone width"""
        try:
            return (resistance - support) / support
        except Exception as e:
            self.logger.error(f"Error calculating SR zone: {e}")
            return pd.Series()
