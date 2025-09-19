import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class TrianglePatternDetector(BaseDetector):
    """Triangle Pattern Detector (Ascending, Descending, Symmetrical)"""
    
    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect triangle patterns"""
        if not self.validate_data(data):
            return []
        
        data = self.preprocess_data(data)
        signals = []
        
        # Triangle detection logic
        for i in range(len(data) - 60):
            if self._is_potential_triangle_start(data, i):
                triangle_signals = self._analyze_triangle_pattern(data, i)
                signals.extend(triangle_signals)
        
        return self.validate_signals(signals)
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def _is_potential_triangle_start(self, data: pd.DataFrame, start_idx: int) -> bool:
        """Check if position could be start of triangle pattern"""
        try:
            if start_idx + 30 >= len(data):
                return False
            
            # Check for trend and volatility conditions
            trend_data = data.iloc[start_idx:start_idx + 30]
            
            # Calculate volatility
            volatility = trend_data['close'].pct_change().std()
            avg_volatility = data['close'].pct_change().iloc[:start_idx].std()
            
            # Check for contracting volatility
            volatility_contracting = volatility < avg_volatility * 0.9
            
            # Check for trend presence
            price_change = (trend_data['close'].iloc[-1] - trend_data['close'].iloc[0]) / trend_data['close'].iloc[0]
            has_trend = abs(price_change) > 0.05
            
            return volatility_contracting and has_trend
        except Exception as e:
            self.logger.error(f"Error checking potential triangle start: {e}")
            return False
    
    def _analyze_triangle_pattern(self, data: pd.DataFrame, start_idx: int) -> List[PatternSignal]:
        """Analyze triangle pattern"""
        try:
            # Detect different triangle types
            ascending_triangle = self._detect_ascending_triangle(data, start_idx)
            descending_triangle = self._detect_descending_triangle(data, start_idx)
            symmetrical_triangle = self._detect_symmetrical_triangle(data, start_idx)
            
            signals = []
            
            if ascending_triangle['is_triangle']:
                signals.append(self._generate_triangle_signal(data, start_idx, ascending_triangle, PatternType.ASCENDING_TRIANGLE))
            
            if descending_triangle['is_triangle']:
                signals.append(self._generate_triangle_signal(data, start_idx, descending_triangle, PatternType.ASCENDING_TRIANGLE))
            
            if symmetrical_triangle['is_triangle']:
                signals.append(self._generate_triangle_signal(data, start_idx, symmetrical_triangle, PatternType.ASCENDING_TRIANGLE))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing triangle pattern: {e}")
            return []
    
    def _detect_ascending_triangle(self, data: pd.DataFrame, start_idx: int) -> Dict[str, Any]:
        """Detect ascending triangle pattern"""
        try:
            # Look for ascending triangle structure
            triangle_length = min(40, len(data) - start_idx)
            triangle_data = data.iloc[start_idx:start_idx + triangle_length]
            
            # Find resistance level (horizontal)
            resistance = triangle_data['high'].max()
            
            # Find ascending support trend
            lows = triangle_data[triangle_data['low'] == triangle_data['low'].rolling(window=3, center=True).min()]
            
            if len(lows) < 2:
                return {'is_triangle': False}
            
            # Check if support is ascending
            support_slope = np.polyfit(range(len(lows)), lows['low'].values, 1)[0]
            
            # Check if resistance is relatively horizontal
            resistance_std = triangle_data[triangle_data['high'] >= resistance * 0.98]['high'].std()
            resistance_horizontal = resistance_std < resistance * 0.02
            
            # Check for breakout
            breakout_info = self._check_triangle_breakout(data, start_idx, triangle_length, 
                                                         resistance, support_slope > 0)
            
            return {
                'is_triangle': support_slope > 0 and resistance_horizontal,
                'triangle_type': 'ascending',
                'resistance': resistance,
                'support_slope': support_slope,
                'triangle_length': triangle_length,
                'breakout_info': breakout_info
            }
        except Exception as e:
            self.logger.error(f"Error detecting ascending triangle: {e}")
            return {'is_triangle': False}
    
    def _detect_descending_triangle(self, data: pd.DataFrame, start_idx: int) -> Dict[str, Any]:
        """Detect descending triangle pattern"""
        try:
            # Look for descending triangle structure
            triangle_length = min(40, len(data) - start_idx)
            triangle_data = data.iloc[start_idx:start_idx + triangle_length]
            
            # Find support level (horizontal)
            support = triangle_data['low'].min()
            
            # Find descending resistance trend
            highs = triangle_data[triangle_data['high'] == triangle_data['high'].rolling(window=3, center=True).max()]
            
            if len(highs) < 2:
                return {'is_triangle': False}
            
            # Check if resistance is descending
            resistance_slope = np.polyfit(range(len(highs)), highs['high'].values, 1)[0]
            
            # Check if support is relatively horizontal
            support_std = triangle_data[triangle_data['low'] <= support * 1.02]['low'].std()
            support_horizontal = support_std < support * 0.02
            
            # Check for breakout
            breakout_info = self._check_triangle_breakout(data, start_idx, triangle_length, 
                                                         support, resistance_slope < 0)
            
            return {
                'is_triangle': resistance_slope < 0 and support_horizontal,
                'triangle_type': 'descending',
                'support': support,
                'resistance_slope': resistance_slope,
                'triangle_length': triangle_length,
                'breakout_info': breakout_info
            }
        except Exception as e:
            self.logger.error(f"Error detecting descending triangle: {e}")
            return {'is_triangle': False}
    
    def _detect_symmetrical_triangle(self, data: pd.DataFrame, start_idx: int) -> Dict[str, Any]:
        """Detect symmetrical triangle pattern"""
        try:
            # Look for symmetrical triangle structure
            triangle_length = min(40, len(data) - start_idx)
            triangle_data = data.iloc[start_idx:start_idx + triangle_length]
            
            # Find trend lines
            highs = triangle_data[triangle_data['high'] == triangle_data['high'].rolling(window=3, center=True).max()]
            lows = triangle_data[triangle_data['low'] == triangle_data['low'].rolling(window=3, center=True).min()]
            
            if len(highs) < 2 or len(lows) < 2:
                return {'is_triangle': False}
            
            # Calculate trend lines
            resistance_slope = np.polyfit(range(len(highs)), highs['high'].values, 1)[0]
            support_slope = np.polyfit(range(len(lows)), lows['low'].values, 1)[0]
            
            # Check for converging trend lines
            converging = abs(resistance_slope - support_slope) < abs(support_slope) * 0.3
            
            # Check for volume pattern
            volume_trend = self._check_volume_triangle_pattern(triangle_data)
            
            # Check for breakout
            breakout_info = self._check_triangle_breakout(data, start_idx, triangle_length, 
                                                         None, support_slope > resistance_slope)
            
            return {
                'is_triangle': converging and volume_trend,
                'triangle_type': 'symmetrical',
                'resistance_slope': resistance_slope,
                'support_slope': support_slope,
                'triangle_length': triangle_length,
                'breakout_info': breakout_info
            }
        except Exception as e:
            self.logger.error(f"Error detecting symmetrical triangle: {e}")
            return {'is_triangle': False}
    
    def _check_triangle_breakout(self, data: pd.DataFrame, start_idx: int, 
                               triangle_length: int, level_price: float, 
                               is_bullish: bool) -> Dict[str, Any]:
        """Check for triangle breakout"""
        try:
            breakout_start = start_idx + triangle_length
            breakout_end = breakout_start + 10
            
            if breakout_end >= len(data):
                return {'breakout_confirmed': False}
            
            breakout_data = data.iloc[breakout_start:breakout_end]
            
            if is_bullish and level_price:
                # Bullish breakout above resistance
                breakout_confirmed = breakout_data['high'].max() > level_price * 1.01
                breakout_price = breakout_data[breakout_data['high'] > level_price * 1.01].iloc[0]['high']
            else:
                # Bearish breakout below support
                breakout_confirmed = breakout_data['low'].min() < level_price * 0.99
                breakout_price = breakout_data[breakout_data['low'] < level_price * 0.99].iloc[0]['low']
            
            if breakout_confirmed:
                volume_spike = breakout_data['volume'].mean() > data['volume'].iloc[:breakout_start].mean() * 1.5
                
                return {
                    'breakout_confirmed': True,
                    'breakout_price': breakout_price,
                    'volume_spike': volume_spike,
                    'breakout_strength': abs(breakout_price - level_price) / level_price if level_price else 0.02
                }
            else:
                return {'breakout_confirmed': False}
        except Exception as e:
            self.logger.error(f"Error checking triangle breakout: {e}")
            return {'breakout_confirmed': False}
    
    def _check_volume_triangle_pattern(self, triangle_data: pd.DataFrame) -> bool:
        """Check for typical volume pattern in triangles"""
        try:
            # Volume should decrease during formation and increase on breakout
            volume_trend = triangle_data['volume']
            
            # Check for volume decrease
            first_half_volume = volume_trend.iloc[:len(volume_trend)//2].mean()
            second_half_volume = volume_trend.iloc[len(volume_trend)//2:].mean()
            
            volume_decreasing = second_half_volume < first_half_volume * 0.9
            
            return volume_decreasing
        except Exception as e:
            self.logger.error(f"Error checking volume pattern: {e}")
            return False
    
    def _generate_triangle_signal(self, data: pd.DataFrame, start_idx: int,
                                triangle_info: dict, pattern_type: PatternType) -> PatternSignal:
        """Generate triangle trading signal"""
        try:
            current_price = data['close'].iloc[-1]
            breakout_info = triangle_info['breakout_info']
            
            if not breakout_info['breakout_confirmed']:
                return None
            
            breakout_price = breakout_info['breakout_price']
            is_bullish = triangle_info['triangle_type'] in ['ascending', 'symmetrical']
            
            # Calculate risk parameters
            if is_bullish:
                # Bullish triangle
                risk_distance = breakout_price - data['low'].iloc[start_idx:start_idx + triangle_info['triangle_length']].min()
                target_distance = risk_distance * self.config.reward_ratio
                target_price = breakout_price + target_distance
                stop_loss = data['low'].iloc[start_idx:start_idx + triangle_info['triangle_length']].min() * 0.98
            else:
                # Bearish triangle
                risk_distance = data['high'].iloc[start_idx:start_idx + triangle_info['triangle_length']].max() - breakout_price
                target_distance = risk_distance * self.config.reward_ratio
                target_price = breakout_price - target_distance
                stop_loss = data['high'].iloc[start_idx:start_idx + triangle_info['triangle_length']].max() * 1.02
            
            # Pattern metadata
            pattern_data = {
                'triangle_type': triangle_info['triangle_type'],
                'triangle_length': triangle_info['triangle_length'],
                'resistance_slope': triangle_info.get('resistance_slope', 0),
                'support_slope': triangle_info.get('support_slope', 0),
                'resistance': triangle_info.get('resistance', 0),
                'support': triangle_info.get('support', 0),
                'breakout_strength': breakout_info['breakout_strength'],
                'volume_spike': breakout_info['volume_spike'],
                'is_bullish': is_bullish,
                'pattern_formation': self._calculate_formation_quality(triangle_info, data, start_idx)
            }
            
            return PatternSignal(
                symbol="UNKNOWN",
                pattern_type=pattern_type,
                confidence=self._calculate_triangle_confidence(triangle_info, breakout_info),
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                timeframe=self.config.timeframe,
                timestamp=data.index[-1],
                metadata=pattern_data,
                signal_strength=self._calculate_signal_strength(pattern_data),
                risk_level=self._determine_risk_level(pattern_data),
                expected_duration="2-4 weeks",
                probability_target=0.55
            )
        except Exception as e:
            self.logger.error(f"Error generating triangle signal: {e}")
            return None
    
    def _calculate_formation_quality(self, triangle_info: dict, data: pd.DataFrame, start_idx: int) -> float:
        """Calculate triangle formation quality score"""
        try:
            quality = 0.5
            
            # Triangle length quality (optimal 20-40 periods)
            optimal_length = 30
            length_score = 1 - abs(triangle_info['triangle_length'] - optimal_length) / optimal_length
            quality += length_score * 0.2
            
            # Volume quality
            volume_score = self._check_volume_triangle_pattern(data.iloc[start_idx:start_idx + triangle_info['triangle_length']])
            if volume_score:
                quality += 0.2
            
            # Convergence quality
            if triangle_info['triangle_type'] == 'symmetrical':
                slope_diff = abs(triangle_info['resistance_slope'] - triangle_info['support_slope'])
                convergence_score = 1 - min(slope_diff, 0.1) / 0.1
                quality += convergence_score * 0.2
            
            return min(quality, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating formation quality: {e}")
            return 0.5
    
    def _calculate_triangle_confidence(self, triangle_info: dict, breakout_info: dict) -> float:
        """Calculate triangle pattern confidence"""
        try:
            confidence = 0.5
            
            # Breakout strength
            confidence += min(breakout_info['breakout_strength'] * 5, 0.3)
            
            # Volume spike
            if breakout_info['volume_spike']:
                confidence += 0.2
            
            # Formation quality
            formation_quality = triangle_info.get('formation_quality', 0.5)
            confidence += formation_quality * 0.3
            
            return min(confidence, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating triangle confidence: {e}")
            return 0.5
    
    def _calculate_signal_strength(self, pattern_data: dict) -> float:
        """Calculate triangle signal strength"""
        try:
            strength = 0.5
            
            # Breakout strength
            strength += min(pattern_data['breakout_strength'] * 5, 0.3)
            
            # Volume spike
            if pattern_data['volume_spike']:
                strength += 0.2
            
            # Formation quality
            strength += pattern_data['pattern_formation'] * 0.3
            
            return min(strength, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _determine_risk_level(self, pattern_data: dict) -> str:
        """Determine triangle pattern risk level"""
        try:
            risk_score = 0
            
            # Breakout strength risk
            breakout_strength = pattern_data['breakout_strength']
            risk_score += breakout_strength * 3
            
            # Triangle type risk (some patterns are riskier than others)
            if pattern_data['triangle_type'] == 'symmetrical':
                risk_score += 0.1
            elif pattern_data['triangle_type'] == 'descending':
                risk_score += 0.2
            
            # Formation quality risk
            formation_quality = pattern_data['pattern_formation']
            risk_score += (1 - formation_quality) * 0.3
            
            if risk_score > 0.6:
                return "high"
            elif risk_score > 0.3:
                return "medium"
            else:
                return "low"
        except Exception as e:
            self.logger.error(f"Error determining risk level: {e}")
            return "medium"