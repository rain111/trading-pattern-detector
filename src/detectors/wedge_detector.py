import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class WedgePatternDetector(BaseDetector):
    """Wedge Pattern Detector (Rising Wedge, Falling Wedge)"""
    
    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect wedge patterns"""
        if not self.validate_data(data):
            return []
        
        data = self.preprocess_data(data)
        signals = []
        
        # Wedge detection logic
        for i in range(len(data) - 80):
            if self._is_potential_wedge_start(data, i):
                wedge_signals = self._analyze_wedge_pattern(data, i)
                signals.extend(wedge_signals)
        
        return self.validate_signals(signals)
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def _is_potential_wedge_start(self, data: pd.DataFrame, start_idx: int) -> bool:
        """Check if position could be start of wedge pattern"""
        try:
            if start_idx + 40 >= len(data):
                return False
            
            # Check for trend and volatility conditions
            trend_data = data.iloc[start_idx:start_idx + 40]
            
            # Calculate volatility
            volatility = trend_data['close'].pct_change().std()
            avg_volatility = data['close'].pct_change().iloc[:start_idx].std()
            
            # Check for expanding or contracting volatility
            volatility_expanding = volatility > avg_volatility * 1.1
            volatility_contracting = volatility < avg_volatility * 0.9
            
            # Check for trend presence
            price_change = (trend_data['close'].iloc[-1] - trend_data['close'].iloc[0]) / trend_data['close'].iloc[0]
            has_trend = abs(price_change) > 0.05
            
            return (volatility_expanding or volatility_contracting) and has_trend
        except Exception as e:
            self.logger.error(f"Error checking potential wedge start: {e}")
            return False
    
    def _analyze_wedge_pattern(self, data: pd.DataFrame, start_idx: int) -> List[PatternSignal]:
        """Analyze wedge pattern"""
        try:
            # Detect different wedge types
            rising_wedge = self._detect_rising_wedge(data, start_idx)
            falling_wedge = self._detect_falling_wedge(data, start_idx)
            
            signals = []
            
            if rising_wedge['is_wedge']:
                signals.append(self._generate_wedge_signal(data, start_idx, rising_wedge, PatternType.WEDGE_PATTERN))
            
            if falling_wedge['is_wedge']:
                signals.append(self._generate_wedge_signal(data, start_idx, falling_wedge, PatternType.WEDGE_PATTERN))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing wedge pattern: {e}")
            return []
    
    def _detect_rising_wedge(self, data: pd.DataFrame, start_idx: int) -> Dict[str, Any]:
        """Detect rising wedge pattern"""
        try:
            # Look for rising wedge structure
            wedge_length = min(60, len(data) - start_idx)
            wedge_data = data.iloc[start_idx:start_idx + wedge_length]
            
            # Find swing points
            swing_highs = self._find_swing_highs(wedge_data)
            swing_lows = self._find_swing_lows(wedge_data)
            
            if len(swing_highs) < 3 or len(swing_lows) < 3:
                return {'is_wedge': False}
            
            # Calculate trend lines
            resistance_slope = np.polyfit(range(len(swing_highs)), swing_highs['high'].values, 1)[0]
            support_slope = np.polyfit(range(len(swing_lows)), swing_lows['low'].values, 1)[0]
            
            # Check for rising wedge (both lines rising, resistance rising faster)
            rising_wedge = (resistance_slope > 0 and support_slope > 0 and 
                           resistance_slope > support_slope and 
                           abs(resistance_slope - support_slope) > 0.001)
            
            # Check for volume pattern (declining volume during formation)
            volume_pattern = self._check_wedge_volume_pattern(wedge_data)
            
            # Check for breakout
            breakout_info = self._check_wedge_breakout(data, start_idx, wedge_length, 
                                                     resistance_slope > support_slope)
            
            return {
                'is_wedge': rising_wedge and volume_pattern,
                'wedge_type': 'rising',
                'resistance_slope': resistance_slope,
                'support_slope': support_slope,
                'wedge_length': wedge_length,
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'volume_pattern': volume_pattern,
                'breakout_info': breakout_info
            }
        except Exception as e:
            self.logger.error(f"Error detecting rising wedge: {e}")
            return {'is_wedge': False}
    
    def _detect_falling_wedge(self, data: pd.DataFrame, start_idx: int) -> Dict[str, Any]:
        """Detect falling wedge pattern"""
        try:
            # Look for falling wedge structure
            wedge_length = min(60, len(data) - start_idx)
            wedge_data = data.iloc[start_idx:start_idx + wedge_length]
            
            # Find swing points
            swing_highs = self._find_swing_highs(wedge_data)
            swing_lows = self._find_swing_lows(wedge_data)
            
            if len(swing_highs) < 3 or len(swing_lows) < 3:
                return {'is_wedge': False}
            
            # Calculate trend lines
            resistance_slope = np.polyfit(range(len(swing_highs)), swing_highs['high'].values, 1)[0]
            support_slope = np.polyfit(range(len(swing_lows)), swing_lows['low'].values, 1)[0]
            
            # Check for falling wedge (both lines falling, support falling faster)
            falling_wedge = (resistance_slope < 0 and support_slope < 0 and 
                            support_slope < resistance_slope and 
                            abs(resistance_slope - support_slope) > 0.001)
            
            # Check for volume pattern (declining volume during formation)
            volume_pattern = self._check_wedge_volume_pattern(wedge_data)
            
            # Check for breakout
            breakout_info = self._check_wedge_breakout(data, start_idx, wedge_length, 
                                                     support_slope < resistance_slope)
            
            return {
                'is_wedge': falling_wedge and volume_pattern,
                'wedge_type': 'falling',
                'resistance_slope': resistance_slope,
                'support_slope': support_slope,
                'wedge_length': wedge_length,
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'volume_pattern': volume_pattern,
                'breakout_info': breakout_info
            }
        except Exception as e:
            self.logger.error(f"Error detecting falling wedge: {e}")
            return {'is_wedge': False}
    
    def _find_swing_highs(self, data: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """Find swing highs in data"""
        try:
            highs = data['high'].rolling(window=window, center=True).max()
            swing_highs = data[highs == data['high']]
            return swing_highs
        except Exception as e:
            self.logger.error(f"Error finding swing highs: {e}")
            return pd.DataFrame()
    
    def _find_swing_lows(self, data: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """Find swing lows in data"""
        try:
            lows = data['low'].rolling(window=window, center=True).min()
            swing_lows = data[lows == data['low']]
            return swing_lows
        except Exception as e:
            self.logger.error(f"Error finding swing lows: {e}")
            return pd.DataFrame()
    
    def _check_wedge_volume_pattern(self, wedge_data: pd.DataFrame) -> bool:
        """Check for typical volume pattern in wedges"""
        try:
            # Volume should decline during formation
            volume_trend = wedge_data['volume']
            
            # Check for volume decline
            first_half_volume = volume_trend.iloc[:len(volume_trend)//2].mean()
            second_half_volume = volume_trend.iloc[len(volume_trend)//2:].mean()
            
            volume_declining = second_half_volume < first_half_volume * 0.8
            
            return volume_declining
        except Exception as e:
            self.logger.error(f"Error checking wedge volume pattern: {e}")
            return False
    
    def _check_wedge_breakout(self, data: pd.DataFrame, start_idx: int, 
                            wedge_length: int, is_bullish_breakout: bool) -> Dict[str, Any]:
        """Check for wedge breakout"""
        try:
            breakout_start = start_idx + wedge_length
            breakout_end = breakout_start + 15
            
            if breakout_end >= len(data):
                return {'breakout_confirmed': False}
            
            breakout_data = data.iloc[breakout_start:breakout_end]
            
            # Calculate wedge boundaries
            wedge_data = data.iloc[start_idx:start_idx + wedge_length]
            
            if is_bullish_breakout:
                # Falling wedge bullish breakout above resistance
                resistance_highs = self._find_swing_highs(wedge_data)['high']
                if len(resistance_highs) > 0:
                    resistance_trend = np.polyfit(range(len(resistance_highs)), resistance_highs.values, 1)[0]
                    breakout_price = resistance_highs.iloc[-1] + resistance_trend * 5
                    breakout_confirmed = breakout_data['high'].max() > breakout_price
                else:
                    breakout_confirmed = False
                    breakout_price = 0
            else:
                # Rising wedge bearish breakout below support
                support_lows = self._find_swing_lows(wedge_data)['low']
                if len(support_lows) > 0:
                    support_trend = np.polyfit(range(len(support_lows)), support_lows.values, 1)[0]
                    breakout_price = support_lows.iloc[-1] + support_trend * 5
                    breakout_confirmed = breakout_data['low'].min() < breakout_price
                else:
                    breakout_confirmed = False
                    breakout_price = 0
            
            if breakout_confirmed:
                volume_spike = breakout_data['volume'].mean() > data['volume'].iloc[:breakout_start].mean() * 1.5
                
                return {
                    'breakout_confirmed': True,
                    'breakout_price': breakout_price,
                    'volume_spike': volume_spike,
                    'breakout_strength': abs(breakout_price - data['close'].iloc[breakout_start]) / data['close'].iloc[breakout_start]
                }
            else:
                return {'breakout_confirmed': False}
        except Exception as e:
            self.logger.error(f"Error checking wedge breakout: {e}")
            return {'breakout_confirmed': False}
    
    def _generate_wedge_signal(self, data: pd.DataFrame, start_idx: int,
                              wedge_info: dict, pattern_type: PatternType) -> PatternSignal:
        """Generate wedge trading signal"""
        try:
            current_price = data['close'].iloc[-1]
            breakout_info = wedge_info['breakout_info']
            
            if not breakout_info['breakout_confirmed']:
                return None
            
            breakout_price = breakout_info['breakout_price']
            is_bullish = wedge_info['wedge_type'] == 'falling'
            
            # Calculate risk parameters
            wedge_data = data.iloc[start_idx:start_idx + wedge_info['wedge_length']]
            
            if is_bullish:
                # Bullish falling wedge
                support_level = wedge_data['low'].min()
                resistance_level = wedge_data['high'].max()
                risk_distance = breakout_price - support_level
                target_distance = risk_distance * self.config.reward_ratio
                target_price = breakout_price + target_distance
                stop_loss = support_level * 0.98
            else:
                # Bearish rising wedge
                support_level = wedge_data['low'].min()
                resistance_level = wedge_data['high'].max()
                risk_distance = resistance_level - breakout_price
                target_distance = risk_distance * self.config.reward_ratio
                target_price = breakout_price - target_distance
                stop_loss = resistance_level * 1.02
            
            # Pattern metadata
            pattern_data = {
                'wedge_type': wedge_info['wedge_type'],
                'wedge_length': wedge_info['wedge_length'],
                'resistance_slope': wedge_info['resistance_slope'],
                'support_slope': wedge_info['support_slope'],
                'support_level': support_level,
                'resistance_level': resistance_level,
                'breakout_strength': breakout_info['breakout_strength'],
                'volume_spike': breakout_info['volume_spike'],
                'is_bullish': is_bullish,
                'pattern_formation': self._calculate_formation_quality(wedge_info, data, start_idx),
                'wedge_angle': abs(wedge_info['resistance_slope'] - wedge_info['support_slope'])
            }
            
            return PatternSignal(
                symbol="UNKNOWN",
                pattern_type=pattern_type,
                confidence=self._calculate_wedge_confidence(wedge_info, breakout_info),
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                timeframe=self.config.timeframe,
                timestamp=data.index[-1],
                metadata=pattern_data,
                signal_strength=self._calculate_signal_strength(pattern_data),
                risk_level=self._determine_risk_level(pattern_data),
                expected_duration="3-6 weeks",
                probability_target=0.50
            )
        except Exception as e:
            self.logger.error(f"Error generating wedge signal: {e}")
            return None
    
    def _calculate_formation_quality(self, wedge_info: dict, data: pd.DataFrame, start_idx: int) -> float:
        """Calculate wedge formation quality score"""
        try:
            quality = 0.5
            
            # Wedge length quality (optimal 40-60 periods)
            optimal_length = 50
            length_score = 1 - abs(wedge_info['wedge_length'] - optimal_length) / optimal_length
            quality += length_score * 0.2
            
            # Volume pattern quality
            if wedge_info['volume_pattern']:
                quality += 0.2
            
            # Angle quality (not too steep, not too flat)
            angle = abs(wedge_info['resistance_slope'] - wedge_info['support_slope'])
            angle_quality = 1 - min(angle * 100, 1.0)  # Convert to percentage
            quality += angle_quality * 0.2
            
            # Swing point quality
            swing_quality = min(len(wedge_info['swing_highs']) + len(wedge_info['swing_lows']) - 4, 1.0) * 0.1
            quality += swing_quality
            
            return min(quality, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating formation quality: {e}")
            return 0.5
    
    def _calculate_wedge_confidence(self, wedge_info: dict, breakout_info: dict) -> float:
        """Calculate wedge pattern confidence"""
        try:
            confidence = 0.5
            
            # Breakout strength
            confidence += min(breakout_info['breakout_strength'] * 3, 0.3)
            
            # Volume spike
            if breakout_info['volume_spike']:
                confidence += 0.2
            
            # Formation quality
            formation_quality = wedge_info.get('formation_quality', 0.5)
            confidence += formation_quality * 0.3
            
            # Wedge type confidence (some types are more reliable)
            if wedge_info['wedge_type'] == 'falling':
                confidence += 0.1
            
            return min(confidence, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating wedge confidence: {e}")
            return 0.5
    
    def _calculate_signal_strength(self, pattern_data: dict) -> float:
        """Calculate wedge signal strength"""
        try:
            strength = 0.5
            
            # Breakout strength
            strength += min(pattern_data['breakout_strength'] * 3, 0.3)
            
            # Volume spike
            if pattern_data['volume_spike']:
                strength += 0.2
            
            # Formation quality
            strength += pattern_data['pattern_formation'] * 0.3
            
            # Wedge angle
            angle_score = 1 - min(pattern_data['wedge_angle'] * 100, 1.0)
            strength += angle_score * 0.1
            
            return min(strength, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _determine_risk_level(self, pattern_data: dict) -> str:
        """Determine wedge pattern risk level"""
        try:
            risk_score = 0
            
            # Breakout strength risk
            breakout_strength = pattern_data['breakout_strength']
            risk_score += breakout_strength * 2
            
            # Wedge angle risk (steeper angles = higher risk)
            wedge_angle = pattern_data['wedge_angle']
            risk_score += wedge_angle * 50
            
            # Formation quality risk
            formation_quality = pattern_data['pattern_formation']
            risk_score += (1 - formation_quality) * 0.3
            
            # Wedge type risk
            if pattern_data['wedge_type'] == 'rising':
                risk_score += 0.2
            
            if risk_score > 0.6:
                return "high"
            elif risk_score > 0.3:
                return "medium"
            else:
                return "low"
        except Exception as e:
            self.logger.error(f"Error determining risk level: {e}")
            return "medium"