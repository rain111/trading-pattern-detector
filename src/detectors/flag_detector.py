import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.interfaces import BaseDetector, PatternConfig, PatternSignal, PatternType
import logging


class FlagPatternDetector(BaseDetector):
    """Flag Pattern Detector"""
    
    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect flag patterns"""
        if not self.validate_data(data):
            return []
        
        data = self.preprocess_data(data)
        signals = []
        
        # Flag detection logic
        for i in range(len(data) - 50):
            if self._is_potential_flag_start(data, i):
                flag_signals = self._analyze_flag_pattern(data, i)
                signals.extend(flag_signals)
        
        return self.validate_signals(signals)
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'volume']
    
    def _is_potential_flag_start(self, data: pd.DataFrame, start_idx: int) -> bool:
        """Check if position could be start of flag pattern"""
        try:
            if start_idx + 15 >= len(data):
                return False
            
            # Check for sharp initial move (flagpole)
            move_data = data.iloc[start_idx:start_idx + 15]
            move = (move_data['close'].iloc[-1] - move_data['close'].iloc[0]) / move_data['close'].iloc[0]
            
            # Volume during initial move
            move_volume = move_data['volume'].mean()
            avg_volume = data['volume'].iloc[:start_idx].mean()
            volume_sufficient = move_volume > avg_volume * 1.2
            
            return abs(move) > 0.08 and volume_sufficient  # 8% move with volume
        except Exception as e:
            self.logger.error(f"Error checking potential flag start: {e}")
            return False
    
    def _analyze_flag_pattern(self, data: pd.DataFrame, start_idx: int) -> List[PatternSignal]:
        """Analyze flag pattern"""
        try:
            # Flag detection logic
            flag_info = self._detect_flag_structure(data, start_idx)
            
            if flag_info['is_flag']:
                return [self._generate_flag_signal(data, start_idx, flag_info)]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error analyzing flag pattern: {e}")
            return []
    
    def _detect_flag_structure(self, data: pd.DataFrame, start_idx: int) -> Dict[str, Any]:
        """Detect flag structure"""
        try:
            # Flagpole detection
            flagpole_end = start_idx + 15
            if flagpole_end >= len(data):
                return {'is_flag': False}
            
            flagpole_data = data.iloc[start_idx:flagpole_end]
            flagpole_height = abs((flagpole_data['close'].iloc[-1] - flagpole_data['close'].iloc[0]) / flagpole_data['close'].iloc[0])
            
            # Flag detection (consolidation phase)
            flag_start = flagpole_end
            flag_end = flag_start + 20  # Maximum flag duration
            
            if flag_end >= len(data):
                return {'is_flag': False}
            
            flag_data = data.iloc[flag_start:flag_end]
            
            # Check for consolidation (flag body)
            flag_range = flag_data['high'].max() - flag_data['low'].min()
            avg_flag_price = flag_data['close'].mean()
            range_ratio = flag_range / avg_flag_price
            
            # Check for parallel boundaries (flag shape)
            upper_boundary = flag_data['high'].max()
            lower_boundary = flag_data['low'].min()
            
            # Check for volume decrease during flag
            flag_volume = flag_data['volume'].mean()
            flagpole_volume = flagpole_data['volume'].mean()
            volume_decrease = flag_volume < flagpole_volume * 0.8
            
            # Check for volume spike at potential breakout
            breakout_volume = self._check_breakout_volume(data, flag_end)
            
            # Check flag direction
            overall_direction = "bullish" if flagpole_data['close'].iloc[-1] > flagpole_data['close'].iloc[0] else "bearish"
            
            return {
                'is_flag': range_ratio < 0.04 and volume_decrease,  # Valid flag pattern
                'flagpole_height': flagpole_height,
                'flag_duration': len(flag_data),
                'flag_range': flag_range,
                'range_ratio': range_ratio,
                'upper_boundary': upper_boundary,
                'lower_boundary': lower_boundary,
                'volume_decrease': volume_decrease,
                'breakout_volume': breakout_volume,
                'overall_direction': overall_direction,
                'start_idx': start_idx,
                'flagpole_end': flagpole_end,
                'flag_end': flag_end
            }
        except Exception as e:
            self.logger.error(f"Error detecting flag structure: {e}")
            return {'is_flag': False}
    
    def _check_breakout_volume(self, data: pd.DataFrame, flag_end: int) -> bool:
        """Check for volume spike at potential breakout"""
        try:
            if flag_end + 5 >= len(data):
                return False
            
            # Check volume after flag
            breakout_volume = data['volume'].iloc[flag_end:flag_end + 5].mean()
            avg_flag_volume = data['volume'].iloc[flag_end - 15:flag_end].mean()
            
            return breakout_volume > avg_flag_volume * 1.5
        except Exception as e:
            self.logger.error(f"Error checking breakout volume: {e}")
            return False
    
    def _generate_flag_signal(self, data: pd.DataFrame, start_idx: int,
                            flag_info: dict) -> PatternSignal:
        """Generate flag trading signal"""
        try:
            current_price = data['close'].iloc[-1]
            flag_end = flag_info['flag_end']
            
            # Determine breakout direction
            if flag_info['overall_direction'] == 'bullish':
                # Bullish flag: breakout above upper boundary
                breakout_price = flag_info['upper_boundary']
                target_distance = breakout_price - flag_info['lower_boundary']
                target_price = breakout_price + target_distance * self.config.reward_ratio
                stop_loss = flag_info['lower_boundary'] * 0.98
            else:
                # Bearish flag: breakout below lower boundary
                breakout_price = flag_info['lower_boundary']
                target_distance = flag_info['upper_boundary'] - breakout_price
                target_price = breakout_price - target_distance * self.config.reward_ratio
                stop_loss = flag_info['upper_boundary'] * 1.02
            
            # Calculate confidence
            confidence = self._calculate_flag_confidence(flag_info)
            
            # Pattern metadata
            pattern_data = {
                'flagpole_height': flag_info['flagpole_height'],
                'flag_duration': flag_info['flag_duration'],
                'flag_range': flag_info['flag_range'],
                'range_ratio': flag_info['range_ratio'],
                'upper_boundary': flag_info['upper_boundary'],
                'lower_boundary': flag_info['lower_boundary'],
                'volume_decrease': flag_info['volume_decrease'],
                'breakout_volume': flag_info['breakout_volume'],
                'overall_direction': flag_info['overall_direction'],
                'flagpole_volume': data['volume'].iloc[start_idx:start_idx + 15].mean(),
                'flag_volume': data['volume'].iloc[flag_end - 15:flag_end].mean(),
                'breakout_strength': abs(current_price - breakout_price) / breakout_price
            }
            
            return PatternSignal(
                symbol="UNKNOWN",
                pattern_type=PatternType.FLAG_PATTERN,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                timeframe=self.config.timeframe,
                timestamp=data.index[-1],
                metadata=pattern_data,
                signal_strength=self._calculate_signal_strength(pattern_data),
                risk_level=self._determine_risk_level(pattern_data),
                expected_duration="1-2 weeks",
                probability_target=0.60
            )
        except Exception as e:
            self.logger.error(f"Error generating flag signal: {e}")
            raise
    
    def _calculate_flag_confidence(self, flag_info: dict) -> float:
        """Calculate flag pattern confidence"""
        try:
            confidence = 0.5
            
            # Flag strength (height of flagpole)
            confidence += min(flag_info['flagpole_height'] * 2, 0.2)
            
            # Volume decrease during flag
            if flag_info['volume_decrease']:
                confidence += 0.2
            
            # Range ratio (tighter flag is better)
            confidence += min((0.04 - flag_info['range_ratio']) * 5, 0.2)
            
            # Breakout volume
            if flag_info['breakout_volume']:
                confidence += 0.1
            
            return min(confidence, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating flag confidence: {e}")
            return 0.5
    
    def _calculate_signal_strength(self, pattern_data: dict) -> float:
        """Calculate flag signal strength"""
        try:
            strength = 0.5
            
            # Flagpole strength
            strength += min(pattern_data['flagpole_height'] * 2, 0.3)
            
            # Volume pattern strength
            volume_ratio = pattern_data['flag_volume'] / pattern_data.get('flagpole_volume', 1)
            if volume_ratio < 0.8:  # Volume decrease
                strength += 0.2
            
            # Breakout strength
            strength += min(pattern_data['breakout_strength'] * 3, 0.2)
            
            # Pattern completeness
            duration_score = min(pattern_data['flag_duration'] / 10, 1.0)  # Optimal duration 10 periods
            strength += duration_score * 0.1
            
            return min(strength, 1.0)
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _determine_risk_level(self, pattern_data: dict) -> str:
        """Determine flag pattern risk level"""
        try:
            risk_score = 0
            
            # Flag range ratio risk
            range_ratio = pattern_data['range_ratio']
            risk_score += range_ratio * 10
            
            # Breakout strength risk
            breakout_strength = pattern_data['breakout_strength']
            risk_score += breakout_strength * 2
            
            # Volume decrease risk (insufficient volume decrease = higher risk)
            if not pattern_data['volume_decrease']:
                risk_score += 0.2
            
            # Flag duration risk (shorter duration = higher risk)
            if pattern_data['flag_duration'] < 5:
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