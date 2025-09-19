# Modular Architecture Design for Trading Pattern Detection

## Modular Design Philosophy

This document outlines the modular architecture for the trading pattern detection system, emphasizing clean separation of concerns, extensibility, and maintainability.

## Implementation Status

**✅ COMPLETED** - All core components have been implemented with comprehensive logging and error handling:
- Core framework interfaces with enhanced error handling
- All pattern detectors (VCP, Flag, Triangle, Wedge)
- Analysis components (Volatility, Volume, Trend, Support/Resistance)
- Data utilities and configuration management
- Plugin system for extensibility
- Comprehensive test suite
- **NEW**: Advanced logging and error handling system

## Key Implementation Features

- **Comprehensive Logging**: Configurable logging with file and console output
- **Error Handling**: Robust error handling throughout all components
- **Test Coverage**: Complete test suite with fixtures and mocking
- **Plugin Architecture**: Dynamic detector loading and management
- **Configuration Management**: YAML-based configuration with validation

## Core Architectural Principles

### 1. **Layered Architecture**
```
┌─────────────────────────────────────────┐
│           Application Layer              │
│  - Detection Engine                     │
│  - Signal Aggregator                    │
│  - Portfolio Manager                     │
├─────────────────────────────────────────┤
│           Pattern Layer                   │
│  - VCP Detector                          │
│  - Flag Detector                        │
│  - Triangle Detector                     │
│  - Wedge Detector                        │
├─────────────────────────────────────────┤
│           Analysis Layer                 │
│  - Volatility Analyzer                   │
│  - Volume Analyzer                       │
│  - Trend Analyzer                        │
│  - Support/Resistance                    │
├─────────────────────────────────────────┤
│           Data Layer                     │
│  - Data Validator                        │
│  - Data Preprocessor                     │
│  - Market Data Client                    │
└─────────────────────────────────────────┘
```

### 2. **Dependency Inversion**
- High-level modules don't depend on low-level modules
- Both depend on abstractions
- Abstractions don't depend on details

### 3. **Single Responsibility Principle**
- Each module has one responsibility
- Clear separation between concerns
- Easy to test and maintain

## Module Architecture

### 1. **Core Module (`src/core/`)**
```python
# Enhanced interfaces with improved functionality
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from datetime import datetime
import logging

@dataclass
class PatternConfig:
    """Base configuration for pattern detection"""
    min_confidence: float = 0.6
    max_lookback: int = 100
    timeframe: str = "1d"
    volume_threshold: float = 1000000.0
    volatility_threshold: float = 0.001
    reward_ratio: float = 2.0
    
@dataclass 
class PatternSignal:
    """Enhanced trading signal with comprehensive metadata"""
    symbol: str
    pattern_type: PatternType
    confidence: float
    entry_price: float
    stop_loss: float
    target_price: float
    timeframe: str
    timestamp: pd.Timestamp
    metadata: Dict[str, Any]
    signal_strength: float = 0.0
    risk_level: str = "medium"
    expected_duration: Optional[str] = None
    probability_target: Optional[float] = None
    
class EnhancedPatternDetector(ABC):
    """Enhanced base class with common functionality"""
    
    def __init__(self, config: PatternConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_components()
        
    def _setup_components(self):
        """Setup analysis components"""
        from ..analysis.volatility_analyzer import VolatilityAnalyzer
        from ..analysis.volume_analyzer import VolumeAnalyzer
        from ..analysis.trend_analyzer import TrendAnalyzer
        
        self.volatility_analyzer = VolatilityAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
    
    @abstractmethod
    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Main pattern detection method"""
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return required columns for the detector"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Enhanced data validation"""
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check data quality
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                if data[col].isna().any():
                    self.logger.error(f"Column '{col}' contains NaN values")
                    return False
        
        # Check price consistency
        if 'high' in data.columns and 'low' in data.columns:
            if (data['high'] < data['low']).any():
                self.logger.error("High prices cannot be lower than low prices")
                return False
        
        return len(data) >= self.config.max_lookback
    
    def calculate_confidence(self, pattern_data: dict) -> float:
        """Calculate pattern confidence score"""
        confidence = 0.5  # Base confidence
        
        # Volume confidence
        if 'volume_ratio' in pattern_data:
            confidence += min(pattern_data['volume_ratio'] * 0.1, 0.2)
        
        # Volatility confidence
        if 'volatility_score' in pattern_data:
            confidence += min(abs(pattern_data['volatility_score']) * 0.2, 0.2)
        
        # Trend confidence
        if 'trend_strength' in pattern_data:
            confidence += min(pattern_data['trend_strength'] * 0.1, 0.1)
        
        return min(confidence, 1.0)
    
    def generate_signals(self, patterns: List[dict]) -> List[PatternSignal]:
        """Convert pattern detections to trading signals"""
        signals = []
        
        for pattern in patterns:
            try:
                confidence = self.calculate_confidence(pattern)
                
                if confidence >= self.config.min_confidence:
                    signal = PatternSignal(
                        symbol=pattern['symbol'],
                        pattern_type=pattern['pattern_type'],
                        confidence=confidence,
                        entry_price=pattern['entry_price'],
                        stop_loss=pattern['stop_loss'],
                        target_price=pattern['target_price'],
                        timeframe=self.config.timeframe,
                        timestamp=pattern['timestamp'],
                        metadata=pattern.get('metadata', {}),
                        signal_strength=self._calculate_signal_strength(pattern),
                        risk_level=self._determine_risk_level(pattern)
                    )
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error generating signal: {e}")
                continue
        
        return signals
    
    def _calculate_signal_strength(self, pattern: dict) -> float:
        """Calculate overall signal strength"""
        strength = pattern.get('confidence', 0.5)
        volume_multiplier = pattern.get('volume_ratio', 1.0)
        strength *= volume_multiplier
        
        return min(strength, 1.0)
    
    def _determine_risk_level(self, pattern: dict) -> str:
        """Determine risk level based on pattern characteristics"""
        volatility = pattern.get('volatility_score', 0)
        
        if volatility > 0.05:
            return "high"
        elif volatility > 0.02:
            return "medium"
        else:
            return "low"
```

### 2. **Analysis Module (`src/analysis/`)**
```python
# volatility_analyzer.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from scipy import stats

class VolatilityAnalyzer:
    """Advanced volatility analysis for pattern detection"""
    
    def __init__(self, atr_period: int = 14, bb_period: int = 20):
        self.atr_period = atr_period
        self.bb_period = bb_period
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high, low, close = data['high'], data['low'], data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def calculate_volatility_contraction(self, data: pd.DataFrame, 
                                       period: int = 20) -> Dict[str, float]:
        """Detect volatility contraction patterns"""
        atr_series = self.calculate_atr(data)
        
        # Calculate ATR trend
        atr_trend = np.polyfit(range(len(atr_series[-period:])), 
                              atr_series[-period:].values, 1)[0]
        
        # Calculate volatility contraction ratio
        recent_atr = atr_series.iloc[-1]
        historical_atr = atr_series.mean()
        contraction_ratio = recent_atr / historical_atr if historical_atr > 0 else 1.0
        
        return {
            'atr_trend': atr_trend,
            'contraction_ratio': contraction_ratio,
            'volatility_score': abs(atr_trend),
            'is_contracting': atr_trend < 0 and contraction_ratio < 0.8
        }
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, 
                                period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands for volatility analysis"""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'sma': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'band_width': (upper_band - lower_band) / sma
        }

# volume_analyzer.py
class VolumeAnalyzer:
    """Volume pattern analysis for confirmation signals"""
    
    def __init__(self, volume_window: int = 20):
        self.volume_window = volume_window
    
    def analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns in the data"""
        avg_volume = data['volume'].rolling(window=self.volume_window).mean()
        volume_ratio = data['volume'] / avg_volume
        
        # Volume trends
        volume_trend = np.polyfit(range(len(data)), data['volume'].values, 1)[0]
        recent_volume_trend = np.polyfit(range(min(10, len(data))), 
                                       data['volume'].tail(10).values, 1)[0]
        
        # Volume spikes
        volume_spikes = volume_ratio > 2.0
        volume_spike_count = volume_spikes.sum()
        
        return {
            'volume_ratio': volume_ratio.iloc[-1] if len(volume_ratio) > 0 else 1.0,
            'volume_trend': volume_trend,
            'recent_volume_trend': recent_volume_trend,
            'volume_spike_count': volume_spike_count,
            'volume_spike_ratio': volume_spike_count / len(data),
            'volume_spike_active': volume_spikes.iloc[-1] if len(volume_spikes) > 0 else False
        }
    
    def confirm_volume_breakout(self, data: pd.DataFrame, 
                               breakout_index: int) -> bool:
        """Confirm breakout with volume analysis"""
        if breakout_index >= len(data):
            return False
        
        breakout_volume = data['volume'].iloc[breakout_index]
        avg_volume = data['volume'].iloc[:breakout_index].mean()
        
        return breakout_volume > avg_volume * 1.5

# trend_analyzer.py
class TrendAnalyzer:
    """Trend detection and analysis"""
    
    def __init__(self, trend_period: int = 50):
        self.trend_period = trend_period
    
    def detect_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze price trends"""
        close_prices = data['close']
        
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
        ma_convergence = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        
        return {
            'slope': slope,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'r_squared': r_value ** 2,
            'ma_convergence': ma_convergence,
            'is_strong_trend': trend_strength > 0.7
        }
    
    def find_swings(self, data: pd.DataFrame) -> List[Dict]:
        """Find price swings for pattern detection"""
        highs = data['high']
        lows = data['low']
        
        swing_highs = []
        swing_lows = []
        
        # Simple swing detection logic
        for i in range(2, len(highs) - 2):
            # Swing high
            if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.iloc[i-2] and highs.iloc[i] > highs.iloc[i+2]):
                swing_highs.append({
                    'index': i,
                    'price': highs.iloc[i],
                    'timestamp': data.index[i]
                })
            
            # Swing low
            if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.iloc[i-2] and lows.iloc[i] < lows.iloc[i+2]):
                swing_lows.append({
                    'index': i,
                    'price': lows.iloc[i],
                    'timestamp': data.index[i]
                })
        
        return {'swing_highs': swing_highs, 'swing_lows': swing_lows}

# support_resistance.py
class SupportResistanceDetector:
    """Support and resistance level detection"""
    
    def find_support(self, data: pd.DataFrame, tolerance: float = 0.02) -> float:
        """Find support levels"""
        lows = data['low']
        
        # Find local minima
        support_levels = []
        
        for i in range(1, len(lows) - 1):
            if (lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.mean() * (1 - tolerance)):
                support_levels.append(lows.iloc[i])
        
        return min(support_levels) if support_levels else lows.min()
    
    def find_resistance(self, data: pd.DataFrame, tolerance: float = 0.02) -> float:
        """Find resistance levels"""
        highs = data['high']
        
        # Find local maxima
        resistance_levels = []
        
        for i in range(1, len(highs) - 1):
            if (highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.mean() * (1 + tolerance)):
                resistance_levels.append(highs.iloc[i])
        
        return max(resistance_levels) if resistance_levels else highs.max()
    
    def draw_trendlines(self, data: pd.DataFrame, swing_points: List[Dict]) -> List[Dict]:
        """Draw trendlines from swing points"""
        trendlines = []
        
        # Simplified trendline drawing logic
        for i in range(len(swing_points)):
            for j in range(i + 1, min(i + 5, len(swing_points))):
                # Calculate trendline between two points
                point1 = swing_points[i]
                point2 = swing_points[j]
                
                # Check if trendline has appropriate slope
                slope = (point2['price'] - point1['price']) / (point2['index'] - point1['index'])
                
                if abs(slope) < 0.1:  # Nearly horizontal
                    trendlines.append({
                        'start_point': point1,
                        'end_point': point2,
                        'slope': slope,
                        'type': 'horizontal'
                    })
                elif slope > 0:  # Upward sloping
                    trendlines.append({
                        'start_point': point1,
                        'end_point': point2,
                        'slope': slope,
                        'type': 'ascending'
                    })
                else:  # Downward sloping
                    trendlines.append({
                        'start_point': point1,
                        'end_point': point2,
                        'slope': slope,
                        'type': 'descending'
                    })
        
        return trendlines
```

### 3. **Detectors Module (`src/detectors/`)**
```python
# base_detector.py
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd
from ..core.interfaces import EnhancedPatternDetector, PatternSignal, PatternConfig

class BaseDetector(EnhancedPatternDetector):
    """Base detector with common functionality"""
    
    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def detect_pattern(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Main pattern detection method"""
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for pattern detection"""
        # Remove NaN values
        data = data.dropna()
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Calculate additional indicators if needed
        if 'returns' not in data.columns:
            data['returns'] = data['close'].pct_change()
        
        return data
    
    def validate_signals(self, signals: List[PatternSignal]) -> List[PatternSignal]:
        """Validate and filter signals"""
        valid_signals = []
        
        for signal in signals:
            # Check confidence threshold
            if signal.confidence < self.config.min_confidence:
                continue
            
            # Check price validity
            if signal.entry_price <= 0 or signal.stop_loss <= 0 or signal.target_price <= 0:
                continue
            
            # Check logical relationships
            if signal.stop_loss == signal.entry_price or signal.target_price == signal.entry_price:
                continue
            
            valid_signals.append(signal)
        
        return valid_signals

# vcp_detector.py
class VCPBreakoutDetector(BaseDetector):
    """VCP (Volatility Contraction Pattern) Breakout Detector"""
    
    def __init__(self, config: PatternConfig):
        super().__init__(config)
        self.stage_analyzer = StageAnalyzer()
    
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
        return ['open', 'high', 'low', 'close', 'volume']
    
    def _is_potential_vcp_start(self, data: pd.DataFrame, start_idx: int) -> bool:
        """Check if position could be start of VCP pattern"""
        if start_idx + 20 >= len(data):
            return False
        
        # Check for initial decline
        decline_data = data.iloc[start_idx:start_idx + 20]
        decline = (decline_data['close'].iloc[-1] - decline_data['close'].iloc[0]) / decline_data['close'].iloc[0]
        
        return decline < -0.10  # 10% decline
    
    def _analyze_vcp_pattern(self, data: pd.DataFrame, start_idx: int) -> List[PatternSignal]:
        """Complete VCP pattern analysis"""
        try:
            # Stage 1: Initial decline (already confirmed)
            
            # Stage 2: Volatility contraction
            if not self.stage_analyzer.detect_volatility_contraction(data, start_idx):
                return []
            
            # Stage 3: Consolidation
            consolidation_info = self.stage_analyzer.detect_consolidation(data, start_idx)
            
            if not consolidation_info['is_consolidation']:
                return []
            
            # Stage 4: Breakout
            breakout_info = self.stage_analyzer.detect_breakout(data, consolidation_info)
            
            if not breakout_info['breakout_confirmed']:
                return []
            
            # Generate signal
            return [self._generate_vcp_signal(data, start_idx, consolidation_info, breakout_info)]
            
        except Exception as e:
            self.logger.error(f"Error analyzing VCP pattern: {e}")
            return []
    
    def _generate_vcp_signal(self, data: pd.DataFrame, start_idx: int,
                           consolidation_info: dict, breakout_info: dict) -> PatternSignal:
        """Generate VCP trading signal"""
        current_price = data['close'].iloc[-1]
        breakout_price = breakout_info['breakout_price']
        
        # Calculate risk parameters
        risk_distance = breakout_price - current_price
        target_price = current_price - (risk_distance * self.config.reward_ratio)
        
        return PatternSignal(
            symbol="UNKNOWN",  # Will be set by caller
            pattern_type=PatternType.VCP_BREAKOUT,
            confidence=breakout_info['confidence'],
            entry_price=current_price,
            stop_loss=breakout_price,
            target_price=target_price,
            timeframe=self.config.timeframe,
            timestamp=data.index[-1],
            metadata={
                'breakout_strength': breakout_info['breakout_strength'],
                'consolidation_range': consolidation_info['price_range'],
                'volume_spike': breakout_info['volume_spike'],
                'support_level': consolidation_info['support'],
                'resistance_level': consolidation_info['resistance']
            }
        )

# flag_detector.py
class FlagPatternDetector(BaseDetector):
    """Flag Pattern Detector"""
    
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
        if start_idx + 15 >= len(data):
            return False
        
        # Check for sharp initial move
        move_data = data.iloc[start_idx:start_idx + 15]
        move = (move_data['close'].iloc[-1] - move_data['close'].iloc[0]) / move_data['close'].iloc[0]
        
        return abs(move) > 0.08  # 8% move
    
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
    
    def _detect_flag_structure(self, data: pd.DataFrame, start_idx: int) -> dict:
        """Detect flag structure"""
        # Implementation would analyze flagpole and flag structure
        return {
            'is_flag': False,  # Placeholder
            'flagpole_height': 0.0,
            'flag_duration': 0,
            'flag_direction': 'neutral'
        }
    
    def _generate_flag_signal(self, data: pd.DataFrame, start_idx: int,
                            flag_info: dict) -> PatternSignal:
        """Generate flag trading signal"""
        current_price = data['close'].iloc[-1]
        
        return PatternSignal(
            symbol="UNKNOWN",
            pattern_type=PatternType.FLAG_PATTERN,
            confidence=0.7,
            entry_price=current_price,
            stop_loss=current_price * 1.02,  # 2% stop
            target_price=current_price * 1.05,  # 5% target
            timeframe=self.config.timeframe,
            timestamp=data.index[-1],
            metadata=flag_info
        )
```

### 4. **Utilities Module (`src/utils/`)**
```python
# data_preprocessor.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

class DataPreprocessor:
    """Data preprocessing utilities for pattern detection"""
    
    def __init__(self):
        self.price_columns = ['open', 'high', 'low', 'close', 'volume']
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate market data"""
        # Remove duplicate timestamps
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by timestamp
        data = data.sort_index()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Remove outliers
        data = self._remove_outliers(data)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in market data"""
        # For price data, forward fill then interpolate
        price_data = data[['open', 'high', 'low', 'close']]
        price_data = price_data.fillna(method='ffill').interpolate()
        
        # For volume, fill with median
        volume_data = data['volume'].fillna(data['volume'].median())
        
        # Other columns forward fill
        other_columns = data.drop(columns=['open', 'high', 'low', 'close', 'volume'])
        other_columns = other_columns.fillna(method='ffill')
        
        return pd.concat([price_data, volume_data, other_columns], axis=1)
    
    def _remove_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers from market data"""
        cleaned_data = data.copy()
        
        for col in ['open', 'high', 'low', 'close']:
            if col in cleaned_data.columns:
                z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                cleaned_data = cleaned_data[z_scores < threshold]
        
        return cleaned_data
    
    def resample_data(self, data: pd.DataFrame, timeframe: str = '1D') -> pd.DataFrame:
        """Resample data to different timeframe"""
        # OHLCV aggregation functions
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Handle other columns
        for col in data.columns:
            if col not in agg_dict and col != 'volume':
                agg_dict[col] = 'last'
        
        return data.resample(timeframe).agg(agg_dict).dropna()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data"""
        data = data.copy()
        
        # Moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # RSI
        data['rsi'] = self._calculate_rsi(data['close'], period=14)
        
        # MACD
        macd_data = self._calculate_macd(data['close'])
        data = pd.concat([data, macd_data], axis=1)
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        })

# signal_aggregator.py
class SignalAggregator:
    """Aggregate and rank signals from multiple detectors"""
    
    def __init__(self):
        self.signals = []
    
    def add_signal(self, signal: PatternSignal):
        """Add a signal to the aggregator"""
        self.signals.append(signal)
    
    def add_signals(self, signals: List[PatternSignal]):
        """Add multiple signals"""
        self.signals.extend(signals)
    
    def rank_signals(self, ranking_method: str = 'confidence') -> List[PatternSignal]:
        """Rank signals by specified method"""
        if not self.signals:
            return []
        
        if ranking_method == 'confidence':
            return sorted(self.signals, key=lambda x: x.confidence, reverse=True)
        elif ranking_method == 'risk_reward':
            return sorted(self.signals, key=lambda x: x.target_price / x.stop_loss, reverse=True)
        elif ranking_method == 'strength':
            return sorted(self.signals, key=lambda x: x.signal_strength, reverse=True)
        else:
            return sorted(self.signals, key=lambda x: x.confidence, reverse=True)
    
    def filter_signals(self, min_confidence: float = 0.6, 
                      min_volume: Optional[float] = None,
                      max_risk: Optional[str] = None) -> List[PatternSignal]:
        """Filter signals by criteria"""
        filtered_signals = []
        
        for signal in self.signals:
            # Confidence filter
            if signal.confidence < min_confidence:
                continue
            
            # Volume filter (if volume metadata available)
            if min_volume is not None:
                volume = signal.metadata.get('volume_spike', 0)
                if volume < min_volume:
                    continue
            
            # Risk filter
            if max_risk is not None:
                risk_level = signal.risk_level
                if (max_risk == 'low' and risk_level != 'low') or \
                   (max_risk == 'medium' and risk_level == 'high'):
                    continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def get_signals_by_symbol(self, symbol: str) -> List[PatternSignal]:
        """Get signals for a specific symbol"""
        return [signal for signal in self.signals if signal.symbol == symbol]
    
    def get_signals_by_pattern_type(self, pattern_type: PatternType) -> List[PatternSignal]:
        """Get signals by pattern type"""
        return [signal for signal in self.signals if signal.pattern_type == pattern_type]
    
    def clear_signals(self):
        """Clear all signals"""
        self.signals.clear()

# market_data_client.py
class MarketDataClient:
    """Market data fetching and processing"""
    
    def __init__(self, data_source: str = 'yahoo'):
        self.data_source = data_source
        self.preprocessor = DataPreprocessor()
    
    def fetch_data(self, symbol: str, period: str = '1y', 
                  timeframe: str = '1d') -> pd.DataFrame:
        """Fetch market data for a symbol"""
        # This would connect to actual market data API
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: List[str], 
                             period: str = '1y',
                             timeframe: str = '1d') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        data_dict = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_data(symbol, period, timeframe)
                if not data.empty:
                    data_dict[symbol] = self.preprocessor.clean_data(data)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return data_dict
    
    def get_universe_data(self, universe_file: str) -> Dict[str, pd.DataFrame]:
        """Get data for universe of symbols"""
        # Read universe from file
        with open(universe_file, 'r') as f:
            symbols = [line.strip() for line in f.readlines() if line.strip()]
        
        return self.fetch_multiple_symbols(symbols)
```

## Configuration System

### Configuration Files

```yaml
# src/config/pattern_parameters.yaml
pattern_parameters:
  vcp_detector:
    initial_decline_period: 20
    max_decline_threshold: -0.15
    volume_threshold: 1000000
    contraction_period: 30
    volatility_threshold: 0.001
    consolidation_period: 25
    max_consolidation_range: 0.05
    min_boundary_touches: 3
    breakout_period: 10
    volume_spike_threshold: 2.0
    reward_ratio: 2.0
    confidence_threshold: 0.7
  
  flag_detector:
    flagpole_min_length: 0.08
    flag_max_duration: 20
    flag_min_duration: 5
    volume_threshold: 1.5
    reward_ratio: 1.5
    confidence_threshold: 0.6

# src/config/market_data_config.yaml
market_data:
  default_source: 'yahoo'
  timeframes: ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
  default_period: '1y'
  cache_enabled: true
  cache_duration: 3600  # 1 hour

# src/config/detection_settings.yaml
detection_settings:
  parallel_processing: true
  max_workers: 4
  batch_size: 100
  cache_results: true
  confidence_threshold: 0.6
  ranking_method: 'confidence'
  output_format: 'json'
  logging_level: 'INFO'
```

## Plugin Architecture

### Plugin Registration System

```python
# plugin_manager.py
from typing import Dict, List, Type
from abc import ABC, abstractmethod

class PluginRegistry:
    """Registry for pattern detection plugins"""
    
    def __init__(self):
        self.detectors: Dict[str, Type[EnhancedPatternDetector]] = {}
        self.analyzers: Dict[str, Type] = {}
        self.utilities: Dict[str, Type] = {}
    
    def register_detector(self, name: str, detector_class: Type[EnhancedPatternDetector]):
        """Register a pattern detector"""
        self.detectors[name] = detector_class
    
    def register_analyzer(self, name: str, analyzer_class: Type):
        """Register an analyzer"""
        self.analyzers[name] = analyzer_class
    
    def register_utility(self, name: str, utility_class: Type):
        """Register a utility"""
        self.utilities[name] = utility_class
    
    def get_detector(self, name: str) -> Type[EnhancedPatternDetector]:
        """Get detector by name"""
        return self.detectors.get(name)
    
    def list_detectors(self) -> List[str]:
        """List all registered detectors"""
        return list(self.detectors.keys())
    
    def create_detector(self, name: str, config: PatternConfig) -> EnhancedPatternDetector:
        """Create detector instance"""
        detector_class = self.get_detector(name)
        if not detector_class:
            raise ValueError(f"Detector '{name}' not found")
        return detector_class(config)

# Global plugin registry
plugin_registry = PluginRegistry()

# Register built-in detectors
plugin_registry.register_detector('vcp', VCPBreakoutDetector)
plugin_registry.register_detector('flag', FlagPatternDetector)
```

## Error Handling and Logging

### Enhanced Error Handling

```python
# error_handler.py
import logging
from typing import Optional, Dict, Any
from datetime import datetime

class TradingPatternError(Exception):
    """Base exception for trading pattern detection"""
    pass

class DataValidationError(TradingPatternError):
    """Data validation error"""
    pass

class DetectionError(TradingPatternError):
    """Pattern detection error"""
    pass

class ConfigurationError(TradingPatternError):
    """Configuration error"""
    pass

class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self, logger_name: str = "trading_patterns"):
        self.logger = logging.getLogger(logger_name)
        self.error_counts = {}
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle error with context"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log error with context
        log_message = f"Error: {error}"
        if context:
            log_message += f" | Context: {context}"
        
        self.logger.error(log_message)
        
        # Return error summary
        return {
            'error_type': error_type,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'error_count': self.error_counts[error_type]
        }
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get error summary"""
        return self.error_counts.copy()
    
    def reset_error_counts(self):
        """Reset error counts"""
        self.error_counts.clear()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_patterns.log'),
        logging.StreamHandler()
    ]
)
```

## Performance Optimization

### Performance Optimization Strategies

```python
# performance_optimizer.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class PerformanceOptimizer:
    """Performance optimization for pattern detection"""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 100):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.cache = {}
    
    def parallel_detection(self, data_dict: Dict[str, pd.DataFrame],
                          detectors: List[EnhancedPatternDetector]) -> Dict[str, List[PatternSignal]]:
        """Parallel pattern detection across multiple symbols"""
        results = {}
        
        def process_symbol(symbol_data):
            symbol, data = symbol_data
            signals = []
            
            for detector in detectors:
                try:
                    detector_signals = detector.detect_pattern(data)
                    signals.extend(detector_signals)
                except Exception as e:
                    logging.error(f"Error processing {symbol} with {detector.__class__.__name__}: {e}")
            
            return symbol, signals
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = dict(executor.map(process_symbol, data_dict.items()))
        
        return results
    
    def batch_processing(self, data_list: List[pd.DataFrame],
                        detector: EnhancedPatternDetector) -> List[List[PatternSignal]]:
        """Batch processing for large datasets"""
        results = []
        
        for i in range(0, len(data_list), self.batch_size):
            batch = data_list[i:i + self.batch_size]
            batch_results = []
            
            for data in batch:
                try:
                    signals = detector.detect_pattern(data)
                    batch_results.append(signals)
                except Exception as e:
                    logging.error(f"Error in batch processing: {e}")
                    batch_results.append([])
            
            results.extend(batch_results)
        
        return results
    
    def cache_results(self, key: str, data: Any, ttl: int = 3600):
        """Cache results with TTL"""
        import time
        self.cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    def get_cached_results(self, key: str) -> Optional[Any]:
        """Get cached results if available"""
        import time
        
        if key in self.cache:
            cache_entry = self.cache[key]
            if time.time() - cache_entry['timestamp'] < cache_entry['ttl']:
                return cache_entry['data']
            else:
                del self.cache[key]
        
        return None
    
    def optimize_data_access(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data access for faster processing"""
        # Convert to efficient dtypes
        for col in data.columns:
            if data[col].dtype == 'float64':
                data[col] = data[col].astype('float32')
            elif data[col].dtype == 'int64':
                data[col] = data[col].astype('int32')
        
        # Set categorical columns
        if 'symbol' in data.columns:
            data['symbol'] = data['symbol'].astype('category')
        
        return data
```

This modular architecture provides a solid foundation for building a comprehensive trading pattern detection system that is maintainable, extensible, and performant.