import pandas as pd
import numpy as np
from typing import List, Dict, Any
from scipy import stats
import logging


class VolatilityAnalyzer:
    """Advanced volatility analysis for pattern detection"""
    
    def __init__(self, atr_period: int = 14, bb_period: int = 20):
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high, low, close = data['high'], data['low'], data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.atr_period).mean()
            
            return atr
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series()
    
    def calculate_volatility_contraction(self, data: pd.DataFrame, 
                                       period: int = 20) -> Dict[str, float]:
        """Detect volatility contraction patterns"""
        try:
            atr_series = self.calculate_atr(data)
            
            if len(atr_series) == 0:
                return {
                    'atr_trend': 0,
                    'contraction_ratio': 1.0,
                    'volatility_score': 0,
                    'is_contracting': False
                }
            
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
        except Exception as e:
            self.logger.error(f"Error calculating volatility contraction: {e}")
            return {
                'atr_trend': 0,
                'contraction_ratio': 1.0,
                'volatility_score': 0,
                'is_contracting': False
            }
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, 
                                period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands for volatility analysis"""
        try:
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
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return {
                'sma': pd.Series(),
                'upper_band': pd.Series(),
                'lower_band': pd.Series(),
                'band_width': pd.Series()
            }
    
    def calculate_volatility_ratio(self, data: pd.DataFrame, 
                                 short_period: int = 10, long_period: int = 30) -> float:
        """Calculate volatility ratio (short/long volatility)"""
        try:
            returns = data['close'].pct_change()
            
            short_vol = returns[-short_period:].std()
            long_vol = returns[-long_period:].std()
            
            return short_vol / long_vol if long_vol > 0 else 1.0
        except Exception as e:
            self.logger.error(f"Error calculating volatility ratio: {e}")
            return 1.0
    
    def detect_volatility_squeeze(self, data: pd.DataFrame, 
                                threshold: float = 0.1) -> bool:
        """Detect volatility squeeze (very low volatility)"""
        try:
            bb_data = self.calculate_bollinger_bands(data)
            band_width = bb_data['band_width']
            
            if len(band_width) == 0:
                return False
            
            return band_width.iloc[-1] < threshold
        except Exception as e:
            self.logger.error(f"Error detecting volatility squeeze: {e}")
            return False
    
    def calculate_keltner_channels(self, data: pd.DataFrame, 
                                 period: int = 20, atr_multiplier: float = 2) -> Dict[str, pd.Series]:
        """Calculate Keltner Channels for volatility analysis"""
        try:
            ema = data['close'].ewm(span=period).mean()
            atr = self.calculate_atr(data)
            
            upper_channel = ema + (atr * atr_multiplier)
            lower_channel = ema - (atr * atr_multiplier)
            
            return {
                'ema': ema,
                'upper_channel': upper_channel,
                'lower_channel': lower_channel
            }
        except Exception as e:
            self.logger.error(f"Error calculating Keltner Channels: {e}")
            return {
                'ema': pd.Series(),
                'upper_channel': pd.Series(),
                'lower_channel': pd.Series()
            }