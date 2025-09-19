import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from core.interfaces import PatternSignal, PatternType
import logging


class SignalAggregator:
    """Aggregate and rank signals from multiple detectors"""
    
    def __init__(self):
        self.signals = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
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
        elif ranking_method == 'probability':
            return sorted(self.signals, key=lambda x: x.probability_target or 0.5, reverse=True)
        elif ranking_method == 'reward_ratio':
            reward_ratios = []
            for signal in self.signals:
                reward_ratio = (signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss)
                reward_ratios.append(reward_ratio)
            # Create list of tuples with index for sorting
            signal_tuples = list(zip(self.signals, reward_ratios))
            # Sort by reward ratio (descending)
            signal_tuples.sort(key=lambda x: x[1], reverse=True)
            return [signal for signal, _ in signal_tuples]
        else:
            return sorted(self.signals, key=lambda x: x.confidence, reverse=True)
    
    def filter_signals(self, min_confidence: float = 0.6, 
                      min_volume: Optional[float] = None,
                      max_risk: Optional[str] = None,
                      pattern_types: Optional[List[PatternType]] = None,
                      time_window: Optional[timedelta] = None) -> List[PatternSignal]:
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
            
            # Pattern type filter
            if pattern_types is not None and signal.pattern_type not in pattern_types:
                continue
            
            # Time window filter
            if time_window is not None:
                signal_time = signal.timestamp
                cutoff_time = datetime.now() - time_window
                if signal_time < cutoff_time:
                    continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def get_signals_by_symbol(self, symbol: str) -> List[PatternSignal]:
        """Get signals for a specific symbol"""
        return [signal for signal in self.signals if signal.symbol == symbol]
    
    def get_signals_by_pattern_type(self, pattern_type: PatternType) -> List[PatternSignal]:
        """Get signals by pattern type"""
        return [signal for signal in self.signals if signal.pattern_type == pattern_type]
    
    def get_signals_by_timeframe(self, timeframe: str) -> List[PatternSignal]:
        """Get signals by timeframe"""
        return [signal for signal in self.signals if signal.timeframe == timeframe]
    
    def get_signals_by_risk_level(self, risk_level: str) -> List[PatternSignal]:
        """Get signals by risk level"""
        return [signal for signal in self.signals if signal.risk_level == risk_level]
    
    def aggregate_by_symbol(self) -> Dict[str, List[PatternSignal]]:
        """Aggregate signals by symbol"""
        symbol_signals = {}
        
        for signal in self.signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        return symbol_signals
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get statistics about all signals"""
        if not self.signals:
            return {}
        
        # Basic counts
        total_signals = len(self.signals)
        
        # Pattern type distribution
        pattern_counts = {}
        for signal in self.signals:
            pattern_type = signal.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        # Risk level distribution
        risk_counts = {}
        for signal in self.signals:
            risk_level = signal.risk_level
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        # Confidence statistics
        confidences = [signal.confidence for signal in self.signals]
        confidence_stats = {
            'mean': np.mean(confidences),
            'median': np.median(confidences),
            'std': np.std(confidences),
            'min': min(confidences),
            'max': max(confidences)
        }
        
        # Signal strength statistics
        strengths = [signal.signal_strength for signal in self.signals]
        strength_stats = {
            'mean': np.mean(strengths),
            'median': np.median(strengths),
            'std': np.std(strengths),
            'min': min(strengths),
            'max': max(strengths)
        }
        
        return {
            'total_signals': total_signals,
            'pattern_distribution': pattern_counts,
            'risk_distribution': risk_counts,
            'confidence_stats': confidence_stats,
            'strength_stats': strength_stats,
            'time_range': {
                'start': min(signal.timestamp for signal in self.signals),
                'end': max(signal.timestamp for signal in self.signals)
            }
        }
    
    def remove_duplicate_signals(self, tolerance: float = 0.02) -> List[PatternSignal]:
        """Remove duplicate signals within tolerance"""
        unique_signals = []
        processed_signals = []
        
        for i, signal1 in enumerate(self.signals):
            is_duplicate = False
            
            for j, signal2 in enumerate(processed_signals):
                if signal1.symbol == signal2.symbol and signal1.pattern_type == signal2.pattern_type:
                    # Check if signals are too similar
                    price_diff = abs(signal1.entry_price - signal2.entry_price) / signal2.entry_price
                    time_diff = abs(signal1.timestamp - signal2.timestamp).days
                    
                    if price_diff < tolerance and time_diff < 5:
                        is_duplicate = True
                        # Keep the stronger signal
                        if signal1.confidence > signal2.confidence:
                            processed_signals[j] = signal1
                        break
            
            if not is_duplicate:
                processed_signals.append(signal1)
        
        return processed_signals
    
    def get_top_signals(self, n: int = 10, ranking_method: str = 'confidence') -> List[PatternSignal]:
        """Get top N signals by ranking method"""
        ranked_signals = self.rank_signals(ranking_method)
        return ranked_signals[:n]
    
    def export_signals(self, format_type: str = 'dict') -> Any:
        """Export signals in specified format"""
        if format_type == 'dict':
            return [signal.__dict__ for signal in self.signals]
        elif format_type == 'dataframe':
            return self.signals_to_dataframe()
        elif format_type == 'json':
            import json
            signals_dict = [signal.__dict__ for signal in self.signals]
            return json.dumps(signals_dict, default=str, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def signals_to_dataframe(self) -> pd.DataFrame:
        """Convert signals to DataFrame for analysis"""
        try:
            data = []
            for signal in self.signals:
                signal_dict = {
                    'symbol': signal.symbol,
                    'pattern_type': signal.pattern_type.value,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'target_price': signal.target_price,
                    'timeframe': signal.timeframe,
                    'timestamp': signal.timestamp,
                    'signal_strength': signal.signal_strength,
                    'risk_level': signal.risk_level,
                    'probability_target': signal.probability_target
                }
                # Add metadata as separate columns
                if signal.metadata:
                    for key, value in signal.metadata.items():
                        signal_dict[f'meta_{key}'] = value
                
                data.append(signal_dict)
            
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Error converting signals to DataFrame: {e}")
            return pd.DataFrame()
    
    def clear_signals(self):
        """Clear all signals"""
        self.signals.clear()
    
    def merge_with_aggregator(self, other_aggregator):
        """Merge signals from another aggregator"""
        self.signals.extend(other_aggregator.signals)
    
    def get_signals_by_time_period(self, start_date: datetime, end_date: datetime) -> List[PatternSignal]:
        """Get signals within a specific time period"""
        return [signal for signal in self.signals 
                if start_date <= signal.timestamp <= end_date]
    
    def calculate_portfolio_risk_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio risk metrics for all signals"""
        try:
            if not self.signals:
                return {}
            
            # Calculate maximum drawdown risk
            entry_prices = [signal.entry_price for signal in self.signals]
            stop_losses = [signal.stop_loss for signal in self.signals]
            
            max_drawdown = max([(entry - stop) / entry for entry, stop in zip(entry_prices, stop_losses)])
            
            # Calculate average risk/reward ratio
            risk_rewards = []
            for signal in self.signals:
                if signal.entry_price != signal.stop_loss:
                    risk_reward = (signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss)
                    risk_rewards.append(risk_reward)
            
            avg_risk_reward = np.mean(risk_rewards) if risk_rewards else 0
            
            # Calculate position sizing suggestions
            total_signals = len(self.signals)
            recommended_position_size = 1.0 / max(total_signals, 1)  # Equal position sizing
            
            return {
                'max_drawdown_risk': max_drawdown,
                'average_risk_reward': avg_risk_reward,
                'total_signals': total_signals,
                'recommended_position_size': recommended_position_size,
                'risk_adjusted_return': avg_risk_reward * (1 - max_drawdown)
            }
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}