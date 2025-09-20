"""
Pattern Detection Engine Integration - Connects frontend with backend pattern detection
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

from ...config import settings
from ...data import DataManager
from ...utils.logger import get_logger

# Import from the backend pattern detection system
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from core.interfaces import PatternConfig, PatternEngine, PatternSignal
from core.market_data import MarketDataIngestor
from detectors import (
    VCPBreakoutDetector,
    FlagPatternDetector,
    CupHandleDetector,
    DoubleBottomDetector,
    HeadAndShouldersDetector,
    RoundingBottomDetector,
    AscendingTriangleDetector,
    DescendingTriangleDetector,
    RisingWedgeDetector,
    FallingWedgeDetector,
)
from core.interfaces import DataValidator

class PatternDetectionEngine:
    """Integrates frontend with backend pattern detection engine"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.data_manager = None
        self.ingestor = None
        self.validator = DataValidator()

    async def initialize(self):
        """Initialize the engine"""
        self.data_manager = DataManager()
        self.ingestor = MarketDataIngestor()
        await self.data_manager.__aenter__()

    async def cleanup(self):
        """Cleanup resources"""
        if self.data_manager:
            await self.data_manager.__aexit__(None, None, None)

    async def detect_patterns(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        pattern_types: List[str],
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect patterns in stock data

        Args:
            symbol: Stock symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            pattern_types: List of pattern types to detect
            confidence_threshold: Minimum confidence threshold

        Returns:
            Dictionary containing detection results
        """
        try:
            self.logger.info(f"Starting pattern detection for {symbol}")

            # Validate inputs
            if not symbol or not symbol.isalnum():
                raise ValueError("Invalid stock symbol")

            if start_date >= end_date:
                raise ValueError("Start date must be before end date")

            if not pattern_types:
                raise ValueError("No pattern types specified")

            # Fetch data using the data manager
            data = await self.data_manager.get_stock_data(symbol, start_date, end_date)

            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return {
                    'success': False,
                    'error': "No data available for the specified date range",
                    'data': pd.DataFrame(),
                    'signals': []
                }

            # Clean and validate data
            data = self.validator.clean_ohlc_data(data)

            try:
                self.validator.validate_price_data(data)
            except ValueError as e:
                self.logger.error(f"Data validation failed: {e}")
                return {
                    'success': False,
                    'error': f"Data validation failed: {e}",
                    'data': data,
                    'signals': []
                }

            # Create pattern detection configuration
            config = PatternConfig(min_confidence=confidence_threshold)

            # Create detectors for specified patterns
            detectors = []
            for pattern_type in pattern_types:
                detector = self._create_detector(pattern_type, config)
                if detector:
                    detectors.append(detector)

            if not detectors:
                raise ValueError("No valid detectors could be created")

            # Create pattern engine
            engine = PatternEngine(detectors)

            # Run pattern detection
            signals = engine.detect_patterns(data, symbol)

            # Process results
            processed_signals = []
            for signal in signals:
                processed_signal = self._process_signal(signal, data)
                if processed_signal:
                    processed_signals.append(processed_signal)

            # Calculate additional metrics
            metrics = self._calculate_metrics(processed_signals, data)

            results = {
                'success': True,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'data': data,
                'signals': processed_signals,
                'metrics': metrics,
                'detection_config': {
                    'confidence_threshold': confidence_threshold,
                    'pattern_types': pattern_types,
                    'data_points': len(data)
                },
                'timestamp': datetime.now()
            }

            self.logger.info(f"Pattern detection completed for {symbol}: {len(processed_signals)} signals found")

            return results

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': pd.DataFrame(),
                'signals': []
            }

    def _create_detector(self, pattern_type: str, config: PatternConfig):
        """Create a detector for the specified pattern type"""
        detector_map = {
            'VCP_BREAKOUT': VCPBreakoutDetector,
            'FLAG_PATTERN': FlagPatternDetector,
            'CUP_HANDLE': CupHandleDetector,
            'DOUBLE_BOTTOM': DoubleBottomDetector,
            'HEAD_AND_SHOULDERS': HeadAndShouldersDetector,
            'ROUNDING_BOTTOM': RoundingBottomDetector,
            'ASCENDING_TRIANGLE': AscendingTriangleDetector,
            'DESCENDING_TRIANGLE': DescendingTriangleDetector,
            'RISING_WEDGE': RisingWedgeDetector,
            'FALLING_WEDGE': FallingWedgeDetector,
        }

        detector_class = detector_map.get(pattern_type)
        if detector_class:
            return detector_class(config)
        else:
            self.logger.warning(f"No detector available for pattern type: {pattern_type}")
            return None

    def _process_signal(self, signal: PatternSignal, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Process a single pattern signal"""
        try:
            # Get signal date
            signal_date = signal.timestamp

            # Check if signal date is in data
            if signal_date not in data.index:
                self.logger.warning(f"Signal date {signal_date} not in data index")
                return None

            # Get price at signal date
            signal_data = data.loc[signal_date]

            # Calculate potential return
            entry_price = signal.entry_price
            target_price = signal.target_price
            potential_return = (target_price - entry_price) / entry_price if entry_price > 0 else 0

            # Calculate risk/reward ratio
            stop_loss = signal.stop_loss
            risk_distance = entry_price - stop_loss
            reward_distance = target_price - entry_price
            risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0

            # Create processed signal dictionary
            processed_signal = {
                'symbol': signal.symbol,
                'pattern_type': signal.pattern_type.value,
                'timestamp': signal_date,
                'confidence': signal.confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'potential_return': potential_return,
                'risk_reward_ratio': risk_reward_ratio,
                'risk_level': signal.risk_level,
                'expected_duration': signal.expected_duration,
                'signal_strength': signal.signal_strength,
                'metadata': signal.metadata or {}
            }

            # Add price context
            processed_signal.update({
                'open': signal_data['open'],
                'high': signal_data['high'],
                'low': signal_data['low'],
                'close': signal_data['close'],
                'volume': signal_data['volume']
            })

            return processed_signal

        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            return None

    def _calculate_metrics(self, signals: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for detected signals"""
        try:
            if not signals:
                return {
                    'total_signals': 0,
                    'avg_confidence': 0,
                    'avg_potential_return': 0,
                    'avg_risk_reward_ratio': 0,
                    'risk_level_distribution': {'high': 0, 'medium': 0, 'low': 0},
                    'pattern_type_distribution': {},
                    'signal_density': 0
                }

            # Basic metrics
            total_signals = len(signals)
            avg_confidence = sum(s.get('confidence', 0) for s in signals) / total_signals
            avg_potential_return = sum(s.get('potential_return', 0) for s in signals) / total_signals
            avg_risk_reward_ratio = sum(s.get('risk_reward_ratio', 0) for s in signals) / total_signals

            # Risk level distribution
            risk_levels = [s.get('risk_level', 'unknown') for s in signals]
            risk_distribution = {
                'high': risk_levels.count('high'),
                'medium': risk_levels.count('medium'),
                'low': risk_levels.count('low')
            }

            # Pattern type distribution
            pattern_types = [s.get('pattern_type', 'unknown') for s in signals]
            pattern_distribution = {}
            for pattern in pattern_types:
                pattern_distribution[pattern] = pattern_distribution.get(pattern, 0) + 1

            # Signal density (signals per month)
            data_duration_days = (data.index.max() - data.index.min()).days
            data_duration_months = data_duration_days / 30.44  # Average days per month
            signal_density = total_signals / data_duration_months if data_duration_months > 0 else 0

            # Confidence distribution
            confidence_scores = [s.get('confidence', 0) for s in signals]
            high_confidence_signals = len([c for c in confidence_scores if c >= 0.8])
            medium_confidence_signals = len([c for c in confidence_scores if 0.5 <= c < 0.8])
            low_confidence_signals = len([c for c in confidence_scores if c < 0.5])

            # Return analysis
            returns = [s.get('potential_return', 0) for s in signals]
            profitable_signals = len([r for r in returns if r > 0])
            unprofitable_signals = len([r for r in returns if r <= 0])

            metrics = {
                'total_signals': total_signals,
                'avg_confidence': avg_confidence,
                'avg_potential_return': avg_potential_return,
                'avg_risk_reward_ratio': avg_risk_reward_ratio,
                'high_confidence_signals': high_confidence_signals,
                'medium_confidence_signals': medium_confidence_signals,
                'low_confidence_signals': low_confidence_signals,
                'profitable_signals': profitable_signals,
                'unprofitable_signals': unprofitable_signals,
                'profitability_rate': profitable_signals / total_signals if total_signals > 0 else 0,
                'risk_level_distribution': risk_distribution,
                'pattern_type_distribution': pattern_distribution,
                'signal_density': signal_density,
                'data_points': len(data),
                'analysis_period_days': data_duration_days,
                'analysis_period_months': data_duration_months
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {
                'total_signals': len(signals),
                'error': str(e)
            }

    async def get_available_symbols(self) -> List[str]:
        """Get list of available symbols with cached data"""
        if not self.data_manager:
            await self.initialize()
        return self.data_manager.get_available_symbols()

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about a symbol's data"""
        if not self.data_manager:
            await self.initialize()

        cache_info = self.data_manager.get_cache_info(symbol)
        if cache_info:
            return {
                'symbol': symbol,
                'cache_info': cache_info,
                'available': True
            }
        else:
            return {
                'symbol': symbol,
                'available': False,
                'message': 'No cached data available'
            }

    async def validate_inputs(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        pattern_types: List[str],
        confidence_threshold: float
    ) -> List[str]:
        """Validate input parameters"""
        errors = []

        # Validate symbol
        if not symbol or not symbol.isalnum():
            errors.append("Stock symbol must be alphanumeric")

        # Validate date range
        if start_date >= end_date:
            errors.append("Start date must be before end date")

        date_range_days = (end_date - start_date).days
        if date_range_days > 365 * 10:
            errors.append("Date range cannot exceed 10 years")
        elif date_range_days < 30:
            errors.append("Date range must be at least 30 days")

        # Validate pattern types
        if not pattern_types:
            errors.append("At least one pattern type must be selected")

        valid_patterns = set(settings.SUPPORTED_PATTERNS)
        invalid_patterns = [p for p in pattern_types if p not in valid_patterns]
        if invalid_patterns:
            errors.append(f"Invalid pattern types: {', '.join(invalid_patterns)}")

        # Validate confidence threshold
        if not 0.0 <= confidence_threshold <= 1.0:
            errors.append("Confidence threshold must be between 0.0 and 1.0")

        return errors

    def get_supported_patterns(self) -> List[str]:
        """Get list of supported pattern types"""
        return settings.SUPPORTED_PATTERNS.copy()