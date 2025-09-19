"""
Trading Pattern Detector

A sophisticated Python-based financial trading pattern detection system
designed to identify various chart patterns in market data.
"""

__version__ = "0.1.0"
__author__ = "Trading Pattern Detector Team"
__email__ = "team@tradingpatterns.com"

from .core.interfaces import (
    PatternConfig,
    PatternSignal,
    PatternType,
    PatternEngine,
    DataValidator,
    BaseDetector,
    EnhancedPatternDetector,
)

from .detectors import (
    VCPBreakoutDetector,
    FlagPatternDetector,
    TrianglePatternDetector,
    WedgePatternDetector,
    CupHandleDetector,
    DoubleBottomDetector,
)

__all__ = [
    "PatternConfig",
    "PatternSignal",
    "PatternType",
    "PatternEngine",
    "DataValidator",
    "BaseDetector",
    "EnhancedPatternDetector",
    "VCPBreakoutDetector",
    "FlagPatternDetector",
    "TrianglePatternDetector",
    "WedgePatternDetector",
    "CupHandleDetector",
    "DoubleBottomDetector",
]
