"""Detectors module for pattern detection components"""

import logging

logger = logging.getLogger(__name__)

from .vcp_detector import VCPBreakoutDetector
from .flag_detector import FlagPatternDetector
from .triangle_detector import TrianglePatternDetector
from .wedge_detector import WedgePatternDetector
from .cup_handle_detector import CupHandleDetector
from .double_bottom_detector import DoubleBottomDetector

__all__ = [
    'VCPBreakoutDetector',
    'FlagPatternDetector',
    'TrianglePatternDetector',
    'WedgePatternDetector',
    'CupHandleDetector',
    'DoubleBottomDetector'
]

logger.info("Pattern detection modules loaded successfully")