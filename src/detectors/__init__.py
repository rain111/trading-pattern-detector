"""Detectors module for pattern detection components"""

import logging

logger = logging.getLogger(__name__)

from .vcp_detector import VCPBreakoutDetector
from .flag_detector import FlagPatternDetector
from .triangle_detector import TrianglePatternDetector
from .wedge_detector import WedgePatternDetector
from .cup_handle_detector import CupHandleDetector
from .double_bottom_detector import DoubleBottomDetector
from .head_and_shoulders_detector import HeadAndShouldersDetector
from .rounding_bottom_detector import RoundingBottomDetector
from .ascending_triangle_detector import AscendingTriangleDetector
from .descending_triangle_detector import DescendingTriangleDetector
from .rising_wedge_detector import RisingWedgeDetector
from .falling_wedge_detector import FallingWedgeDetector

__all__ = [
    'VCPBreakoutDetector',
    'FlagPatternDetector',
    'TrianglePatternDetector',
    'WedgePatternDetector',
    'CupHandleDetector',
    'DoubleBottomDetector',
    'HeadAndShouldersDetector',
    'RoundingBottomDetector',
    'AscendingTriangleDetector',
    'DescendingTriangleDetector',
    'RisingWedgeDetector',
    'FallingWedgeDetector'
]

logger.info("Pattern detection modules loaded successfully")