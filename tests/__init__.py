"""Test suite for trading pattern detection system"""

from .test_core_interfaces import TestPatternConfig, TestPatternSignal, TestDataValidator, TestEnhancedPatternDetector, TestBaseDetector, TestPatternEngine
# Additional test modules will be created as needed
# from .test_detectors import TestDetectors
# from .test_analyzers import TestAnalyzers
# from .test_utilities import TestUtilities
# from .test_config_system import TestConfigSystem
# from .test_plugin_system import TestPluginSystem
# from .test_integration import TestIntegration

__all__ = [
    'TestPatternConfig',
    'TestPatternSignal',
    'TestDataValidator',
    'TestEnhancedPatternDetector',
    'TestBaseDetector',
    'TestPatternEngine'
]