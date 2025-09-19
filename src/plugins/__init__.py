"""Plugin system for pattern detection"""

import logging

logger = logging.getLogger(__name__)

from .plugin_registry import PluginRegistry
from .plugin_manager import PluginManager

__all__ = ["PluginRegistry", "PluginManager"]

logger.info("Plugin system modules loaded successfully")
