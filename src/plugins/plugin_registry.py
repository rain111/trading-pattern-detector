import importlib
import logging
from typing import Dict, List, Type, Any, Optional
from core.interfaces import EnhancedPatternDetector, PatternConfig
import inspect


class PluginRegistry:
    """Registry for pattern detection plugins"""

    def __init__(self):
        self.detectors: Dict[str, Type[EnhancedPatternDetector]] = {}
        self.analyzers: Dict[str, Type] = {}
        self.utilities: Dict[str, Type] = {}
        self.plugins: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def register_detector(
        self,
        name: str,
        detector_class: Type[EnhancedPatternDetector],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a pattern detector"""
        try:
            # Validate detector class
            if not issubclass(detector_class, EnhancedPatternDetector):
                raise ValueError(
                    f"Detector {name} must inherit from EnhancedPatternDetector"
                )

            self.detectors[name] = detector_class

            # Store plugin metadata
            plugin_info = {
                "type": "detector",
                "class": detector_class,
                "module": detector_class.__module__,
                "description": metadata.get("description", ""),
                "version": metadata.get("version", "1.0.0"),
                "author": metadata.get("author", ""),
                "category": metadata.get("category", "pattern"),
                "tags": metadata.get("tags", []),
            }

            self.plugins[name] = plugin_info

            self.logger.info(f"Detector '{name}' registered successfully")

        except Exception as e:
            self.logger.error(f"Error registering detector {name}: {e}")
            raise

    def register_analyzer(
        self, name: str, analyzer_class: Type, metadata: Optional[Dict[str, Any]] = None
    ):
        """Register an analyzer"""
        try:
            self.analyzers[name] = analyzer_class

            # Store plugin metadata
            plugin_info = {
                "type": "analyzer",
                "class": analyzer_class,
                "module": analyzer_class.__module__,
                "description": metadata.get("description", ""),
                "version": metadata.get("version", "1.0.0"),
                "author": metadata.get("author", ""),
                "category": metadata.get("category", "analysis"),
            }

            self.plugins[name] = plugin_info

            self.logger.info(f"Analyzer '{name}' registered successfully")

        except Exception as e:
            self.logger.error(f"Error registering analyzer {name}: {e}")
            raise

    def register_utility(
        self, name: str, utility_class: Type, metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a utility"""
        try:
            self.utilities[name] = utility_class

            # Store plugin metadata
            plugin_info = {
                "type": "utility",
                "class": utility_class,
                "module": utility_class.__module__,
                "description": metadata.get("description", ""),
                "version": metadata.get("version", "1.0.0"),
                "author": metadata.get("author", ""),
                "category": metadata.get("category", "utility"),
            }

            self.plugins[name] = plugin_info

            self.logger.info(f"Utility '{name}' registered successfully")

        except Exception as e:
            self.logger.error(f"Error registering utility {name}: {e}")
            raise

    def get_detector(self, name: str) -> Type[EnhancedPatternDetector]:
        """Get detector by name"""
        detector_class = self.detectors.get(name)
        if not detector_class:
            raise ValueError(f"Detector '{name}' not found")
        return detector_class

    def get_analyzer(self, name: str) -> Type:
        """Get analyzer by name"""
        analyzer_class = self.analyzers.get(name)
        if not analyzer_class:
            raise ValueError(f"Analyzer '{name}' not found")
        return analyzer_class

    def get_utility(self, name: str) -> Type:
        """Get utility by name"""
        utility_class = self.utilities.get(name)
        if not utility_class:
            raise ValueError(f"Utility '{name}' not found")
        return utility_class

    def list_detectors(self) -> List[str]:
        """List all registered detectors"""
        return list(self.detectors.keys())

    def list_analyzers(self) -> List[str]:
        """List all registered analyzers"""
        return list(self.analyzers.keys())

    def list_utilities(self) -> List[str]:
        """List all registered utilities"""
        return list(self.utilities.keys())

    def list_plugins(self, plugin_type: Optional[str] = None) -> List[str]:
        """List all plugins, optionally filtered by type"""
        if plugin_type:
            return [
                name
                for name, info in self.plugins.items()
                if info["type"] == plugin_type
            ]
        return list(self.plugins.keys())

    def get_plugin_info(self, name: str) -> Dict[str, Any]:
        """Get information about a plugin"""
        if name not in self.plugins:
            raise ValueError(f"Plugin '{name}' not found")
        return self.plugins[name]

    def create_detector(
        self, name: str, config: PatternConfig
    ) -> EnhancedPatternDetector:
        """Create detector instance"""
        detector_class = self.get_detector(name)
        return detector_class(config)

    def create_analyzer(self, name: str, **kwargs) -> Any:
        """Create analyzer instance"""
        analyzer_class = self.get_analyzer(name)
        return analyzer_class(**kwargs)

    def create_utility(self, name: str, **kwargs) -> Any:
        """Create utility instance"""
        utility_class = self.get_utility(name)
        return utility_class(**kwargs)

    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin"""
        try:
            if name in self.detectors:
                del self.detectors[name]
            if name in self.analyzers:
                del self.analyzers[name]
            if name in self.utilities:
                del self.utilities[name]
            if name in self.plugins:
                del self.plugins[name]

            self.logger.info(f"Plugin '{name}' unregistered successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error unregistering plugin {name}: {e}")
            return False

    def clear_plugins(self):
        """Clear all plugins"""
        self.detectors.clear()
        self.analyzers.clear()
        self.utilities.clear()
        self.plugins.clear()

        self.logger.info("All plugins cleared")

    def get_plugin_dependencies(self, name: str) -> List[str]:
        """Get plugin dependencies"""
        plugin_info = self.get_plugin_info(name)
        dependencies = plugin_info.get("dependencies", [])
        return dependencies

    def validate_plugin_dependencies(self, name: str) -> List[str]:
        """Validate plugin dependencies"""
        dependencies = self.get_plugin_dependencies(name)
        missing_dependencies = []

        for dep in dependencies:
            if dep not in self.plugins:
                missing_dependencies.append(dep)

        return missing_dependencies

    def get_plugin_by_category(self, category: str) -> Dict[str, List[str]]:
        """Get plugins by category"""
        categorized_plugins = {}

        for name, info in self.plugins.items():
            if info["category"] not in categorized_plugins:
                categorized_plugins[info["category"]] = []
            categorized_plugins[info["category"]].append(name)

        return categorized_plugins

    def get_plugin_tags(self) -> Dict[str, List[str]]:
        """Get all plugin tags"""
        tags = {}

        for name, info in self.plugins.items():
            plugin_tags = info.get("tags", [])
            for tag in plugin_tags:
                if tag not in tags:
                    tags[tag] = []
                tags[tag].append(name)

        return tags

    def search_plugins(self, query: str) -> List[str]:
        """Search plugins by name, description, or tags"""
        query_lower = query.lower()
        matches = []

        for name, info in self.plugins.items():
            # Search by name
            if query_lower in name.lower():
                matches.append(name)
                continue

            # Search by description
            if query_lower in info.get("description", "").lower():
                matches.append(name)
                continue

            # Search by tags
            plugin_tags = info.get("tags", [])
            if any(query_lower in tag.lower() for tag in plugin_tags):
                matches.append(name)

        return matches

    def get_detector_requirements(self, name: str) -> Dict[str, Any]:
        """Get detector requirements"""
        detector_class = self.get_detector(name)

        # Get required columns
        if hasattr(detector_class, "get_required_columns"):
            required_columns = detector_class.get_required_columns()
        else:
            required_columns = ["open", "high", "low", "close", "volume"]

        # Get detector-specific parameters
        detector_params = {}

        # Try to get from class attributes
        if hasattr(detector_class, "default_parameters"):
            detector_params = detector_class.default_parameters

        return {
            "required_columns": required_columns,
            "default_parameters": detector_params,
            "class": detector_class,
        }
