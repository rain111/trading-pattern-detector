import os
import sys
import importlib.util
import logging
from typing import List, Dict, Any, Optional
from plugins.plugin_registry import PluginRegistry
from core.interfaces import PatternConfig
import inspect


class PluginManager:
    """Manages plugin loading, registration, and lifecycle"""
    
    def __init__(self, plugin_dir: str = 'plugins', registry: Optional[PluginRegistry] = None):
        self.plugin_dir = plugin_dir
        self.registry = registry or PluginRegistry()
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Built-in plugins
        self._register_builtin_plugins()
    
    def _register_builtin_plugins(self):
        """Register built-in plugins"""
        try:
            # Import built-in detectors
            from detectors.vcp_detector import VCPBreakoutDetector
            from detectors.flag_detector import FlagPatternDetector
            from detectors.triangle_detector import TrianglePatternDetector
            from detectors.wedge_detector import WedgePatternDetector
            
            # Register detectors with metadata
            self.registry.register_detector(
                'vcp', VCPBreakoutDetector, 
                description='Volatility Contraction Pattern Breakout Detector',
                version='1.0.0',
                author='Trading Pattern System',
                category='pattern',
                tags=['vcp', 'breakout', 'volatility', 'technical-analysis']
            )
            
            self.registry.register_detector(
                'flag', FlagPatternDetector,
                description='Flag Pattern Continuation Detector',
                version='1.0.0',
                author='Trading Pattern System',
                category='pattern',
                tags=['flag', 'continuation', 'technical-analysis']
            )
            
            self.registry.register_detector(
                'triangle', TrianglePatternDetector,
                description='Triangle Pattern Detection (Ascending, Descending, Symmetrical)',
                version='1.0.0',
                author='Trading Pattern System',
                category='pattern',
                tags=['triangle', 'continuation', 'technical-analysis']
            )
            
            self.registry.register_detector(
                'wedge', WedgePatternDetector,
                description='Wedge Pattern Detection (Rising, Falling)',
                version='1.0.0',
                author='Trading Pattern System',
                category='pattern',
                tags=['wedge', 'reversal', 'technical-analysis']
            )
            
            self.logger.info("Built-in plugins registered successfully")
            
        except Exception as e:
            self.logger.error(f"Error registering built-in plugins: {e}")
    
    def load_plugin_from_file(self, plugin_path: str, plugin_name: Optional[str] = None) -> bool:
        """Load a plugin from a Python file"""
        try:
            if not os.path.exists(plugin_path):
                self.logger.error(f"Plugin file not found: {plugin_path}")
                return False
            
            # Generate plugin name if not provided
            if plugin_name is None:
                plugin_name = os.path.splitext(os.path.basename(plugin_path))[0]
            
            # Load the module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec is None or spec.loader is None:
                self.logger.error(f"Could not load plugin spec from {plugin_path}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Register plugin classes found in the module
            self._register_plugin_classes(module, plugin_name)
            
            self.loaded_plugins[plugin_name] = module
            self.logger.info(f"Plugin '{plugin_name}' loaded successfully from {plugin_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading plugin from {plugin_path}: {e}")
            return False
    
    def _register_plugin_classes(self, module, plugin_name: str):
        """Register plugin classes found in a module"""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                # Check if it's a detector
                from core.interfaces import EnhancedPatternDetector
                if issubclass(obj, EnhancedPatternDetector) and obj != EnhancedPatternDetector:
                    self.registry.register_detector(
                        f"{plugin_name}.{name.lower()}",
                        obj,
                        description=getattr(obj, '__doc__', f'{name} plugin'),
                        version=getattr(obj, '__version__', '1.0.0'),
                        author=getattr(obj, '__author__', 'Unknown'),
                        category='pattern',
                        tags=getattr(obj, '__tags__', [])
                    )
                
                # Check if it's an analyzer
                elif 'Analyzer' in name and hasattr(obj, 'analyze'):
                    self.registry.register_analyzer(
                        f"{plugin_name}.{name.lower()}",
                        obj,
                        description=getattr(obj, '__doc__', f'{name} analyzer'),
                        version=getattr(obj, '__version__', '1.0.0'),
                        author=getattr(obj, '__author__', 'Unknown'),
                        category='analysis'
                    )
                
                # Check if it's a utility
                elif 'Utility' in name and hasattr(obj, 'process'):
                    self.registry.register_utility(
                        f"{plugin_name}.{name.lower()}",
                        obj,
                        description=getattr(obj, '__doc__', f'{name} utility'),
                        version=getattr(obj, '__version__', '1.0.0'),
                        author=getattr(obj, '__author__', 'Unknown'),
                        category='utility'
                    )
    
    def load_plugins_from_directory(self, directory: Optional[str] = None) -> List[str]:
        """Load all plugins from a directory"""
        if directory is None:
            directory = self.plugin_dir
        
        loaded_plugins = []
        
        if not os.path.exists(directory):
            self.logger.warning(f"Plugin directory not found: {directory}")
            return loaded_plugins
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                plugin_path = os.path.join(directory, filename)
                plugin_name = os.path.splitext(filename)[0]
                
                if self.load_plugin_from_file(plugin_path, plugin_name):
                    loaded_plugins.append(plugin_name)
        
        self.logger.info(f"Loaded {len(loaded_plugins)} plugins from {directory}")
        return loaded_plugins
    
    def get_available_detectors(self) -> List[str]:
        """Get list of available detectors"""
        return self.registry.list_detectors()
    
    def get_available_analyzers(self) -> List[str]:
        """Get list of available analyzers"""
        return self.registry.list_analyzers()
    
    def get_available_utilities(self) -> List[str]:
        """Get list of available utilities"""
        return self.registry.list_utilities()
    
    def create_detector(self, name: str, config: PatternConfig) -> Any:
        """Create a detector instance"""
        try:
            return self.registry.create_detector(name, config)
        except Exception as e:
            self.logger.error(f"Error creating detector {name}: {e}")
            raise
    
    def create_analyzer(self, name: str, **kwargs) -> Any:
        """Create an analyzer instance"""
        try:
            return self.registry.create_analyzer(name, **kwargs)
        except Exception as e:
            self.logger.error(f"Error creating analyzer {name}: {e}")
            raise
    
    def create_utility(self, name: str, **kwargs) -> Any:
        """Create a utility instance"""
        try:
            return self.registry.create_utility(name, **kwargs)
        except Exception as e:
            self.logger.error(f"Error creating utility {name}: {e}")
            raise
    
    def get_detector_info(self, name: str) -> Dict[str, Any]:
        """Get information about a detector"""
        try:
            return self.registry.get_plugin_info(name)
        except Exception as e:
            self.logger.error(f"Error getting detector info for {name}: {e}")
            return {}
    
    def validate_plugin_dependencies(self, name: str) -> List[str]:
        """Validate plugin dependencies"""
        return self.registry.validate_plugin_dependencies(name)
    
    def get_plugins_by_category(self, category: str) -> List[str]:
        """Get plugins by category"""
        categorized = self.registry.get_plugin_by_category(category)
        return categorized.get(category, [])
    
    def search_plugins(self, query: str) -> List[str]:
        """Search plugins"""
        return self.registry.search_plugins(query)
    
    def get_plugin_health(self) -> Dict[str, Dict[str, Any]]:
        """Get plugin health status"""
        health_status = {}
        
        for name in self.registry.list_plugins():
            plugin_info = self.registry.get_plugin_info(name)
            
            health_info = {
                'loaded': name in self.loaded_plugins,
                'dependencies_ok': len(self.validate_plugin_dependencies(name)) == 0,
                'type': plugin_info['type'],
                'category': plugin_info['category'],
                'version': plugin_info['version']
            }
            
            health_status[name] = health_info
        
        return health_status
    
    def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Update plugin configuration"""
        try:
            self.plugin_configs[plugin_name] = config
            self.logger.info(f"Configuration updated for plugin {plugin_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating plugin config for {plugin_name}: {e}")
            return False
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get plugin configuration"""
        return self.plugin_configs.get(plugin_name, {})
    
    def remove_plugin(self, plugin_name: str) -> bool:
        """Remove a plugin"""
        try:
            # Remove from registry
            self.registry.unregister_plugin(plugin_name)
            
            # Remove from loaded plugins
            if plugin_name in self.loaded_plugins:
                del self.loaded_plugins[plugin_name]
            
            # Remove configuration
            if plugin_name in self.plugin_configs:
                del self.plugin_configs[plugin_name]
            
            self.logger.info(f"Plugin {plugin_name} removed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing plugin {plugin_name}: {e}")
            return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin"""
        try:
            # Find plugin file
            plugin_file = None
            for filename in os.listdir(self.plugin_dir):
                if filename.endswith('.py') and filename.startswith(plugin_name):
                    plugin_file = os.path.join(self.plugin_dir, filename)
                    break
            
            if plugin_file is None:
                self.logger.error(f"Plugin file not found for {plugin_name}")
                return False
            
            # Remove existing plugin
            self.remove_plugin(plugin_name)
            
            # Reload plugin
            return self.load_plugin_from_file(plugin_file, plugin_name)
            
        except Exception as e:
            self.logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return False
    
    def get_plugin_summary(self) -> Dict[str, Any]:
        """Get summary of all plugins"""
        return {
            'total_plugins': len(self.registry.list_plugins()),
            'detectors': len(self.registry.list_detectors()),
            'analyzers': len(self.registry.list_analyzers()),
            'utilities': len(self.registry.list_utilities()),
            'loaded_plugins': len(self.loaded_plugins),
            'categories': self.registry.get_plugin_by_category(),
            'health': self.get_plugin_health()
        }
    
    def export_plugin_list(self, output_path: str) -> bool:
        """Export plugin list to file"""
        try:
            import json
            
            plugin_list = {
                'plugins': self.registry.list_plugins(),
                'detectors': self.registry.list_detectors(),
                'analyzers': self.registry.list_analyzers(),
                'utilities': self.registry.list_utilities(),
                'summary': self.get_plugin_summary()
            }
            
            with open(output_path, 'w') as f:
                json.dump(plugin_list, f, indent=2)
            
            self.logger.info(f"Plugin list exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting plugin list: {e}")
            return False