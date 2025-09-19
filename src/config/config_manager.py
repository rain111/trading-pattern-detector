import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from .pattern_parameters import PatternParameters
from .market_data_config import MarketDataConfig
from .detection_settings import DetectionSettings


class ConfigManager:
    """Centralized configuration management system"""
    
    def __init__(self, config_dir: str = 'config'):
        self.config_dir = config_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize configuration objects
        self.pattern_params = PatternParameters()
        self.market_data_config = MarketDataConfig()
        self.detection_settings = DetectionSettings()
        
        # Load configurations
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all configuration files"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Load pattern parameters
            pattern_config_path = os.path.join(self.config_dir, 'pattern_parameters.yaml')
            if os.path.exists(pattern_config_path):
                self.pattern_params = PatternParameters.load_from_yaml(pattern_config_path)
            
            # Load market data configuration
            market_data_config_path = os.path.join(self.config_dir, 'market_data_config.yaml')
            if os.path.exists(market_data_config_path):
                self.market_data_config = MarketDataConfig.load_from_yaml(market_data_config_path)
            
            # Load detection settings
            detection_settings_path = os.path.join(self.config_dir, 'detection_settings.yaml')
            if os.path.exists(detection_settings_path):
                self.detection_settings = DetectionSettings.load_from_yaml(detection_settings_path)
            
            self.logger.info("Configuration files loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading configurations: {e}")
            # Use default configurations if loading fails
    
    def get_pattern_parameters(self) -> PatternParameters:
        """Get pattern parameters"""
        return self.pattern_params
    
    def get_market_data_config(self) -> MarketDataConfig:
        """Get market data configuration"""
        return self.market_data_config
    
    def get_detection_settings(self) -> DetectionSettings:
        """Get detection settings"""
        return self.detection_settings
    
    def get_detector_config(self, detector_name: str) -> Dict[str, Any]:
        """Get configuration for a specific detector"""
        return self.pattern_params.get_detector_params(detector_name)
    
    def update_detector_config(self, detector_name: str, params: Dict[str, Any]) -> None:
        """Update configuration for a specific detector"""
        self.pattern_params.update_detector_params(detector_name, params)
    
    def validate_all_configurations(self) -> Dict[str, List[str]]:
        """Validate all configurations and return errors"""
        all_errors = {
            'pattern_parameters': self.pattern_params.validate(),
            'market_data_config': self.market_data_config.validate(),
            'detection_settings': self.detection_settings.validate()
        }
        
        # Log errors
        for config_name, errors in all_errors.items():
            if errors:
                self.logger.error(f"Validation errors in {config_name}: {errors}")
            else:
                self.logger.info(f"{config_name} validation passed")
        
        return all_errors
    
    def save_all_configurations(self) -> None:
        """Save all configurations to files"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Save pattern parameters
            pattern_config_path = os.path.join(self.config_dir, 'pattern_parameters.yaml')
            self.pattern_params.save_to_yaml(pattern_config_path)
            
            # Save market data configuration
            market_data_config_path = os.path.join(self.config_dir, 'market_data_config.yaml')
            self.market_data_config.save_to_yaml(market_data_config_path)
            
            # Save detection settings
            detection_settings_path = os.path.join(self.config_dir, 'detection_settings.yaml')
            self.detection_settings.save_to_yaml(detection_settings_path)
            
            self.logger.info("All configurations saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving configurations: {e}")
            raise
    
    def create_default_config_files(self) -> None:
        """Create default configuration files"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Create default pattern parameters
            pattern_config_path = os.path.join(self.config_dir, 'pattern_parameters.yaml')
            if not os.path.exists(pattern_config_path):
                self.pattern_params.save_to_yaml(pattern_config_path)
            
            # Create default market data configuration
            market_data_config_path = os.path.join(self.config_dir, 'market_data_config.yaml')
            if not os.path.exists(market_data_config_path):
                self.market_data_config.save_to_yaml(market_data_config_path)
            
            # Create default detection settings
            detection_settings_path = os.path.join(self.config_dir, 'detection_settings.yaml')
            if not os.path.exists(detection_settings_path):
                self.detection_settings.save_to_yaml(detection_settings_path)
            
            self.logger.info("Default configuration files created")
            
        except Exception as e:
            self.logger.error(f"Error creating default config files: {e}")
            raise
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations"""
        return {
            'pattern_parameters': {
                'vcp_detector': len(self.pattern_params.vcp_detector),
                'flag_detector': len(self.pattern_params.flag_detector),
                'triangle_detector': len(self.pattern_params.triangle_detector),
                'wedge_detector': len(self.pattern_params.wedge_detector)
            },
            'market_data_config': {
                'default_source': self.market_data_config.default_source,
                'timeframes': len(self.market_data_config.timeframes),
                'cache_enabled': self.market_data_config.cache_enabled,
                'cache_duration': self.market_data_config.cache_duration
            },
            'detection_settings': {
                'parallel_processing': self.detection_settings.parallel_processing,
                'max_workers': self.detection_settings.max_workers,
                'batch_size': self.detection_settings.batch_size,
                'confidence_threshold': self.detection_settings.confidence_threshold,
                'ranking_method': self.detection_settings.ranking_method
            }
        }
    
    def export_configuration(self, output_path: str, format_type: str = 'yaml') -> None:
        """Export all configurations to a single file"""
        try:
            config_data = {
                'pattern_parameters': self.pattern_params.to_dict(),
                'market_data_config': self.market_data_config.to_dict(),
                'detection_settings': self.detection_settings.to_dict()
            }
            
            if format_type == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format_type == 'json':
                import json
                with open(output_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            self.logger.info(f"Configuration exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            raise
    
    def import_configuration(self, input_path: str) -> None:
        """Import configuration from a file"""
        try:
            if input_path.endswith('.yaml') or input_path.endswith('.yml'):
                with open(input_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif input_path.endswith('.json'):
                import json
                with open(input_path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {input_path}")
            
            # Update configurations
            if 'pattern_parameters' in config_data:
                self.pattern_params = PatternParameters.from_dict(config_data['pattern_parameters'])
            
            if 'market_data_config' in config_data:
                self.market_data_config = MarketDataConfig.from_dict(config_data['market_data_config'])
            
            if 'detection_settings' in config_data:
                self.detection_settings = DetectionSettings.from_dict(config_data['detection_settings'])
            
            self.logger.info(f"Configuration imported from {input_path}")
            
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            raise
    
    def reset_to_defaults(self) -> None:
        """Reset all configurations to defaults"""
        self.pattern_params = PatternParameters()
        self.market_data_config = MarketDataConfig()
        self.detection_settings = DetectionSettings()
        self.logger.info("Configurations reset to defaults")