"""
Frontend configuration settings with enhanced backend integration
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Application paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data"  # Store data in parent data/ folder
STATIC_DIR = BASE_DIR / "static"
SRC_DIR = BASE_DIR.parent / "src"

# Data storage configuration
PARQUET_DIR = DATA_DIR / "parquet"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = DATA_DIR / "logs"

# Create directories if they don't exist
for dir_path in [PARQUET_DIR, CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Application settings
APP_NAME = "Trading Pattern Detection"
APP_VERSION = "2.0.0"  # Updated version for enhanced system

# Data management settings
DATA_FETCH_TIMEOUT = 30  # seconds
CACHE_TTL = 3600  # 1 hour in seconds
MAX_PARQUET_SIZE = 100000  # Maximum rows per parquet file
DATA_UPDATE_INTERVAL = 86400  # 24 hours in seconds

# Enhanced backend settings
BACKEND_ENABLED = True
BACKEND_CONFIG_PATH = SRC_DIR / "data" / "config"
BACKEND_CONFIG_FILE = BACKEND_CONFIG_PATH / "backend_config.json"

# Async processing settings
MAX_CONCURRENT_REQUESTS = 10
ASYNC_TIMEOUT = 60  # seconds
PARALLEL_PROCESSING_WORKERS = 4
MEMORY_CACHE_SIZE = 1000  # Maximum cached items

# Performance monitoring
ENABLE_PERFORMANCE_TRACKING = True
PERFORMANCE_LOG_INTERVAL = 300  # seconds
MEMORY_MONITORING_INTERVAL = 60  # seconds

# UI settings
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2024-01-01"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MAX_RESULTS_DISPLAY = 100
PROGRESS_UPDATE_INTERVAL = 0.5  # seconds

# Pattern settings
SUPPORTED_PATTERNS = [
    "VCP_BREAKOUT",
    "FLAG_PATTERN",
    "CUP_HANDLE",
    "DOUBLE_BOTTOM",
    "HEAD_AND_SHOULDERS",
    "ROUNDING_BOTTOM",
    "ASCENDING_TRIANGLE",
    "DESCENDING_TRIANGLE",
    "RISING_WEDGE",
    "FALLING_WEDGE"
]

# yfinance settings
YFINANCE_RETRY_ATTEMPTS = 3
YFINANCE_RETRY_DELAY = 1  # seconds
YFINANCE_RATE_LIMIT = 2  # requests per second

# Streamlit settings
PAGE_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Color scheme
COLORS = {
    "primary": "#1E88E5",
    "secondary": "#43A047",
    "accent": "#FF6B35",
    "background": "#F5F5F5",
    "text": "#333333",
    "success": "#4CAF50",
    "error": "#F44336",
    "warning": "#FF9800",
    "info": "#2196F3"
}

# Backend configuration
BACKEND_CONFIG = {
    "enabled": BACKEND_ENABLED,
    "data_sources": ["yfinance"],
    "cache_config": {
        "memory_cache_size": MEMORY_CACHE_SIZE,
        "cache_ttl": CACHE_TTL,
        "enable_disk_cache": True,
        "parquet_compression": "snappy"
    },
    "async_config": {
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "async_timeout": ASYNC_TIMEOUT,
        "parallel_workers": PARALLEL_PROCESSING_WORKERS
    },
    "performance_config": {
        "enable_tracking": ENABLE_PERFORMANCE_TRACKING,
        "log_interval": PERFORMANCE_LOG_INTERVAL,
        "monitor_memory": True
    },
    "validation_config": {
        "enable_strict_validation": True,
        "data_quality_threshold": 0.95,
        "handle_nan_values": "interpolate"
    }
}

class BackendConfigManager:
    """Configuration manager for enhanced backend integration"""

    def __init__(self):
        self.config = BACKEND_CONFIG.copy()
        self._load_backend_config()

    def _load_backend_config(self):
        """Load backend configuration from file if it exists"""
        try:
            if BACKEND_CONFIG_FILE.exists():
                with open(BACKEND_CONFIG_FILE, 'r') as f:
                    file_config = json.load(f)
                    # Merge file config with default config
                    self._merge_config(self.config, file_config)
        except Exception as e:
            print(f"Warning: Could not load backend config file: {e}")

    def _merge_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]):
        """Recursively merge configuration dictionaries"""
        for key, value in override_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save_to_file(self):
        """Save current configuration to file"""
        try:
            BACKEND_CONFIG_PATH.mkdir(parents=True, exist_ok=True)
            with open(BACKEND_CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving backend config: {e}")

# Global configuration instance
config_manager = BackendConfigManager()

# Get enhanced settings based on backend configuration
ENABLE_ENHANCED_MODE = config_manager.get('enabled', True)
MAX_CONCURRENT_REQUESTS = config_manager.get('async_config.max_concurrent_requests', 10)
MEMORY_CACHE_SIZE = config_manager.get('cache_config.memory_cache_size', 1000)
ENABLE_PERFORMANCE_TRACKING = config_manager.get('performance_config.enable_tracking', True)