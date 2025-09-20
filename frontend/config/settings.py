"""
Frontend configuration settings
"""

import os
from pathlib import Path

# Application paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data"  # Store data in parent data/ folder
STATIC_DIR = BASE_DIR / "static"

# Data storage configuration
PARQUET_DIR = DATA_DIR / "parquet"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = DATA_DIR / "logs"

# Create directories if they don't exist
for dir_path in [PARQUET_DIR, CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Application settings
APP_NAME = "Trading Pattern Detection"
APP_VERSION = "1.0.0"

# Data management settings
DATA_FETCH_TIMEOUT = 30  # seconds
CACHE_TTL = 3600  # 1 hour in seconds
MAX_PARQUET_SIZE = 100000  # Maximum rows per parquet file
DATA_UPDATE_INTERVAL = 86400  # 24 hours in seconds

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