from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import yaml
import os
import logging


@dataclass
class MarketDataConfig:
    """Configuration for market data settings"""

    # Data source configuration
    default_source: str = "yahoo"
    timeframes: List[str] = None
    default_period: str = "1y"
    cache_enabled: bool = True
    cache_duration: int = 3600  # 1 hour in seconds

    # Data quality settings
    min_data_points: int = 100
    max_outlier_threshold: float = 3.0
    volume_fill_method: str = "median"

    # API settings
    api_key: Optional[str] = None
    rate_limit_calls: int = 5
    rate_limit_period: int = 60  # seconds

    # Retry settings
    max_retries: int = 3
    retry_delay: int = 1  # seconds

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketDataConfig":
        """Create from dictionary"""
        return cls(
            default_source=data.get("default_source", "yahoo"),
            timeframes=data.get(
                "timeframes", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
            ),
            default_period=data.get("default_period", "1y"),
            cache_enabled=data.get("cache_enabled", True),
            cache_duration=data.get("cache_duration", 3600),
            min_data_points=data.get("min_data_points", 100),
            max_outlier_threshold=data.get("max_outlier_threshold", 3.0),
            volume_fill_method=data.get("volume_fill_method", "median"),
            api_key=data.get("api_key"),
            rate_limit_calls=data.get("rate_limit_calls", 5),
            rate_limit_period=data.get("rate_limit_period", 60),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "default_source": self.default_source,
            "timeframes": self.timeframes,
            "default_period": self.default_period,
            "cache_enabled": self.cache_enabled,
            "cache_duration": self.cache_duration,
            "min_data_points": self.min_data_points,
            "max_outlier_threshold": self.max_outlier_threshold,
            "volume_fill_method": self.volume_fill_method,
            "api_key": self.api_key,
            "rate_limit_calls": self.rate_limit_calls,
            "rate_limit_period": self.rate_limit_period,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }

    def save_to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        try:
            with open(filepath, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving market data config to {filepath}: {e}")
            raise

    @classmethod
    def load_from_yaml(cls, filepath: str) -> "MarketDataConfig":
        """Load configuration from YAML file"""
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except Exception as e:
            logging.error(f"Error loading market data config from {filepath}: {e}")
            # Return default configuration if file doesn't exist
            return cls()

    def validate(self) -> List[str]:
        """Validate configuration parameters"""
        errors = []

        # Validate data source
        if self.default_source not in ["yahoo", "alpha_vantage", "quandl"]:
            errors.append(f"Invalid data source: {self.default_source}")

        # Validate timeframes
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
        for timeframe in self.timeframes:
            if timeframe not in valid_timeframes:
                errors.append(f"Invalid timeframe: {timeframe}")

        # Validate cache duration
        if self.cache_duration <= 0:
            errors.append("Cache duration must be positive")

        # Validate rate limiting
        if self.rate_limit_calls <= 0:
            errors.append("Rate limit calls must be positive")

        if self.rate_limit_period <= 0:
            errors.append("Rate limit period must be positive")

        # Validate retry settings
        if self.max_retries < 0:
            errors.append("Max retries must be non-negative")

        if self.retry_delay < 0:
            errors.append("Retry delay must be non-negative")

        return errors
