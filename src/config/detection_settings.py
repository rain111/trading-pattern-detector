from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import yaml
import os
import logging


@dataclass
class DetectionSettings:
    """Configuration for detection settings"""

    # Processing settings
    parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 100

    # Cache settings
    cache_results: bool = True
    cache_ttl: int = 3600  # 1 hour

    # Signal filtering
    confidence_threshold: float = 0.6
    min_volume_threshold: Optional[float] = None
    max_risk_level: Optional[str] = None

    # Ranking and aggregation
    ranking_method: str = "confidence"
    remove_duplicates: bool = True
    duplicate_tolerance: float = 0.02

    # Output settings
    output_format: str = "json"
    include_metadata: bool = True
    include_statistics: bool = True

    # Logging settings
    logging_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "trading_patterns.log"

    # Performance settings
    enable_caching: bool = True
    max_cache_size: int = 1000  # maximum number of cached results
    optimize_memory: bool = True

    def __post_init__(self):
        # Validate logging level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging_level not in valid_levels:
            self.logging_level = "INFO"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionSettings":
        """Create from dictionary"""
        return cls(
            parallel_processing=data.get("parallel_processing", True),
            max_workers=data.get("max_workers", 4),
            batch_size=data.get("batch_size", 100),
            cache_results=data.get("cache_results", True),
            cache_ttl=data.get("cache_ttl", 3600),
            confidence_threshold=data.get("confidence_threshold", 0.6),
            min_volume_threshold=data.get("min_volume_threshold"),
            max_risk_level=data.get("max_risk_level"),
            ranking_method=data.get("ranking_method", "confidence"),
            remove_duplicates=data.get("remove_duplicates", True),
            duplicate_tolerance=data.get("duplicate_tolerance", 0.02),
            output_format=data.get("output_format", "json"),
            include_metadata=data.get("include_metadata", True),
            include_statistics=data.get("include_statistics", True),
            logging_level=data.get("logging_level", "INFO"),
            log_to_file=data.get("log_to_file", True),
            log_file_path=data.get("log_file_path", "trading_patterns.log"),
            enable_caching=data.get("enable_caching", True),
            max_cache_size=data.get("max_cache_size", 1000),
            optimize_memory=data.get("optimize_memory", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "parallel_processing": self.parallel_processing,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "cache_results": self.cache_results,
            "cache_ttl": self.cache_ttl,
            "confidence_threshold": self.confidence_threshold,
            "min_volume_threshold": self.min_volume_threshold,
            "max_risk_level": self.max_risk_level,
            "ranking_method": self.ranking_method,
            "remove_duplicates": self.remove_duplicates,
            "duplicate_tolerance": self.duplicate_tolerance,
            "output_format": self.output_format,
            "include_metadata": self.include_metadata,
            "include_statistics": self.include_statistics,
            "logging_level": self.logging_level,
            "log_to_file": self.log_to_file,
            "log_file_path": self.log_file_path,
            "enable_caching": self.enable_caching,
            "max_cache_size": self.max_cache_size,
            "optimize_memory": self.optimize_memory,
        }

    def save_to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        try:
            with open(filepath, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving detection settings to {filepath}: {e}")
            raise

    @classmethod
    def load_from_yaml(cls, filepath: str) -> "DetectionSettings":
        """Load configuration from YAML file"""
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except Exception as e:
            logging.error(f"Error loading detection settings from {filepath}: {e}")
            # Return default configuration if file doesn't exist
            return cls()

    def validate(self) -> List[str]:
        """Validate configuration parameters"""
        errors = []

        # Validate parallel processing settings
        if self.max_workers <= 0:
            errors.append("Max workers must be positive")

        if self.batch_size <= 0:
            errors.append("Batch size must be positive")

        # Validate confidence threshold
        if not 0 <= self.confidence_threshold <= 1:
            errors.append("Confidence threshold must be between 0 and 1")

        # Validate duplicate tolerance
        if not 0 <= self.duplicate_tolerance <= 1:
            errors.append("Duplicate tolerance must be between 0 and 1")

        # Validate cache settings
        if self.cache_ttl <= 0:
            errors.append("Cache TTL must be positive")

        if self.max_cache_size <= 0:
            errors.append("Max cache size must be positive")

        # Validate risk level
        valid_risk_levels = ["low", "medium", "high"]
        if self.max_risk_level and self.max_risk_level not in valid_risk_levels:
            errors.append(f"Invalid risk level: {self.max_risk_level}")

        # Validate output format
        valid_formats = ["json", "dict", "dataframe"]
        if self.output_format not in valid_formats:
            errors.append(f"Invalid output format: {self.output_format}")

        # Validate ranking method
        valid_methods = [
            "confidence",
            "risk_reward",
            "strength",
            "probability",
            "reward_ratio",
        ]
        if self.ranking_method not in valid_methods:
            errors.append(f"Invalid ranking method: {self.ranking_method}")

        return errors
