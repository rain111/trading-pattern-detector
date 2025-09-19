from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml
import os
import logging
from typing import List


@dataclass
class PatternParameters:
    """Configuration for pattern detection parameters"""

    # VCP Detector Parameters
    vcp_detector: Dict[str, Any] = None

    # Flag Detector Parameters
    flag_detector: Dict[str, Any] = None

    # Triangle Detector Parameters
    triangle_detector: Dict[str, Any] = None

    # Wedge Detector Parameters
    wedge_detector: Dict[str, Any] = None

    def __post_init__(self):
        if self.vcp_detector is None:
            self.vcp_detector = self._get_default_vcp_params()

        if self.flag_detector is None:
            self.flag_detector = self._get_default_flag_params()

        if self.triangle_detector is None:
            self.triangle_detector = self._get_default_triangle_params()

        if self.wedge_detector is None:
            self.wedge_detector = self._get_default_wedge_params()

    @staticmethod
    def _get_default_vcp_params() -> Dict[str, Any]:
        """Default VCP detector parameters"""
        return {
            "initial_decline_period": 20,
            "max_decline_threshold": -0.15,
            "volume_threshold": 1000000,
            "contraction_period": 30,
            "volatility_threshold": 0.001,
            "consolidation_period": 25,
            "max_consolidation_range": 0.05,
            "min_boundary_touches": 3,
            "breakout_period": 10,
            "volume_spike_threshold": 2.0,
            "reward_ratio": 2.0,
            "confidence_threshold": 0.7,
            "min_volatility_contraction": 0.15,
            "support_test_threshold": 0.98,
            "resistance_test_threshold": 1.02,
        }

    @staticmethod
    def _get_default_flag_params() -> Dict[str, Any]:
        """Default flag detector parameters"""
        return {
            "flagpole_min_length": 0.08,
            "flag_max_duration": 20,
            "flag_min_duration": 5,
            "volume_threshold": 1.5,
            "reward_ratio": 1.5,
            "confidence_threshold": 0.6,
            "max_flag_range": 0.04,
            "min_volume_decrease": 0.2,
            "breakout_threshold": 0.01,
            "slope_threshold": 0.001,
        }

    @staticmethod
    def _get_default_triangle_params() -> Dict[str, Any]:
        """Default triangle detector parameters"""
        return {
            "triangle_min_length": 20,
            "triangle_max_length": 40,
            "triangle_min_swing_points": 3,
            "volume_threshold": 1.3,
            "reward_ratio": 1.8,
            "confidence_threshold": 0.6,
            "max_range_ratio": 0.04,
            "min_convergence": 0.1,
            "breakout_threshold": 0.01,
            "volume_decline_threshold": 0.8,
        }

    @staticmethod
    def _get_default_wedge_params() -> Dict[str, Any]:
        """Default wedge detector parameters"""
        return {
            "wedge_min_length": 40,
            "wedge_max_length": 60,
            "wedge_min_swing_points": 4,
            "volume_threshold": 1.4,
            "reward_ratio": 1.6,
            "confidence_threshold": 0.5,
            "min_angle_difference": 0.001,
            "max_angle_difference": 0.01,
            "volume_decline_threshold": 0.8,
            "breakout_threshold": 0.015,
            "slope_difference_threshold": 0.002,
        }

    def get_detector_params(self, detector_name: str) -> Dict[str, Any]:
        """Get parameters for a specific detector"""
        if detector_name == "vcp":
            return self.vcp_detector
        elif detector_name == "flag":
            return self.flag_detector
        elif detector_name == "triangle":
            return self.triangle_detector
        elif detector_name == "wedge":
            return self.wedge_detector
        else:
            raise ValueError(f"Unknown detector: {detector_name}")

    def update_detector_params(
        self, detector_name: str, params: Dict[str, Any]
    ) -> None:
        """Update parameters for a specific detector"""
        if detector_name == "vcp":
            self.vcp_detector.update(params)
        elif detector_name == "flag":
            self.flag_detector.update(params)
        elif detector_name == "triangle":
            self.triangle_detector.update(params)
        elif detector_name == "wedge":
            self.wedge_detector.update(params)
        else:
            raise ValueError(f"Unknown detector: {detector_name}")

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to dictionary for serialization"""
        return {
            "vcp_detector": self.vcp_detector,
            "flag_detector": self.flag_detector,
            "triangle_detector": self.triangle_detector,
            "wedge_detector": self.wedge_detector,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> "PatternParameters":
        """Create from dictionary"""
        return cls(
            vcp_detector=data.get("vcp_detector", cls._get_default_vcp_params()),
            flag_detector=data.get("flag_detector", cls._get_default_flag_params()),
            triangle_detector=data.get(
                "triangle_detector", cls._get_default_triangle_params()
            ),
            wedge_detector=data.get("wedge_detector", cls._get_default_wedge_params()),
        )

    def save_to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        try:
            with open(filepath, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving pattern parameters to {filepath}: {e}")
            raise

    @classmethod
    def load_from_yaml(cls, filepath: str) -> "PatternParameters":
        """Load configuration from YAML file"""
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except Exception as e:
            logging.error(f"Error loading pattern parameters from {filepath}: {e}")
            # Return default configuration if file doesn't exist
            return cls()

    def validate(self) -> List[str]:
        """Validate configuration parameters"""
        errors = []

        # Validate VCP parameters
        vcp_params = self.vcp_detector
        if vcp_params.get("max_decline_threshold", 0) > vcp_params.get(
            "min_decline_threshold", -1
        ):
            errors.append(
                "VCP max_decline_threshold should be less than min_decline_threshold"
            )

        # Validate Flag parameters
        flag_params = self.flag_detector
        if flag_params.get("flag_max_duration", 0) < flag_params.get(
            "flag_min_duration", 0
        ):
            errors.append("Flag max_duration should be greater than min_duration")

        # Validate Triangle parameters
        triangle_params = self.triangle_detector
        if triangle_params.get("triangle_max_length", 0) < triangle_params.get(
            "triangle_min_length", 0
        ):
            errors.append("Triangle max_length should be greater than min_length")

        # Validate Wedge parameters
        wedge_params = self.wedge_detector
        if wedge_params.get("wedge_max_length", 0) < wedge_params.get(
            "wedge_min_length", 0
        ):
            errors.append("Wedge max_length should be greater than min_length")

        return errors
