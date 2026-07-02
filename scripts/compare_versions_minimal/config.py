"""
Configuration module for TrackEval metrics comparison.

This module defines configurable metric groups and provides easy ways to
add/remove/customize metrics for comparison.
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field


# Predefined metric groups (class-level constant)
METRIC_GROUPS: Dict[str, List[str]] = {
    'primary': [
        'HOTA___AUC',   # Higher Order Tracking Accuracy (averaged over thresholds)
        'MOTA',         # Multiple Object Tracking Accuracy
        'IDF1',         # ID F1 Score
    ],
    'detection': [
        'DetA___AUC',   # Detection Accuracy (averaged over thresholds)
        'DetRe___AUC',  # Detection Recall (averaged over thresholds)
        'DetPr___AUC',  # Detection Precision (averaged over thresholds)
        'CLR_Re',       # CLEAR Recall
        'CLR_Pr',       # CLEAR Precision
    ],
    'association': [
        'AssA___AUC',   # Association Accuracy (averaged over thresholds)
        'AssRe___AUC',  # Association Recall (averaged over thresholds)
        'AssPr___AUC',  # Association Precision (averaged over thresholds)
        'IDR',          # ID Recall
        'IDP',          # ID Precision
    ],
    'localization': [
        'LocA___AUC',   # Localization Accuracy (averaged over thresholds)
        'MOTP',         # Multiple Object Tracking Precision
    ],
    'counts': [
        'Dets',         # Total Detections
        'GT_Dets',      # Ground Truth Detections
        'IDs',          # Number of tracked IDs
        'GT_IDs',       # Ground Truth IDs
        'CLR_TP',       # True Positives
        'CLR_FN',       # False Negatives
        'CLR_FP',       # False Positives
        'IDSW',         # ID Switches
        'Frag',         # Fragmentations
    ],
    'advanced': [
        'sMOTA',        # Soft MOTA
        'CLR_F1',       # CLEAR F1 Score
        'MTR',          # Mostly Tracked Ratio
        'PTR',          # Partially Tracked Ratio
        'MLR',          # Mostly Lost Ratio
        'MT',           # Mostly Tracked
        'PT',           # Partially Tracked
        'ML',           # Mostly Lost
    ],
}


@dataclass
class MetricsConfig:
    """
    Configuration class for metrics to compare.

    Allows easy customization of which metrics to include in comparison
    and visualization. Metrics are organized into groups for better organization.

    Attributes:
        groups: List of metric group names to include (e.g., ['primary', 'detection'])
        custom_metrics: Additional custom metric names not in predefined groups
        exclude_metrics: Specific metrics to exclude even if in selected groups
        include_thresholds: Whether to include threshold-based metrics (HOTA_5 to HOTA_95)
    """

    groups: List[str] = field(default_factory=lambda: ['primary'])
    custom_metrics: List[str] = field(default_factory=list)
    exclude_metrics: List[str] = field(default_factory=list)
    include_thresholds: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate group names
        invalid_groups = set(self.groups) - set(METRIC_GROUPS.keys())
        if invalid_groups:
            raise ValueError(
                f"Invalid metric groups: {invalid_groups}. "
                f"Available groups: {list(METRIC_GROUPS.keys())}"
            )

    def get_metrics(self) -> List[str]:
        """
        Get the final list of metrics based on configuration.

        Returns:
            List of metric names to compare
        """
        metrics = set()

        # Add metrics from selected groups
        for group in self.groups:
            metrics.update(METRIC_GROUPS[group])

        # Add custom metrics
        metrics.update(self.custom_metrics)

        # Add threshold metrics if requested
        if self.include_thresholds:
            metrics.update(self._get_threshold_metrics())

        # Remove excluded metrics
        metrics -= set(self.exclude_metrics)

        return sorted(list(metrics))

    def _get_threshold_metrics(self) -> List[str]:
        """
        Generate threshold-based metric names (e.g., HOTA___5, HOTA___10, ..., HOTA___95).

        Returns:
            List of threshold metric names
        """
        threshold_metrics = []
        base_metrics = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA']

        for base in base_metrics:
            for threshold in range(5, 100, 5):
                threshold_metrics.append(f'{base}___{threshold}')

        return threshold_metrics

    def get_group_metrics(self, group_name: str) -> List[str]:
        """
        Get metrics for a specific group.

        Args:
            group_name: Name of the metric group

        Returns:
            List of metric names in the group

        Raises:
            KeyError: If group_name doesn't exist
        """
        if group_name not in METRIC_GROUPS:
            raise KeyError(
                f"Unknown metric group: {group_name}. "
                f"Available: {list(METRIC_GROUPS.keys())}"
            )
        return METRIC_GROUPS[group_name]

    def add_custom_group(self, group_name: str, metrics: List[str]):
        """
        Add a custom metric group.

        Args:
            group_name: Name for the new group
            metrics: List of metric names in the group
        """
        METRIC_GROUPS[group_name] = metrics

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MetricsConfig':
        """
        Create MetricsConfig from a dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            MetricsConfig instance
        """
        return cls(
            groups=config_dict.get('groups', ['primary']),
            custom_metrics=config_dict.get('custom_metrics', []),
            exclude_metrics=config_dict.get('exclude_metrics', []),
            include_thresholds=config_dict.get('include_thresholds', False)
        )

    @classmethod
    def all_metrics(cls) -> 'MetricsConfig':
        """
        Create a config that includes all available metrics.

        Returns:
            MetricsConfig with all groups enabled
        """
        return cls(
            groups=list(METRIC_GROUPS.keys()),
            include_thresholds=True
        )

    def __repr__(self) -> str:
        """String representation of the configuration."""
        metrics = self.get_metrics()
        return (
            f"MetricsConfig(groups={self.groups}, "
            f"total_metrics={len(metrics)}, "
            f"include_thresholds={self.include_thresholds})"
        )


# Predefined configurations for common use cases
DEFAULT_CONFIG = MetricsConfig(groups=['primary'])
FULL_CONFIG = MetricsConfig.all_metrics()
DETECTION_CONFIG = MetricsConfig(groups=['primary', 'detection', 'counts'])
ASSOCIATION_CONFIG = MetricsConfig(groups=['primary', 'association', 'counts'])
