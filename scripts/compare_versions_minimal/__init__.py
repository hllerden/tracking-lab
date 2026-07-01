"""
TrackEval Benchmark Comparison Tool

A comprehensive tool for comparing TrackEval benchmark results across versions
with support for multiple metrics, visualizations, and report formats.

Main Components:
    - TrackEvalComparator: Core comparison logic
    - MetricsConfig: Configurable metric selection
    - ChartGenerator: Visualization generation
    - ReportGenerator: Multi-format report generation

Usage:
    # As a command-line tool
    python -m scripts.compare_versions_minimal v1.0.1-minimal v1.0.2-minimal

    # As a Python module
    from scripts.compare_versions_minimal import TrackEvalComparator, MetricsConfig

    config = MetricsConfig(groups=['primary', 'counts'])
    comparator = TrackEvalComparator('v1.0.1-minimal', 'v1.0.2-minimal', config)
    comparator.load_data()
    comparator.compare()
    results = comparator.get_summary()
"""

from .comparator import TrackEvalComparator
from .config import (
    MetricsConfig,
    DEFAULT_CONFIG,
    FULL_CONFIG,
    DETECTION_CONFIG,
    ASSOCIATION_CONFIG
)
from .visualizer import ChartGenerator
from .reporter import ReportGenerator
from .utils import (
    find_report_file,
    list_available_versions,
    load_trackeval_csv,
    get_sequence_diff,
    format_percentage,
    format_delta,
    is_higher_better,
    get_metric_display_name
)

__version__ = '1.0.0'
__author__ = 'opencv-yolo project'
__all__ = [
    'TrackEvalComparator',
    'MetricsConfig',
    'ChartGenerator',
    'ReportGenerator',
    'DEFAULT_CONFIG',
    'FULL_CONFIG',
    'DETECTION_CONFIG',
    'ASSOCIATION_CONFIG',
    'find_report_file',
    'list_available_versions',
    'load_trackeval_csv',
    'get_sequence_diff',
    'format_percentage',
    'format_delta',
    'is_higher_better',
    'get_metric_display_name',
]
