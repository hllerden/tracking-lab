"""
Utility functions for TrackEval comparison tool.

Provides helper functions for file operations, data validation,
and formatting.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd


def find_report_file(version: str, base_dir: str = "output/reports") -> Path:
    """
    Find the pedestrian_detailed.csv file for a given version.

    Args:
        version: Version identifier (e.g., 'v1.0.1-minimal')
        base_dir: Base directory for reports (default: 'output/reports')

    Returns:
        Path to the CSV file

    Raises:
        FileNotFoundError: If version directory or CSV file doesn't exist
    """
    # Support both relative and absolute paths
    if os.path.isabs(base_dir):
        report_dir = Path(base_dir)
    else:
        # Get script directory and navigate to project root
        script_dir = Path(__file__).parent.parent.parent
        report_dir = script_dir / base_dir

    version_dir = report_dir / version
    csv_file = version_dir / "pedestrian_detailed.csv"

    # Check if version directory exists
    if not version_dir.exists():
        available_versions = list_available_versions(base_dir)
        raise FileNotFoundError(
            f"Version directory not found: {version_dir}\n"
            f"Available versions: {', '.join(available_versions)}"
        )

    # Check if CSV file exists
    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_file}\n"
            f"Expected file: pedestrian_detailed.csv"
        )

    return csv_file


def list_available_versions(base_dir: str = "output/reports") -> List[str]:
    """
    List all available version directories.

    Args:
        base_dir: Base directory for reports

    Returns:
        List of version names
    """
    if os.path.isabs(base_dir):
        report_dir = Path(base_dir)
    else:
        script_dir = Path(__file__).parent.parent.parent
        report_dir = script_dir / base_dir

    if not report_dir.exists():
        return []

    versions = [
        d.name for d in report_dir.iterdir()
        if d.is_dir() and (d / "pedestrian_detailed.csv").exists()
    ]

    return sorted(versions)


def load_trackeval_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load and parse TrackEval CSV file.

    Args:
        csv_path: Path to pedestrian_detailed.csv

    Returns:
        DataFrame with 'seq' as index

    Raises:
        ValueError: If CSV is corrupted or has invalid format
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file {csv_path}: {e}")

    # Validate required column
    if 'seq' not in df.columns:
        raise ValueError(
            f"Invalid CSV format: 'seq' column not found in {csv_path}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    # Set 'seq' as index
    df = df.set_index('seq')

    # Validate that COMBINED row exists
    if 'COMBINED' not in df.index:
        raise ValueError(
            f"CSV missing 'COMBINED' row in {csv_path}\n"
            f"Found sequences: {df.index.tolist()}"
        )

    return df


def get_sequence_diff(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Find differences in sequences between two DataFrames.

    Args:
        df1: First DataFrame (with 'seq' as index)
        df2: Second DataFrame (with 'seq' as index)

    Returns:
        Tuple of (common_sequences, only_in_df1, only_in_df2)
    """
    # Remove 'COMBINED' for comparison (it's always present)
    seqs1 = set(df1.index) - {'COMBINED'}
    seqs2 = set(df2.index) - {'COMBINED'}

    common = sorted(list(seqs1 & seqs2))
    only_in_1 = sorted(list(seqs1 - seqs2))
    only_in_2 = sorted(list(seqs2 - seqs1))

    return common, only_in_1, only_in_2


def format_percentage(value: float, precision: int = 2, show_sign: bool = True) -> str:
    """
    Format a value as a percentage string.

    Args:
        value: Value to format (e.g., 0.1234 for 12.34%)
        precision: Number of decimal places
        show_sign: Whether to show + sign for positive values

    Returns:
        Formatted percentage string
    """
    percent = value * 100
    sign = '+' if show_sign and percent > 0 else ''
    return f"{sign}{percent:.{precision}f}%"


def format_delta(value: float, precision: int = 4, show_sign: bool = True) -> str:
    """
    Format a delta value with appropriate precision.

    Args:
        value: Delta value
        precision: Number of decimal places
        show_sign: Whether to show + sign for positive values

    Returns:
        Formatted delta string
    """
    sign = '+' if show_sign and value > 0 else ''
    return f"{sign}{value:.{precision}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is zero

    Returns:
        Result of division or default value
    """
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def get_metric_display_name(metric: str) -> str:
    """
    Get a human-readable display name for a metric.

    Args:
        metric: Metric name (e.g., 'HOTA(0)', 'CLR_Re')

    Returns:
        Display name with description
    """
    # Metric descriptions
    descriptions = {
        'HOTA(0)': 'HOTA (Higher Order Tracking Accuracy)',
        'MOTA': 'MOTA (Multiple Object Tracking Accuracy)',
        'IDF1': 'IDF1 (ID F1 Score)',
        'DetA(0)': 'DetA (Detection Accuracy)',
        'AssA(0)': 'AssA (Association Accuracy)',
        'LocA(0)': 'LocA (Localization Accuracy)',
        'MOTP': 'MOTP (Tracking Precision)',
        'CLR_Re': 'Recall (CLEAR)',
        'CLR_Pr': 'Precision (CLEAR)',
        'CLR_F1': 'F1 Score (CLEAR)',
        'DetRe(0)': 'Detection Recall',
        'DetPr(0)': 'Detection Precision',
        'AssRe(0)': 'Association Recall',
        'AssPr(0)': 'Association Precision',
        'IDR': 'ID Recall',
        'IDP': 'ID Precision',
        'Dets': 'Total Detections',
        'GT_Dets': 'Ground Truth Detections',
        'IDs': 'Tracked IDs',
        'GT_IDs': 'Ground Truth IDs',
        'IDSW': 'ID Switches',
        'Frag': 'Fragmentations',
        'CLR_TP': 'True Positives',
        'CLR_FN': 'False Negatives',
        'CLR_FP': 'False Positives',
        'MT': 'Mostly Tracked',
        'PT': 'Partially Tracked',
        'ML': 'Mostly Lost',
        'MTR': 'Mostly Tracked Ratio',
        'PTR': 'Partially Tracked Ratio',
        'MLR': 'Mostly Lost Ratio',
        'sMOTA': 'Soft MOTA',
    }

    return descriptions.get(metric, metric)


def is_higher_better(metric: str) -> bool:
    """
    Determine if higher values are better for a given metric.

    Args:
        metric: Metric name

    Returns:
        True if higher is better, False otherwise
    """
    # Metrics where lower is better
    lower_better = {
        'CLR_FN',   # False Negatives
        'CLR_FP',   # False Positives
        'IDSW',     # ID Switches
        'Frag',     # Fragmentations
        'ML',       # Mostly Lost
        'MLR',      # Mostly Lost Ratio
        'FP_per_frame',  # FP per frame
    }

    return metric not in lower_better


def create_output_directory(output_dir: str) -> Path:
    """
    Create output directory if it doesn't exist.

    Args:
        output_dir: Path to output directory

    Returns:
        Path object for the directory
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
