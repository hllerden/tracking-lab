"""
TrackEval Comparator - Main comparison logic.

This module provides the TrackEvalComparator class that handles loading,
validating, and comparing TrackEval benchmark results across versions.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from .config import MetricsConfig, DEFAULT_CONFIG
from .utils import (
    find_report_file,
    load_trackeval_csv,
    get_sequence_diff,
    safe_divide,
    is_higher_better
)


class TrackEvalComparator:
    """
    Main comparator class for TrackEval benchmark results.

    This class handles loading CSV files from two versions, validating data,
    calculating differences, and identifying improvements/regressions.

    Attributes:
        version1: First version identifier
        version2: Second version identifier
        config: MetricsConfig instance defining which metrics to compare
        df1: DataFrame for version 1 (loaded via load_data())
        df2: DataFrame for version 2 (loaded via load_data())
        comparison: DataFrame with comparison results (created via compare())
    """

    def __init__(
        self,
        version1: str,
        version2: str,
        config: Optional[MetricsConfig] = None,
        base_dir: str = "output/reports"
    ):
        """
        Initialize the comparator.

        Args:
            version1: First version identifier (e.g., 'v1.0.1-minimal')
            version2: Second version identifier (e.g., 'v1.0.2-minimal')
            config: MetricsConfig instance (uses DEFAULT_CONFIG if None)
            base_dir: Base directory for reports

        Raises:
            FileNotFoundError: If version directories or CSV files not found
        """
        self.version1 = version1
        self.version2 = version2
        self.config = config or DEFAULT_CONFIG
        self.base_dir = base_dir

        # Find CSV files
        self.csv_path1 = find_report_file(version1, base_dir)
        self.csv_path2 = find_report_file(version2, base_dir)

        # Data storage (populated by load_data())
        self.df1: Optional[pd.DataFrame] = None
        self.df2: Optional[pd.DataFrame] = None
        self.comparison: Optional[pd.DataFrame] = None

        # Sequence information (populated by validate_data())
        self.common_sequences: List[str] = []
        self.only_in_v1: List[str] = []
        self.only_in_v2: List[str] = []

    def load_data(self):
        """
        Load CSV data from both versions.

        Raises:
            ValueError: If CSV files are corrupted or have invalid format
        """
        print(f"Loading {self.version1}...")
        self.df1 = load_trackeval_csv(self.csv_path1)

        print(f"Loading {self.version2}...")
        self.df2 = load_trackeval_csv(self.csv_path2)

        print(f"✓ Loaded {len(self.df1)} sequences from {self.version1}")
        print(f"✓ Loaded {len(self.df2)} sequences from {self.version2}")

    def validate_data(self, mode: str = 'intersection') -> bool:
        """
        Validate data and check for sequence mismatches.

        Args:
            mode: How to handle sequence mismatches
                  - 'intersection': Only compare common sequences (default)
                  - 'union': Include all sequences (NaN for missing)
                  - 'strict': Raise error if sequences don't match

        Returns:
            True if validation passed

        Raises:
            ValueError: If data not loaded or validation fails in strict mode
        """
        if self.df1 is None or self.df2 is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Find sequence differences
        common, only_v1, only_v2 = get_sequence_diff(self.df1, self.df2)

        self.common_sequences = common
        self.only_in_v1 = only_v1
        self.only_in_v2 = only_v2

        # Report differences
        if only_v1 or only_v2:
            warnings.warn(
                f"\nSequence mismatch detected:\n"
                f"  Common sequences: {len(common)}\n"
                f"  Only in {self.version1}: {only_v1}\n"
                f"  Only in {self.version2}: {only_v2}\n"
            )

            if mode == 'strict':
                raise ValueError(
                    "Sequence mismatch in strict mode. "
                    "Use mode='intersection' or 'union' to handle mismatches."
                )

        # Check for missing metrics
        metrics = self.config.get_metrics()
        missing_v1 = set(metrics) - set(self.df1.columns)
        missing_v2 = set(metrics) - set(self.df2.columns)

        if missing_v1:
            warnings.warn(
                f"Metrics not found in {self.version1}: {missing_v1}\n"
                f"These metrics will be skipped."
            )

        if missing_v2:
            warnings.warn(
                f"Metrics not found in {self.version2}: {missing_v2}\n"
                f"These metrics will be skipped."
            )

        return True

    def compare(self, mode: str = 'intersection') -> pd.DataFrame:
        """
        Compare metrics between versions and calculate differences.

        Args:
            mode: Comparison mode ('intersection', 'union', or 'strict')

        Returns:
            DataFrame with comparison results containing:
                - v1_value: Metric value from version 1
                - v2_value: Metric value from version 2
                - delta: Absolute difference (v2 - v1)
                - delta_pct: Percentage change
                - improved: Boolean indicating if metric improved

        Raises:
            ValueError: If data not loaded or validated
        """
        if self.df1 is None or self.df2 is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Validate if not already done
        if not self.common_sequences:
            self.validate_data(mode=mode)

        # Determine sequences to compare
        if mode == 'intersection':
            sequences = self.common_sequences + ['COMBINED']
        elif mode == 'union':
            sequences = sorted(list(set(self.df1.index) | set(self.df2.index)))
        else:  # strict mode already handled in validate_data()
            sequences = self.common_sequences + ['COMBINED']

        # Get metrics that exist in both DataFrames
        available_metrics = self.config.get_metrics()
        metrics = [m for m in available_metrics
                   if m in self.df1.columns and m in self.df2.columns]

        if not metrics:
            raise ValueError(
                f"No common metrics found between versions.\n"
                f"Requested: {available_metrics}\n"
                f"Available in {self.version1}: {list(self.df1.columns)}\n"
                f"Available in {self.version2}: {list(self.df2.columns)}"
            )

        # Build comparison DataFrame
        results = []

        for seq in sequences:
            for metric in metrics:
                # Get values (handle missing sequences)
                v1_val = self.df1.loc[seq, metric] if seq in self.df1.index else float('nan')
                v2_val = self.df2.loc[seq, metric] if seq in self.df2.index else float('nan')

                # Calculate delta
                delta = v2_val - v1_val

                # Calculate percentage change
                delta_pct = safe_divide(delta, v1_val, default=float('nan')) * 100

                # Determine if improved
                higher_better = is_higher_better(metric)
                improved = (delta > 0) if higher_better else (delta < 0)

                results.append({
                    'sequence': seq,
                    'metric': metric,
                    'v1_value': v1_val,
                    'v2_value': v2_val,
                    'delta': delta,
                    'delta_pct': delta_pct,
                    'improved': improved
                })

        self.comparison = pd.DataFrame(results)
        return self.comparison

    def get_improvements(self, threshold: float = 0.0) -> pd.DataFrame:
        """
        Get metrics that improved from v1 to v2.

        Args:
            threshold: Minimum absolute delta to consider (filters noise)

        Returns:
            DataFrame with only improved metrics

        Raises:
            ValueError: If comparison not performed yet
        """
        if self.comparison is None:
            raise ValueError("Comparison not performed. Call compare() first.")

        improved = self.comparison[
            (self.comparison['improved'] == True) &
            (self.comparison['delta'].abs() > threshold)
        ]

        return improved.sort_values('delta_pct', ascending=False)

    def get_regressions(self, threshold: float = 0.0) -> pd.DataFrame:
        """
        Get metrics that regressed from v1 to v2.

        Args:
            threshold: Minimum absolute delta to consider (filters noise)

        Returns:
            DataFrame with only regressed metrics

        Raises:
            ValueError: If comparison not performed yet
        """
        if self.comparison is None:
            raise ValueError("Comparison not performed. Call compare() first.")

        regressed = self.comparison[
            (self.comparison['improved'] == False) &
            (self.comparison['delta'].abs() > threshold)
        ]

        return regressed.sort_values('delta_pct', ascending=True)

    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of the comparison.

        Returns:
            Dictionary with summary statistics

        Raises:
            ValueError: If comparison not performed yet
        """
        if self.comparison is None:
            raise ValueError("Comparison not performed. Call compare() first.")

        # Filter for COMBINED row for overall metrics
        combined = self.comparison[self.comparison['sequence'] == 'COMBINED']

        summary = {
            'version1': self.version1,
            'version2': self.version2,
            'total_sequences_compared': len(self.common_sequences),
            'total_metrics_compared': len(self.config.get_metrics()),
            'improvements': int(combined['improved'].sum()),
            'regressions': int((~combined['improved']).sum()),
            'sequences_only_in_v1': self.only_in_v1,
            'sequences_only_in_v2': self.only_in_v2,
        }

        # Add key metric comparisons from COMBINED row
        key_metrics = ['HOTA(0)', 'MOTA', 'IDF1']
        for metric in key_metrics:
            if metric in combined['metric'].values:
                row = combined[combined['metric'] == metric].iloc[0]
                summary[f'{metric}_v1'] = float(row['v1_value'])
                summary[f'{metric}_v2'] = float(row['v2_value'])
                summary[f'{metric}_delta'] = float(row['delta'])
                summary[f'{metric}_delta_pct'] = float(row['delta_pct'])

        return summary

    def __repr__(self) -> str:
        """String representation of the comparator."""
        status = "loaded" if self.df1 is not None else "not loaded"
        compared = "compared" if self.comparison is not None else "not compared"
        return (
            f"TrackEvalComparator({self.version1} vs {self.version2}, "
            f"{status}, {compared})"
        )
