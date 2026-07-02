"""
Multi-tracker comparator - compare every tracker report of two versions.

Discovers all tracker report directories sharing a version prefix
(e.g. 'v1.1.0' -> 'v1.1.0-tracker-compare-botsort-reid-off', ...) and
compares each tracker present in both versions. Trackers or sequences
present in only one version are reported with missing values instead of
being dropped or compared.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .comparator import TrackEvalComparator
from .config import MetricsConfig, DEFAULT_CONFIG
from .utils import (
    discover_tracker_reports,
    is_higher_better,
    list_available_versions,
    load_trackeval_csv,
)


class MultiTrackerComparator:
    """
    Compare all tracker reports between two version prefixes.

    Attributes:
        version1: First version prefix (e.g., 'v1.1.0')
        version2: Second version prefix
        config: MetricsConfig defining metrics to compare
        results: Per-tracker result dicts (populated by compare())
    """

    def __init__(
        self,
        version1: str,
        version2: str,
        config: Optional[MetricsConfig] = None,
        base_dir: str = "output/reports",
        trackers: Optional[List[str]] = None
    ):
        """
        Args:
            version1: First version prefix
            version2: Second version prefix
            config: MetricsConfig instance (uses DEFAULT_CONFIG if None)
            base_dir: Base directory for reports
            trackers: Optional tracker-name filter (subset of discovered names)

        Raises:
            FileNotFoundError: If neither version matches any report directory
        """
        self.version1 = version1
        self.version2 = version2
        self.config = config or DEFAULT_CONFIG
        self.base_dir = base_dir

        self.reports1 = discover_tracker_reports(version1, base_dir)
        self.reports2 = discover_tracker_reports(version2, base_dir)

        if not self.reports1 and not self.reports2:
            available = list_available_versions(base_dir)
            raise FileNotFoundError(
                f"No tracker reports found for '{version1}' or '{version2}' in {base_dir}.\n"
                f"Available report directories: {', '.join(available) or '(none)'}"
            )

        if trackers:
            requested = set(trackers)
            known = set(self.reports1) | set(self.reports2)
            unknown = requested - known
            if unknown:
                raise ValueError(
                    f"Unknown tracker names: {sorted(unknown)}. "
                    f"Discovered trackers: {sorted(known)}"
                )
            self.reports1 = {k: v for k, v in self.reports1.items() if k in requested}
            self.reports2 = {k: v for k, v in self.reports2.items() if k in requested}

        self.common_trackers = sorted(set(self.reports1) & set(self.reports2))
        self.only_in_v1 = sorted(set(self.reports1) - set(self.reports2))
        self.only_in_v2 = sorted(set(self.reports2) - set(self.reports1))

        # Populated by compare(): tracker name -> result dict
        self.results: Dict[str, Dict] = {}

    def compare(self) -> Dict[str, Dict]:
        """
        Run the comparison for every discovered tracker.

        Each result dict contains:
            status: 'both' | 'only_v1' | 'only_v2'
            comparison: DataFrame with columns
                (sequence, metric, v1_value, v2_value, delta, delta_pct, improved);
                for one-sided trackers the missing side is NaN and delta/improved
                are NaN/None
            seq_only_in_v1 / seq_only_in_v2: sequence-level mismatches ('both' only)

        Returns:
            Dict of tracker name to result dict, ordered common-first
        """
        self.results = {}

        for tracker in self.common_trackers:
            comparator = TrackEvalComparator(
                self.version1,
                self.version2,
                config=self.config,
                base_dir=self.base_dir,
                csv_path1=self.reports1[tracker],
                csv_path2=self.reports2[tracker],
            )
            comparator.df1 = load_trackeval_csv(comparator.csv_path1)
            comparator.df2 = load_trackeval_csv(comparator.csv_path2)

            # Sequence mismatches are reported by the multi reporter; the
            # per-comparator warning would repeat for every tracker.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                comparator.validate_data(mode='union')
                comparator.compare(mode='union')

            self.results[tracker] = {
                'status': 'both',
                'comparison': comparator.comparison,
                'seq_only_in_v1': comparator.only_in_v1,
                'seq_only_in_v2': comparator.only_in_v2,
            }

        for tracker, side in [(t, 'only_v1') for t in self.only_in_v1] + \
                             [(t, 'only_v2') for t in self.only_in_v2]:
            csv_path = self.reports1[tracker] if side == 'only_v1' else self.reports2[tracker]
            self.results[tracker] = {
                'status': side,
                'comparison': self._one_sided_comparison(csv_path, side),
                'seq_only_in_v1': [],
                'seq_only_in_v2': [],
            }

        return self.results

    def _one_sided_comparison(self, csv_path: Path, side: str) -> pd.DataFrame:
        """Build a comparison-shaped DataFrame from a single version's CSV."""
        df = load_trackeval_csv(csv_path)
        metrics = [m for m in self.config.get_metrics() if m in df.columns]

        rows = []
        for seq in sorted(df.index):
            for metric in metrics:
                value = float(df.loc[seq, metric])
                rows.append({
                    'sequence': seq,
                    'metric': metric,
                    'v1_value': value if side == 'only_v1' else float('nan'),
                    'v2_value': value if side == 'only_v2' else float('nan'),
                    'delta': float('nan'),
                    'delta_pct': float('nan'),
                    'improved': None,
                })
        return pd.DataFrame(rows)

    def get_summary(self) -> Dict:
        """
        Summarize the multi-tracker comparison.

        Returns:
            Dict with tracker counts, mismatch lists and per-tracker COMBINED
            key metrics.

        Raises:
            ValueError: If compare() has not been run
        """
        if not self.results:
            raise ValueError("Comparison not performed. Call compare() first.")

        improvements = 0
        regressions = 0
        for result in self.results.values():
            combined = result['comparison'][result['comparison']['sequence'] == 'COMBINED']
            improvements += int((combined['improved'] == True).sum())
            regressions += int((combined['improved'] == False).sum())

        return {
            'version1': self.version1,
            'version2': self.version2,
            'trackers_compared': len(self.common_trackers),
            'common_trackers': self.common_trackers,
            'only_in_v1': self.only_in_v1,
            'only_in_v2': self.only_in_v2,
            'total_metrics_compared': len(self.config.get_metrics()),
            'improvements': improvements,
            'regressions': regressions,
        }
