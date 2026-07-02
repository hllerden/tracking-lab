"""
Chart Generator - Visualization module for comparison results.

This module provides the ChartGenerator class for creating various
visualizations of benchmark comparison results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

# Import visualization libraries with graceful fallback
try:
    warnings.filterwarnings(
        "ignore",
        message="Unable to import Axes3D.*",
        category=UserWarning,
        module="matplotlib.projections"
    )
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None  # Set to None for type hints
    warnings.warn("matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    if SEABORN_AVAILABLE and MATPLOTLIB_AVAILABLE:
        sns.set_theme(style="whitegrid")
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None  # Set to None for type hints
    warnings.warn("seaborn not available. Install with: pip install seaborn")

from .utils import is_higher_better, get_metric_display_name

# Type hint for matplotlib Figure
if MATPLOTLIB_AVAILABLE:
    from matplotlib.figure import Figure
else:
    from typing import Any as Figure


class ChartGenerator:
    """
    Generate visualization charts for comparison results.

    This class creates various types of charts to visualize the comparison
    between two benchmark versions.

    Attributes:
        comparison_df: DataFrame with comparison results from TrackEvalComparator
        version1: First version name
        version2: Second version name
        output_dir: Directory to save generated charts
    """

    def __init__(
        self,
        comparison_df: pd.DataFrame,
        version1: str,
        version2: str,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the chart generator.

        Args:
            comparison_df: Comparison DataFrame from TrackEvalComparator.compare()
            version1: Name of first version
            version2: Name of second version
            output_dir: Directory to save charts (optional)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        self.comparison_df = comparison_df
        self.version1 = version1
        self.version2 = version2
        self.output_dir = Path(output_dir) if output_dir else None

        # Create output directory if specified
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color scheme
        self.color_v1 = '#3498db'  # Blue
        self.color_v2 = '#2ecc71'  # Green
        self.color_improvement = '#27ae60'  # Dark green
        self.color_regression = '#e74c3c'   # Red

    def generate_bar_chart(
        self,
        metrics: List[str],
        sequence: str = 'COMBINED',
        figsize: Tuple[int, int] = (12, 6)
    ) -> Optional[Figure]:
        """
        Generate a bar chart comparing metrics between versions.

        Args:
            metrics: List of metric names to compare
            sequence: Sequence to compare (default: 'COMBINED')
            figsize: Figure size (width, height)

        Returns:
            matplotlib Figure object or None if data not available
        """
        # Filter data for the specified sequence and metrics
        data = self.comparison_df[
            (self.comparison_df['sequence'] == sequence) &
            (self.comparison_df['metric'].isin(metrics))
        ]

        if data.empty:
            warnings.warn(f"No data available for sequence '{sequence}' with metrics {metrics}")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data
        x = np.arange(len(metrics))
        width = 0.35

        v1_values = [data[data['metric'] == m]['v1_value'].values[0] if m in data['metric'].values else 0
                     for m in metrics]
        v2_values = [data[data['metric'] == m]['v2_value'].values[0] if m in data['metric'].values else 0
                     for m in metrics]

        # Create bars
        bars1 = ax.bar(x - width/2, v1_values, width, label=self.version1, color=self.color_v1, alpha=0.8)
        bars2 = ax.bar(x + width/2, v2_values, width, label=self.version2, color=self.color_v2, alpha=0.8)

        # Customize chart
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Metric Comparison - {sequence}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([get_metric_display_name(m).split('(')[0] for m in metrics], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if self.output_dir:
            output_path = self.output_dir / f'bar_chart_{sequence}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved bar chart: {output_path}")

        return fig

    def generate_heatmap(
        self,
        metrics: List[str],
        sequences: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8),
        show_deltas: bool = True
    ) -> Optional[Figure]:
        """
        Generate a heatmap showing metric improvements across sequences.

        Args:
            metrics: List of metrics to include
            sequences: List of sequences to include (None = all except COMBINED)
            figsize: Figure size
            show_deltas: If True, show delta values; if False, show v2 values

        Returns:
            matplotlib Figure object or None if data not available
        """
        if not SEABORN_AVAILABLE:
            warnings.warn("seaborn required for heatmaps. Skipping.")
            return None

        # Filter sequences
        if sequences is None:
            sequences = sorted([s for s in self.comparison_df['sequence'].unique()
                              if s != 'COMBINED'])

        # Filter data
        data = self.comparison_df[
            (self.comparison_df['sequence'].isin(sequences)) &
            (self.comparison_df['metric'].isin(metrics))
        ]

        if data.empty:
            warnings.warn(f"No data available for heatmap")
            return None

        # Create pivot table
        if show_deltas:
            pivot_data = data.pivot(index='sequence', columns='metric', values='delta_pct')
            title = f'Metric Changes (%) - {self.version2} vs {self.version1}'
            cmap = 'RdYlGn'
            center = 0
            fmt = '.2f'
        else:
            pivot_data = data.pivot(index='sequence', columns='metric', values='v2_value')
            title = f'Metric Values - {self.version2}'
            cmap = 'YlGnBu'
            center = None
            fmt = '.3f'

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt=fmt, cmap=cmap, center=center,
                   linewidths=0.5, ax=ax, cbar_kws={'label': '% Change' if show_deltas else 'Value'})

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sequences', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if self.output_dir:
            filename = 'heatmap_deltas.png' if show_deltas else 'heatmap_values.png'
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved heatmap: {output_path}")

        return fig

    def generate_radar_chart(
        self,
        metrics: List[str],
        sequence: str = 'COMBINED',
        figsize: Tuple[int, int] = (10, 10)
    ) -> Optional[Figure]:
        """
        Generate a radar chart comparing multiple metrics.

        Args:
            metrics: List of metrics to compare (3-8 metrics recommended)
            sequence: Sequence to compare
            figsize: Figure size

        Returns:
            matplotlib Figure object or None if data not available
        """
        # Filter data
        data = self.comparison_df[
            (self.comparison_df['sequence'] == sequence) &
            (self.comparison_df['metric'].isin(metrics))
        ]

        if data.empty or len(data) < 3:
            warnings.warn(f"Need at least 3 metrics for radar chart. Got {len(data)}")
            return None

        # Prepare data
        categories = [get_metric_display_name(m).split('(')[0] for m in metrics]
        v1_values = [data[data['metric'] == m]['v1_value'].values[0] if m in data['metric'].values else 0
                     for m in metrics]
        v2_values = [data[data['metric'] == m]['v2_value'].values[0] if m in data['metric'].values else 0
                     for m in metrics]

        # Number of variables
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        v1_values += v1_values[:1]  # Complete the circle
        v2_values += v2_values[:1]
        angles += angles[:1]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

        # Plot data
        ax.plot(angles, v1_values, 'o-', linewidth=2, label=self.version1, color=self.color_v1)
        ax.fill(angles, v1_values, alpha=0.25, color=self.color_v1)

        ax.plot(angles, v2_values, 'o-', linewidth=2, label=self.version2, color=self.color_v2)
        ax.fill(angles, v2_values, alpha=0.25, color=self.color_v2)

        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1.0)  # Assuming normalized metrics
        ax.set_title(f'Multi-Metric Comparison - {sequence}', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()

        if self.output_dir:
            output_path = self.output_dir / f'radar_chart_{sequence}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved radar chart: {output_path}")

        return fig

    def generate_improvement_chart(
        self,
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 8)
    ) -> Optional[Figure]:
        """
        Generate a chart showing top improvements and regressions.

        Args:
            top_n: Number of top changes to show
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        # Filter for COMBINED sequence and normalize direction so positive is good.
        combined = self.comparison_df[self.comparison_df['sequence'] == 'COMBINED'].copy()
        combined['improvement_pct'] = combined.apply(
            lambda row: row['delta_pct'] if is_higher_better(row['metric']) else -row['delta_pct'],
            axis=1
        )

        # Get top improvements and regressions
        improvements = combined.nlargest(top_n, 'improvement_pct')
        regressions = combined.nsmallest(top_n, 'improvement_pct')

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Top improvements
        if not improvements.empty:
            metrics_imp = [get_metric_display_name(m).split('(')[0] for m in improvements['metric']]
            values_imp = improvements['improvement_pct'].values

            bars1 = ax1.barh(range(len(metrics_imp)), values_imp, color=self.color_improvement)
            ax1.set_yticks(range(len(metrics_imp)))
            ax1.set_yticklabels(metrics_imp)
            ax1.set_xlabel('Improvement (%)', fontweight='bold')
            ax1.set_title(f'Top {top_n} Improvements', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars1, values_imp)):
                ax1.text(val, i, f' +{val:.2f}%', va='center', fontweight='bold')

        # Top regressions
        if not regressions.empty:
            metrics_reg = [get_metric_display_name(m).split('(')[0] for m in regressions['metric']]
            values_reg = regressions['improvement_pct'].values

            bars2 = ax2.barh(range(len(metrics_reg)), values_reg, color=self.color_regression)
            ax2.set_yticks(range(len(metrics_reg)))
            ax2.set_yticklabels(metrics_reg)
            ax2.set_xlabel('Regression (%)', fontweight='bold')
            ax2.set_title(f'Top {top_n} Regressions', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars2, values_reg)):
                ax2.text(val, i, f' {val:.2f}%', va='center', fontweight='bold')

        plt.suptitle(f'{self.version2} vs {self.version1}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if self.output_dir:
            output_path = self.output_dir / 'improvements_regressions.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved improvement chart: {output_path}")

        return fig

    def save_all_charts(
        self,
        primary_metrics: List[str] = ['HOTA___AUC', 'MOTA', 'IDF1'],
        sequences: Optional[List[str]] = None
    ):
        """
        Generate and save all available charts.

        Args:
            primary_metrics: Main metrics for detailed charts
            sequences: Sequences to include in heatmaps
        """
        if not self.output_dir:
            warnings.warn("No output directory specified. Charts will not be saved.")
            return

        print("\nGenerating charts...")

        # Bar chart for primary metrics
        self.generate_bar_chart(primary_metrics, sequence='COMBINED')

        # Heatmap
        if sequences:
            self.generate_heatmap(primary_metrics, sequences=sequences)

        # Radar chart
        self.generate_radar_chart(primary_metrics, sequence='COMBINED')

        # Improvement/Regression chart
        self.generate_improvement_chart(top_n=10)

        print("\n✓ All charts generated successfully")

    def close_all(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all')
