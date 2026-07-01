#!/usr/bin/env python3
"""
Command-line interface for TrackEval benchmark comparison tool.

Usage:
    python -m scripts.compare_versions_minimal v1.0.1-minimal v1.0.2-minimal
    python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 --groups primary detection counts
    python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 --output output/comparison/ --html --json
"""

import argparse
import sys
from pathlib import Path

from .comparator import TrackEvalComparator
from .config import MetricsConfig
from .reporter import ReportGenerator
from .visualizer import ChartGenerator
from .utils import list_available_versions


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare TrackEval benchmark results between two versions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison with primary metrics
  python -m scripts.compare_versions_minimal v1.0.1-minimal v1.0.2-minimal

  # Compare specific metric groups
  python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 --groups primary detection counts

  # Custom metrics
  python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 --metrics HOTA MOTA IDF1 Dets GT_Dets

  # Generate full report with charts
  python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 \\
      --output output/comparison/ --html --json --charts

  # Handle sequence mismatches
  python -m scripts.compare_versions_minimal v1.0.0 v1.0.3 --mode intersection

  # Detailed terminal output
  python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 --detailed

Available metric groups:
  - primary: HOTA, MOTA, IDF1
  - detection: DetA, DetRe, DetPr, CLR_Re, CLR_Pr
  - association: AssA, AssRe, AssPr, IDR, IDP
  - localization: LocA, MOTP
  - counts: Dets, GT_Dets, IDs, GT_IDs, CLR_TP, CLR_FN, CLR_FP, IDSW, Frag
  - advanced: sMOTA, CLR_F1, MTR, PTR, MLR, MT, PT, ML
        """
    )

    # Positional arguments (optional when using --list-versions)
    parser.add_argument('version1', nargs='?', help='First version to compare (e.g., v1.0.1-minimal)')
    parser.add_argument('version2', nargs='?', help='Second version to compare (e.g., v1.0.2-minimal)')

    # Metric selection
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument(
        '--groups',
        nargs='+',
        choices=['primary', 'detection', 'association', 'localization', 'counts', 'advanced'],
        default=['primary'],
        help='Metric groups to compare (default: primary)'
    )
    metric_group.add_argument(
        '--metrics',
        nargs='+',
        help='Specific metrics to compare (overrides --groups)'
    )
    metric_group.add_argument(
        '--all',
        action='store_true',
        help='Compare all available metrics'
    )

    # Comparison options
    parser.add_argument(
        '--mode',
        choices=['intersection', 'union', 'strict'],
        default='intersection',
        help='How to handle sequence mismatches (default: intersection)'
    )

    parser.add_argument(
        '--thresholds',
        action='store_true',
        help='Include threshold-based metrics (HOTA_5 to HOTA_95)'
    )

    # Output options
    parser.add_argument(
        '--output',
        '-o',
        help='Output directory for reports and charts'
    )

    parser.add_argument(
        '--base-dir',
        default='output/reports',
        help='Base directory for report files (default: output/reports)'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed per-sequence results in terminal'
    )

    # Report formats
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML report'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Generate JSON report'
    )

    parser.add_argument(
        '--charts',
        action='store_true',
        help='Generate visualization charts'
    )

    # Utility commands
    parser.add_argument(
        '--list-versions',
        action='store_true',
        help='List all available versions and exit'
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Handle --list-versions
    if args.list_versions:
        versions = list_available_versions(args.base_dir)
        if versions:
            print("Available versions:")
            for v in versions:
                print(f"  - {v}")
        else:
            print(f"No versions found in {args.base_dir}")
        return 0

    # Validate version arguments
    if not args.version1 or not args.version2:
        print("Error: Both version1 and version2 are required", file=sys.stderr)
        return 1

    # Build configuration
    try:
        if args.all:
            config = MetricsConfig.all_metrics()
        elif args.metrics:
            config = MetricsConfig(groups=['primary'], custom_metrics=args.metrics)
        else:
            config = MetricsConfig(
                groups=args.groups,
                include_thresholds=args.thresholds
            )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Create comparator
    try:
        print(f"\nInitializing comparison: {args.version1} vs {args.version2}")
        print(f"Configuration: {config}")
        print(f"Comparison mode: {args.mode}\n")

        comparator = TrackEvalComparator(
            args.version1,
            args.version2,
            config=config,
            base_dir=args.base_dir
        )

        # Load data
        comparator.load_data()

        # Validate and compare
        comparator.validate_data(mode=args.mode)
        comparator.compare(mode=args.mode)

        print("\n✓ Comparison completed successfully")

    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nAvailable versions:")
        for v in list_available_versions(args.base_dir):
            print(f"  - {v}")
        return 1
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Generate reports
    try:
        reporter = ReportGenerator(comparator, output_dir=args.output)

        # Always show terminal report
        reporter.generate_terminal_report(detailed=args.detailed)

        # Generate charts first if requested (so HTML can embed them)
        if args.charts or args.html:
            if not args.output:
                print("\nWarning: --output required for charts/HTML. Skipping chart generation.")
            else:
                print("\nGenerating charts...")
                visualizer = ChartGenerator(
                    comparator.comparison,
                    args.version1,
                    args.version2,
                    output_dir=args.output
                )

                # Get sequences for visualization
                sequences = comparator.common_sequences

                # Generate all charts
                primary_metrics = ['HOTA(0)', 'MOTA', 'IDF1']
                visualizer.save_all_charts(primary_metrics=primary_metrics, sequences=sequences)
                visualizer.close_all()

        # Generate requested report formats
        if args.json:
            reporter.generate_json_report()

        if args.html:
            if args.output:
                # HTML will embed the charts that were just created
                reporter.generate_html_report(include_charts=True)
            else:
                print("\nWarning: --output required for HTML report. Skipping.")

        print("\n✓ All operations completed successfully")
        return 0

    except Exception as e:
        print(f"\nError generating reports: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
