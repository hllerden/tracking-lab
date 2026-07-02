#!/usr/bin/env python3
"""
Command-line interface for TrackEval benchmark comparison tool.

Usage:
    python -m scripts.compare_versions_minimal v1.0.1-minimal v1.0.2-minimal
    python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 --groups primary detection counts
    python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 --output output/comparison/ --html --json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from html import escape
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(_REPO_ROOT / 'output' / '.matplotlib'))

from .comparator import TrackEvalComparator
from .config import MetricsConfig
from .reporter import ReportGenerator
from .utils import find_report_file, is_higher_better, list_available_versions, load_trackeval_csv


TRACKER_RANK_METRICS = ['HOTA___AUC', 'MOTA', 'IDF1', 'IDSW', 'Frag']


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

    # Multi-tracker ranking mode
    parser.add_argument(
        '--rank',
        nargs='+',
        help='Rank multiple TrackEval report names from --base-dir'
    )

    parser.add_argument(
        '--rank-prefix',
        help='Rank all report names in --base-dir starting with this prefix'
    )

    parser.add_argument(
        '--rank-metric',
        default='HOTA___AUC',
        help='Metric used for ranking (default: HOTA___AUC)'
    )

    return parser.parse_args()


def _resolve_rank_versions(args):
    """Resolve explicit or prefix-based report names for ranking."""
    if args.rank:
        return args.rank

    versions = list_available_versions(args.base_dir)
    if args.rank_prefix:
        matched = [v for v in versions if v.startswith(args.rank_prefix)]
        return sorted(matched)

    return []


def _format_metric_name(metric):
    if metric == 'HOTA___AUC':
        return 'HOTA'
    return metric


def _rank_values(rows, metric):
    reverse = is_higher_better(metric)
    return sorted(rows, key=lambda row: row['metrics'].get(metric, float('nan')), reverse=reverse)


def _build_tracker_ranking(args):
    """Build overall and per-sequence ranking data from TrackEval detailed CSVs."""
    versions = _resolve_rank_versions(args)
    if not versions:
        raise ValueError("No tracker reports matched. Use --rank or --rank-prefix.")

    trackers = []
    common_sequences = None
    for version in versions:
        csv_path = find_report_file(version, args.base_dir)
        df = load_trackeval_csv(csv_path)
        sequences = set(df.index) - {'COMBINED'}
        common_sequences = sequences if common_sequences is None else common_sequences & sequences

        tracker_name = version
        if args.rank_prefix and version.startswith(args.rank_prefix):
            tracker_name = version[len(args.rank_prefix):].lstrip('-_')

        combined = df.loc['COMBINED']
        metrics = {}
        for metric in TRACKER_RANK_METRICS:
            if metric in df.columns:
                metrics[metric] = float(combined[metric])

        if args.rank_metric not in df.columns:
            raise ValueError(
                f"Metric '{args.rank_metric}' not found in {version}. "
                f"Available examples: {', '.join(list(df.columns[:20]))}"
            )

        trackers.append({
            'version': version,
            'tracker': tracker_name,
            'csv_path': str(csv_path),
            'df': df,
            'metrics': metrics,
        })

    overall = _rank_values(trackers, args.rank_metric)

    sequence_rows = []
    for sequence in sorted(common_sequences or []):
        rows = []
        for tracker in trackers:
            row = tracker['df'].loc[sequence]
            metrics = {
                metric: float(row[metric])
                for metric in TRACKER_RANK_METRICS
                if metric in tracker['df'].columns
            }
            rows.append({
                'sequence': sequence,
                'version': tracker['version'],
                'tracker': tracker['tracker'],
                'metrics': metrics,
            })

        ranked = _rank_values(rows, args.rank_metric)
        for rank, row in enumerate(ranked, start=1):
            sequence_rows.append({
                'rank': rank,
                **row
            })

    winner_counts = {}
    for row in sequence_rows:
        if row['rank'] == 1:
            winner_counts[row['tracker']] = winner_counts.get(row['tracker'], 0) + 1

    return {
        'rank_metric': args.rank_metric,
        'higher_is_better': is_higher_better(args.rank_metric),
        'versions': versions,
        'overall': overall,
        'sequence_rows': sequence_rows,
        'winner_counts': winner_counts,
    }


def _print_tracker_ranking(ranking):
    direction = 'higher is better' if ranking['higher_is_better'] else 'lower is better'
    metric_label = _format_metric_name(ranking['rank_metric'])

    print("\n" + "=" * 78)
    print("  TrackEval Tracker Ranking")
    print(f"  Sort metric: {metric_label} ({direction})")
    print("=" * 78)
    print(f"{'#':<4} {'Tracker':<24} {'HOTA':>9} {'MOTA':>9} {'IDF1':>9} {'IDSW':>9} {'Frag':>9}")
    print("-" * 78)

    for rank, row in enumerate(ranking['overall'], start=1):
        metrics = row['metrics']
        print(
            f"{rank:<4} {row['tracker']:<24} "
            f"{metrics.get('HOTA___AUC', float('nan')):>9.3f} "
            f"{metrics.get('MOTA', float('nan')):>9.3f} "
            f"{metrics.get('IDF1', float('nan')):>9.3f} "
            f"{metrics.get('IDSW', float('nan')):>9.0f} "
            f"{metrics.get('Frag', float('nan')):>9.0f}"
        )

    best = ranking['overall'][0]['tracker']
    print(f"\nBest by {metric_label}: {best}")

    if ranking['winner_counts']:
        print("\nSequence winner counts:")
        for tracker, count in sorted(ranking['winner_counts'].items(), key=lambda item: item[1], reverse=True):
            print(f"  {tracker:<24} {count}")


def _ranking_to_jsonable(ranking):
    def compact_row(row):
        return {
            'rank': row.get('rank'),
            'version': row['version'],
            'tracker': row['tracker'],
            'sequence': row.get('sequence'),
            'metrics': row['metrics'],
        }

    return {
        'generated_at': datetime.now().isoformat(),
        'rank_metric': ranking['rank_metric'],
        'higher_is_better': ranking['higher_is_better'],
        'versions': ranking['versions'],
        'overall': [compact_row({'rank': rank, **row}) for rank, row in enumerate(ranking['overall'], start=1)],
        'sequence_rows': [compact_row(row) for row in ranking['sequence_rows']],
        'winner_counts': ranking['winner_counts'],
    }


def _write_tracker_ranking_reports(ranking, output_dir, write_json=False, write_html=False):
    if not output_dir or (not write_json and not write_html):
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = _ranking_to_jsonable(ranking)

    if write_json:
        path = out / 'tracker_ranking_report.json'
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"✓ Saved tracker ranking JSON: {path.resolve()}")

    if write_html:
        path = out / 'tracker_ranking_report.html'
        metric_label = escape(_format_metric_name(ranking['rank_metric']))
        direction = 'higher is better' if ranking['higher_is_better'] else 'lower is better'

        def best_classes(rows, metrics):
            classes = {}
            for metric in metrics:
                values = []
                for idx, row in enumerate(rows):
                    value = row['metrics'].get(metric)
                    if value == value:
                        values.append((idx, value))
                if not values:
                    continue
                values.sort(key=lambda item: item[1], reverse=is_higher_better(metric))
                classes[(values[0][0], metric)] = 'best-cell'
                if len(values) > 1:
                    classes[(values[1][0], metric)] = 'second-cell'
            return classes

        def metric_cell(value, css_class='', decimals=3):
            if value != value:
                text = 'n/a'
            elif decimals == 0:
                text = f"{value:.0f}"
            else:
                text = f"{value:.{decimals}f}"
            class_attr = f' class="{css_class}"' if css_class else ''
            return f"<td{class_attr}>{text}</td>"

        metric_order = ['HOTA___AUC', 'MOTA', 'IDF1', 'IDSW', 'Frag']
        overall_classes = best_classes(ranking['overall'], metric_order)

        overall_rows = []
        for rank, row in enumerate(ranking['overall'], start=1):
            idx = rank - 1
            m = row['metrics']
            overall_rows.append(
                f"<tr class=\"{'winner-row' if rank == 1 else ''}\">"
                f"<td>{rank}</td><td>{escape(row['tracker'])}</td>"
                f"{metric_cell(m.get('HOTA___AUC', float('nan')), overall_classes.get((idx, 'HOTA___AUC'), ''))}"
                f"{metric_cell(m.get('MOTA', float('nan')), overall_classes.get((idx, 'MOTA'), ''))}"
                f"{metric_cell(m.get('IDF1', float('nan')), overall_classes.get((idx, 'IDF1'), ''))}"
                f"{metric_cell(m.get('IDSW', float('nan')), overall_classes.get((idx, 'IDSW'), ''), decimals=0)}"
                f"{metric_cell(m.get('Frag', float('nan')), overall_classes.get((idx, 'Frag'), ''), decimals=0)}"
                "</tr>"
            )

        sequence_rows = []
        sequence_groups = {}
        for row in ranking['sequence_rows']:
            sequence_groups.setdefault(row['sequence'], []).append(row)

        sequence_cell_classes = {}
        for sequence, rows in sequence_groups.items():
            row_classes = best_classes(rows, metric_order)
            for idx, row in enumerate(rows):
                for metric in metric_order:
                    if (idx, metric) in row_classes:
                        sequence_cell_classes[(sequence, row['tracker'], metric)] = row_classes[(idx, metric)]

        for row in ranking['sequence_rows']:
            m = row['metrics']
            sequence_rows.append(
                f"<tr class=\"{'winner-row' if row['rank'] == 1 else ''}\">"
                f"<td>{escape(row['sequence'])}</td><td>{row['rank']}</td><td>{escape(row['tracker'])}</td>"
                f"{metric_cell(m.get('HOTA___AUC', float('nan')), sequence_cell_classes.get((row['sequence'], row['tracker'], 'HOTA___AUC'), ''))}"
                f"{metric_cell(m.get('MOTA', float('nan')), sequence_cell_classes.get((row['sequence'], row['tracker'], 'MOTA'), ''))}"
                f"{metric_cell(m.get('IDF1', float('nan')), sequence_cell_classes.get((row['sequence'], row['tracker'], 'IDF1'), ''))}"
                f"{metric_cell(m.get('IDSW', float('nan')), sequence_cell_classes.get((row['sequence'], row['tracker'], 'IDSW'), ''), decimals=0)}"
                f"{metric_cell(m.get('Frag', float('nan')), sequence_cell_classes.get((row['sequence'], row['tracker'], 'Frag'), ''), decimals=0)}"
                "</tr>"
            )

        winner_rows = [
            f"<tr><td>{escape(tracker)}</td><td>{count}</td></tr>"
            for tracker, count in sorted(ranking['winner_counts'].items(), key=lambda item: item[1], reverse=True)
        ]

        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Tracker Ranking Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; background: #f6f8fa; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 32px; background: white; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
    th {{ background: #243b53; color: white; }}
    tr:nth-child(even) {{ background: #f0f4f8; }}
    .winner-row {{ background: #edf7f2; }}
    .winner-row:nth-child(even) {{ background: #e7f2ee; }}
    .best-cell {{
      background: #d9eadf;
      color: #163b24;
      font-weight: 700;
      box-shadow: inset 3px 0 0 #6f9f83;
    }}
    .second-cell {{
      background: #e8eef6;
      color: #25364a;
      font-weight: 600;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      align-items: center;
      margin: 12px 0 20px;
      color: #52606d;
      font-size: 14px;
    }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 18px; height: 12px; border-radius: 2px; display: inline-block; }}
    .swatch-best {{ background: #d9eadf; border-left: 3px solid #6f9f83; }}
    .swatch-second {{ background: #e8eef6; }}
    .meta {{ color: #52606d; }}
  </style>
</head>
<body>
  <h1>TrackEval Tracker Ranking</h1>
  <p class="meta">Sort metric: <strong>{metric_label}</strong> ({direction})</p>
  <h2>Overall Ranking (COMBINED)</h2>
  <div class="legend">
    <span><i class="swatch swatch-best"></i>Best value for that metric</span>
    <span><i class="swatch swatch-second"></i>Second best</span>
  </div>
  <table>
    <thead><tr><th>#</th><th>Tracker</th><th>HOTA</th><th>MOTA</th><th>IDF1</th><th>IDSW</th><th>Frag</th></tr></thead>
    <tbody>{''.join(overall_rows)}</tbody>
  </table>
  <h2>Sequence Winner Counts</h2>
  <table>
    <thead><tr><th>Tracker</th><th>Wins</th></tr></thead>
    <tbody>{''.join(winner_rows)}</tbody>
  </table>
  <h2>Per-Sequence Ranking</h2>
  <table>
    <thead><tr><th>Sequence</th><th>#</th><th>Tracker</th><th>HOTA</th><th>MOTA</th><th>IDF1</th><th>IDSW</th><th>Frag</th></tr></thead>
    <tbody>{''.join(sequence_rows)}</tbody>
  </table>
</body>
</html>
"""
        with open(path, 'w') as f:
            f.write(html)
        print(f"✓ Saved tracker ranking HTML: {path.resolve()}")


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

    # Handle multi-tracker ranking mode
    if args.rank or args.rank_prefix:
        try:
            ranking = _build_tracker_ranking(args)
            _print_tracker_ranking(ranking)
            _write_tracker_ranking_reports(
                ranking,
                args.output,
                write_json=args.json,
                write_html=args.html
            )
            return 0
        except Exception as e:
            print(f"\nError ranking trackers: {e}", file=sys.stderr)
            return 1

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
                from .visualizer import ChartGenerator
                visualizer = ChartGenerator(
                    comparator.comparison,
                    args.version1,
                    args.version2,
                    output_dir=args.output
                )

                # Get sequences for visualization
                sequences = comparator.common_sequences

                # Generate all charts
                primary_metrics = [m for m in ['HOTA___AUC', 'MOTA', 'IDF1'] if m in comparator.comparison['metric'].values]
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
