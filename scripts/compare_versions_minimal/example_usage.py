#!/usr/bin/env python3
"""
Example usage of the TrackEval comparison tool as a Python library.

This demonstrates how to use the tool programmatically instead of via CLI.
"""

from scripts.compare_versions_minimal import (
    TrackEvalComparator,
    MetricsConfig,
    ReportGenerator,
    ChartGenerator
)


def basic_comparison_example():
    """Basic comparison with primary metrics."""
    print("=== Basic Comparison Example ===\n")

    # Create configuration for primary metrics
    config = MetricsConfig(groups=['primary'])

    # Initialize comparator
    comparator = TrackEvalComparator(
        'v1.0.1-minimal',
        'v1.0.2-minimal',
        config=config
    )

    # Load and compare
    comparator.load_data()
    comparator.validate_data()
    comparison_df = comparator.compare()

    # Get summary
    summary = comparator.get_summary()
    print(f"Total sequences: {summary['total_sequences_compared']}")
    print(f"Improvements: {summary['improvements']}")
    print(f"Regressions: {summary['regressions']}")
    print()


def advanced_comparison_example():
    """Advanced comparison with custom metrics."""
    print("=== Advanced Comparison Example ===\n")

    # Create configuration with multiple groups and custom metrics
    config = MetricsConfig(
        groups=['primary', 'detection', 'counts'],
        custom_metrics=['IDSW', 'Frag'],  # Add ID switches and fragmentations
        exclude_metrics=['DetPr(0)'],     # Exclude detection precision
        include_thresholds=False
    )

    # Initialize and run comparison
    comparator = TrackEvalComparator(
        'v1.0.1-minimal',
        'v1.0.2-minimal',
        config=config
    )

    comparator.load_data()
    comparator.validate_data(mode='intersection')
    comparator.compare()

    # Get significant changes only (> 1%)
    improvements = comparator.get_improvements(threshold=0.01)
    regressions = comparator.get_regressions(threshold=0.01)

    print(f"Significant improvements: {len(improvements)}")
    print(f"Significant regressions: {len(regressions)}")
    print()


def full_report_example():
    """Generate all reports and visualizations."""
    print("=== Full Report Generation Example ===\n")

    # Configuration
    config = MetricsConfig(groups=['primary', 'counts'])

    # Comparison
    comparator = TrackEvalComparator(
        'v1.0.1-minimal',
        'v1.0.2-minimal',
        config=config
    )

    comparator.load_data()
    comparator.validate_data()
    comparator.compare()

    # Generate reports
    reporter = ReportGenerator(comparator, output_dir='output/comparison_example')

    # Terminal report
    reporter.generate_terminal_report(detailed=False)

    # JSON and HTML reports
    try:
        reporter.generate_all_reports(formats=['json', 'html'])
        print("\n✓ Reports generated in output/comparison_example/")
    except Exception as e:
        print(f"Warning: Could not generate all reports: {e}")

    # Generate charts (requires matplotlib)
    try:
        visualizer = ChartGenerator(
            comparator.comparison,
            'v1.0.1-minimal',
            'v1.0.2-minimal',
            output_dir='output/comparison_example'
        )

        primary_metrics = ['HOTA(0)', 'MOTA', 'IDF1']
        visualizer.generate_bar_chart(primary_metrics)
        visualizer.generate_improvement_chart(top_n=5)
        visualizer.close_all()

        print("✓ Charts generated in output/comparison_example/")
    except Exception as e:
        print(f"Warning: Could not generate charts: {e}")
        print("  Install matplotlib with: pip install matplotlib")

    print()


def custom_group_example():
    """Add and use custom metric groups."""
    print("=== Custom Metric Group Example ===\n")

    # Create configuration
    config = MetricsConfig()

    # Add custom group
    config.add_custom_group('tracking_quality', [
        'HOTA(0)',
        'MOTA',
        'IDF1',
        'IDSW',
        'Frag',
        'MT',
        'ML'
    ])

    # Use the custom group
    config.groups = ['tracking_quality']

    print(f"Custom metrics: {config.get_metrics()}")
    print()


if __name__ == '__main__':
    # Run examples
    basic_comparison_example()
    advanced_comparison_example()
    custom_group_example()

    # Full report (commented out by default to avoid generating files)
    # full_report_example()

    print("\n✓ All examples completed successfully")
