"""
Report Generator - Generate comparison reports in various formats.

This module provides the ReportGenerator class for creating formatted
reports of benchmark comparisons.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
import base64

# Try to import rich for colored terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    warnings.warn("rich not available for colored output. Install with: pip install rich")

from .utils import format_percentage, format_delta, get_metric_display_name, is_higher_better


class ReportGenerator:
    """
    Generate comparison reports in various formats.

    This class creates reports from TrackEvalComparator results in
    multiple formats: terminal, HTML, JSON, and optionally PDF.

    Attributes:
        comparator: TrackEvalComparator instance with comparison results
        output_dir: Directory to save reports
    """

    def __init__(self, comparator, output_dir: Optional[str] = None):
        """
        Initialize the report generator.

        Args:
            comparator: TrackEvalComparator instance with loaded and compared data
            output_dir: Directory to save reports (optional)
        """
        self.comparator = comparator
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Rich console if available
        self.console = Console() if RICH_AVAILABLE else None

    def generate_terminal_report(self, detailed: bool = False):
        """
        Generate a formatted terminal report.

        Args:
            detailed: If True, show detailed per-sequence results
        """
        if self.comparator.comparison is None:
            print("Error: No comparison data available. Run compare() first.")
            return

        summary = self.comparator.get_summary()

        if RICH_AVAILABLE:
            self._generate_rich_report(summary, detailed)
        else:
            self._generate_plain_report(summary, detailed)

    def _generate_rich_report(self, summary: Dict, detailed: bool):
        """Generate a colorful terminal report using Rich library."""
        # Header
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold cyan]TrackEval Benchmark Comparison[/bold cyan]\n"
            f"[yellow]{summary['version1']}[/yellow] vs [green]{summary['version2']}[/green]",
            border_style="bold blue"
        ))

        # Summary statistics
        summary_table = Table(title="Summary", box=box.ROUNDED, show_header=True)
        summary_table.add_column("Metric", style="cyan", width=30)
        summary_table.add_column("Value", style="white", justify="right")

        summary_table.add_row("Total Sequences Compared", str(summary['total_sequences_compared']))
        summary_table.add_row("Total Metrics Compared", str(summary['total_metrics_compared']))
        summary_table.add_row(
            "Improvements",
            f"[green]{summary['improvements']}[/green]"
        )
        summary_table.add_row(
            "Regressions",
            f"[red]{summary['regressions']}[/red]"
        )

        if summary['sequences_only_in_v1']:
            summary_table.add_row(
                f"Only in {summary['version1']}",
                f"[yellow]{', '.join(summary['sequences_only_in_v1'])}[/yellow]"
            )

        if summary['sequences_only_in_v2']:
            summary_table.add_row(
                f"Only in {summary['version2']}",
                f"[yellow]{', '.join(summary['sequences_only_in_v2'])}[/yellow]"
            )

        self.console.print(summary_table)

        # Key metrics comparison (COMBINED row)
        self._print_key_metrics_table(summary)

        # Detailed per-sequence results
        if detailed:
            self._print_detailed_results()

    def _generate_plain_report(self, summary: Dict, detailed: bool):
        """Generate a plain text terminal report (fallback when Rich not available)."""
        print("\n" + "="*70)
        print(f"  TrackEval Benchmark Comparison")
        print(f"  {summary['version1']} vs {summary['version2']}")
        print("="*70)

        print("\nSummary:")
        print(f"  Total Sequences Compared: {summary['total_sequences_compared']}")
        print(f"  Total Metrics Compared: {summary['total_metrics_compared']}")
        print(f"  Improvements: {summary['improvements']}")
        print(f"  Regressions: {summary['regressions']}")

        if summary['sequences_only_in_v1']:
            print(f"  Only in {summary['version1']}: {', '.join(summary['sequences_only_in_v1'])}")

        if summary['sequences_only_in_v2']:
            print(f"  Only in {summary['version2']}: {', '.join(summary['sequences_only_in_v2'])}")

        # Key metrics
        self._print_key_metrics_plain(summary)

        if detailed:
            self._print_detailed_results_plain()

    def _print_key_metrics_table(self, summary: Dict):
        """Print key metrics comparison table using Rich."""
        key_metrics = ['HOTA(0)', 'MOTA', 'IDF1']

        table = Table(title="Key Metrics (COMBINED)", box=box.ROUNDED, show_header=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column(summary['version1'], style="yellow", justify="right")
        table.add_column(summary['version2'], style="green", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Change (%)", justify="right")

        for metric in key_metrics:
            v1_key = f'{metric}_v1'
            v2_key = f'{metric}_v2'
            delta_key = f'{metric}_delta'
            pct_key = f'{metric}_delta_pct'

            if v1_key in summary:
                v1_val = summary[v1_key]
                v2_val = summary[v2_key]
                delta = summary[delta_key]
                pct = summary[pct_key]

                # Color code based on improvement
                delta_str = format_delta(delta, precision=4)
                pct_str = format_percentage(pct / 100, precision=2)

                if delta > 0:
                    delta_str = f"[green]{delta_str}[/green]"
                    pct_str = f"[green]{pct_str}[/green]"
                elif delta < 0:
                    delta_str = f"[red]{delta_str}[/red]"
                    pct_str = f"[red]{pct_str}[/red]"

                table.add_row(
                    get_metric_display_name(metric).split('(')[0],
                    f"{v1_val:.4f}",
                    f"{v2_val:.4f}",
                    delta_str,
                    pct_str
                )

        self.console.print()
        self.console.print(table)

    def _print_key_metrics_plain(self, summary: Dict):
        """Print key metrics in plain text format."""
        key_metrics = ['HOTA(0)', 'MOTA', 'IDF1']

        print("\nKey Metrics (COMBINED):")
        print(f"{'Metric':<15} {summary['version1']:<12} {summary['version2']:<12} {'Delta':<12} {'Change (%)':<12}")
        print("-" * 70)

        for metric in key_metrics:
            v1_key = f'{metric}_v1'
            if v1_key in summary:
                v1_val = summary[v1_key]
                v2_val = summary[v2_key]
                delta = summary[delta_key]
                pct = summary[pct_key]

                print(
                    f"{metric:<15} {v1_val:<12.4f} {v2_val:<12.4f} "
                    f"{format_delta(delta, precision=4):<12} {format_percentage(pct/100, precision=2):<12}"
                )

    def _print_detailed_results(self):
        """Print detailed per-sequence results using Rich."""
        # Get sequences (excluding COMBINED)
        sequences = sorted([s for s in self.comparator.comparison['sequence'].unique()
                          if s != 'COMBINED'])

        if not sequences:
            return

        for seq in sequences:
            seq_data = self.comparator.comparison[
                self.comparator.comparison['sequence'] == seq
            ]

            table = Table(title=f"Sequence: {seq}", box=box.SIMPLE, show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column(self.comparator.version1, justify="right")
            table.add_column(self.comparator.version2, justify="right")
            table.add_column("Delta", justify="right")

            for _, row in seq_data.iterrows():
                metric_name = get_metric_display_name(row['metric']).split('(')[0]
                delta_str = format_delta(row['delta'], precision=4)

                if row['improved']:
                    delta_str = f"[green]{delta_str}[/green]"
                else:
                    delta_str = f"[red]{delta_str}[/red]"

                table.add_row(
                    metric_name,
                    f"{row['v1_value']:.4f}",
                    f"{row['v2_value']:.4f}",
                    delta_str
                )

            self.console.print()
            self.console.print(table)

    def _print_detailed_results_plain(self):
        """Print detailed results in plain text format."""
        sequences = sorted([s for s in self.comparator.comparison['sequence'].unique()
                          if s != 'COMBINED'])

        for seq in sequences:
            print(f"\n{seq}:")
            print(f"{'Metric':<20} {self.comparator.version1:<12} {self.comparator.version2:<12} {'Delta':<12}")
            print("-" * 60)

            seq_data = self.comparator.comparison[
                self.comparator.comparison['sequence'] == seq
            ]

            for _, row in seq_data.iterrows():
                metric_name = row['metric'][:18]
                print(
                    f"{metric_name:<20} {row['v1_value']:<12.4f} {row['v2_value']:<12.4f} "
                    f"{format_delta(row['delta'], precision=4):<12}"
                )

    def generate_json_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate a JSON report of the comparison.

        Args:
            output_path: Path to save JSON file (optional)

        Returns:
            Dictionary with comparison results
        """
        if self.comparator.comparison is None:
            raise ValueError("No comparison data available. Run compare() first.")

        summary = self.comparator.get_summary()

        # Add timestamp
        summary['generated_at'] = datetime.now().isoformat()

        # Add detailed comparison data
        summary['detailed_comparison'] = self.comparator.comparison.to_dict(orient='records')

        # Add improvements and regressions
        improvements = self.comparator.get_improvements()
        regressions = self.comparator.get_regressions()

        summary['top_improvements'] = improvements.head(10).to_dict(orient='records')
        summary['top_regressions'] = regressions.head(10).to_dict(orient='records')

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
        elif self.output_dir:
            output_file = self.output_dir / 'comparison_report.json'
        else:
            output_file = None

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Saved JSON report: {output_file}")

        return summary

    def generate_html_report(self, output_path: Optional[str] = None, include_charts: bool = True):
        """
        Generate an HTML report with embedded charts.

        Args:
            output_path: Path to save HTML file (optional)
            include_charts: If True, embed chart images in HTML (default: True)
        """
        if self.comparator.comparison is None:
            raise ValueError("No comparison data available. Run compare() first.")

        summary = self.comparator.get_summary()

        # Build HTML with optional charts
        html_content = self._build_html_template(summary, include_charts=include_charts)

        # Save to file
        if output_path:
            output_file = Path(output_path)
        elif self.output_dir:
            output_file = self.output_dir / 'comparison_report.html'
        else:
            output_file = Path('comparison_report.html')

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"✓ Saved HTML report: {output_file}")

    def _embed_image_as_base64(self, image_path: Path) -> Optional[str]:
        """
        Embed an image as base64 for HTML.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string or None if file doesn't exist
        """
        if not image_path.exists():
            return None

        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_data = base64.b64encode(image_data).decode('utf-8')
                return f"data:image/png;base64,{base64_data}"
        except Exception as e:
            warnings.warn(f"Failed to embed image {image_path}: {e}")
            return None

    def _build_html_template(self, summary: Dict, include_charts: bool = True) -> str:
        """Build HTML template for the report."""
        # Get COMBINED metrics for display
        combined = self.comparator.comparison[
            self.comparator.comparison['sequence'] == 'COMBINED'
        ].sort_values('delta_pct', ascending=False)

        # Build metrics table rows
        metrics_rows = ""
        for _, row in combined.iterrows():
            improved_class = "improved" if row['improved'] else "regressed"
            metrics_rows += f"""
            <tr class="{improved_class}">
                <td>{get_metric_display_name(row['metric'])}</td>
                <td>{row['v1_value']:.4f}</td>
                <td>{row['v2_value']:.4f}</td>
                <td>{format_delta(row['delta'], precision=4)}</td>
                <td>{format_percentage(row['delta_pct']/100, precision=2)}</td>
            </tr>
            """

        # Build charts section if requested
        charts_html = ""
        if include_charts and self.output_dir:
            charts_html = self._build_charts_section()
        elif include_charts:
            warnings.warn("Charts requested but no output_dir specified. Skipping charts in HTML.")

        # Build per-sequence tables
        sequences_html = self._build_sequences_section()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Comparison Report</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                    border-radius: 10px;
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                    color: white;
                    padding: 30px 40px;
                }}
                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }}
                .header p {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}
                .summary {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px 40px;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                }}
                .summary-item {{
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                }}
                .summary-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                .summary-value {{
                    font-size: 1.3em;
                    font-weight: bold;
                }}

                /* Tab Navigation */
                .tab-navigation {{
                    display: flex;
                    background-color: #34495e;
                    overflow-x: auto;
                }}
                .tab-button {{
                    padding: 15px 25px;
                    background-color: transparent;
                    border: none;
                    color: rgba(255,255,255,0.7);
                    cursor: pointer;
                    font-size: 1em;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    white-space: nowrap;
                    border-bottom: 3px solid transparent;
                }}
                .tab-button:hover {{
                    background-color: rgba(255,255,255,0.1);
                    color: white;
                }}
                .tab-button.active {{
                    background-color: white;
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                }}

                /* Tab Content */
                .tab-content {{
                    display: none;
                    padding: 40px;
                    animation: fadeIn 0.3s ease;
                }}
                .tab-content.active {{
                    display: block;
                }}
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(10px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}

                h2 {{
                    color: #2c3e50;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #ecf0f1;
                    font-size: 1.8em;
                }}
                h3 {{
                    color: #7f8c8d;
                    margin-top: 25px;
                    margin-bottom: 15px;
                    font-size: 1.3em;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    border-radius: 8px;
                    overflow: hidden;
                }}
                th {{
                    background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                    color: white;
                    padding: 15px;
                    text-align: left;
                    font-weight: 600;
                    text-transform: uppercase;
                    font-size: 0.9em;
                    letter-spacing: 0.5px;
                }}
                td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                tr:hover {{
                    background-color: #f8f9fa;
                }}
                .improved {{
                    background-color: #d5f4e6;
                    border-left: 4px solid #27ae60;
                }}
                .regressed {{
                    background-color: #fadbd8;
                    border-left: 4px solid #e74c3c;
                }}
                .chart-container {{
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                }}
                .chart-container img {{
                    width: 100%;
                    max-width: 1200px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }}
                .badge {{
                    display: inline-block;
                    padding: 6px 14px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: 600;
                }}
                .badge-success {{
                    background-color: #27ae60;
                    color: white;
                }}
                .badge-danger {{
                    background-color: #e74c3c;
                    color: white;
                }}
                .timestamp {{
                    color: #95a5a6;
                    font-size: 0.9em;
                    margin-top: 30px;
                    text-align: right;
                    font-style: italic;
                }}

                /* Sub-tab Navigation (for Per-Sequence) */
                .sub-tab-navigation {{
                    display: flex;
                    background-color: #ecf0f1;
                    border-radius: 8px;
                    padding: 5px;
                    margin-bottom: 30px;
                    gap: 5px;
                    flex-wrap: wrap;
                }}
                .sub-tab-button {{
                    padding: 10px 20px;
                    background-color: transparent;
                    border: none;
                    color: #7f8c8d;
                    cursor: pointer;
                    font-size: 0.95em;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    border-radius: 5px;
                    white-space: nowrap;
                }}
                .sub-tab-button:hover {{
                    background-color: rgba(52, 152, 219, 0.1);
                    color: #3498db;
                }}
                .sub-tab-button.active {{
                    background-color: #3498db;
                    color: white;
                    box-shadow: 0 2px 5px rgba(52, 152, 219, 0.3);
                }}

                /* Sub-tab Content */
                .sub-tab-content {{
                    display: none;
                }}
                .sub-tab-content.active {{
                    display: block;
                    animation: fadeIn 0.3s ease;
                }}

                /* Scrollbar styling */
                ::-webkit-scrollbar {{
                    width: 10px;
                    height: 10px;
                }}
                ::-webkit-scrollbar-track {{
                    background: #f1f1f1;
                }}
                ::-webkit-scrollbar-thumb {{
                    background: #888;
                    border-radius: 5px;
                }}
                ::-webkit-scrollbar-thumb:hover {{
                    background: #555;
                }}
            </style>
            <script>
                function showTab(tabId) {{
                    // Hide all tabs
                    const tabs = document.querySelectorAll('.tab-content');
                    tabs.forEach(tab => tab.classList.remove('active'));

                    // Deactivate all buttons
                    const buttons = document.querySelectorAll('.tab-button');
                    buttons.forEach(btn => btn.classList.remove('active'));

                    // Show selected tab
                    document.getElementById(tabId).classList.add('active');

                    // Activate selected button
                    event.target.classList.add('active');

                    // If switching to per-sequence tab, activate first sub-tab
                    if (tabId === 'tab-sequences') {{
                        const firstSubTab = document.querySelector('.sub-tab-button');
                        if (firstSubTab) {{
                            setTimeout(() => firstSubTab.click(), 50);
                        }}
                    }}
                }}

                function showSubTab(subTabId) {{
                    // Hide all sub-tabs
                    const subTabs = document.querySelectorAll('.sub-tab-content');
                    subTabs.forEach(tab => tab.classList.remove('active'));

                    // Deactivate all sub-buttons
                    const subButtons = document.querySelectorAll('.sub-tab-button');
                    subButtons.forEach(btn => btn.classList.remove('active'));

                    // Show selected sub-tab
                    document.getElementById(subTabId).classList.add('active');

                    // Activate selected sub-button
                    event.target.classList.add('active');
                }}

                // Show first tab on load
                window.addEventListener('DOMContentLoaded', function() {{
                    document.querySelector('.tab-button').click();
                }});
            </script>
        </head>
        <body>
            <div class="container">
                <!-- Header Section -->
                <div class="header">
                    <h1>📊 TrackEval Benchmark Comparison</h1>
                    <p>{summary['version1']} vs {summary['version2']}</p>
                </div>

                <!-- Summary Section -->
                <div class="summary">
                    <div class="summary-item">
                        <span class="summary-label">Version 1</span>
                        <span class="summary-value">{summary['version1']}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Version 2</span>
                        <span class="summary-value">{summary['version2']}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Sequences Compared</span>
                        <span class="summary-value">{summary['total_sequences_compared']}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Metrics Compared</span>
                        <span class="summary-value">{summary['total_metrics_compared']}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Improvements</span>
                        <span class="summary-value">
                            <span class="badge badge-success">{summary['improvements']}</span>
                        </span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Regressions</span>
                        <span class="summary-value">
                            <span class="badge badge-danger">{summary['regressions']}</span>
                        </span>
                    </div>
                </div>

                <!-- Tab Navigation -->
                <div class="tab-navigation">
                    <button class="tab-button" onclick="showTab('tab-overview')">📈 Overview</button>
                    <button class="tab-button" onclick="showTab('tab-charts')">📊 Visualizations</button>
                    <button class="tab-button" onclick="showTab('tab-sequences')">📋 Per-Sequence</button>
                </div>

                <!-- Tab 1: Overview -->
                <div id="tab-overview" class="tab-content">
                    <h2>Combined Metrics Comparison</h2>
                    <p style="color: #7f8c8d; margin-bottom: 20px;">
                        Overall performance metrics across all test sequences
                    </p>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>{summary['version1']}</th>
                                <th>{summary['version2']}</th>
                                <th>Delta</th>
                                <th>Change (%)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {metrics_rows}
                        </tbody>
                    </table>
                    <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>

                <!-- Tab 2: Charts -->
                <div id="tab-charts" class="tab-content">
                    {charts_html if charts_html else '<p style="color: #7f8c8d; text-align: center; padding: 40px;">No charts available. Run with --html flag to generate charts.</p>'}
                </div>

                <!-- Tab 3: Per-Sequence Details -->
                <div id="tab-sequences" class="tab-content">
                    {sequences_html if sequences_html else '<p style="color: #7f8c8d; text-align: center; padding: 40px;">No sequence data available.</p>'}
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _build_charts_section(self) -> str:
        """Build HTML section with embedded chart images."""
        charts = []
        chart_files = [
            ('bar_chart_COMBINED.png', '📊 Primary Metrics Comparison', 'Side-by-side comparison of key tracking metrics'),
            ('improvements_regressions.png', '📈 Top Improvements and Regressions', 'Most significant changes between versions'),
            ('heatmap_deltas.png', '🔥 Metric Changes Heatmap', 'Color-coded metric improvements across all sequences'),
            ('radar_chart_COMBINED.png', '🎯 Multi-Metric Radar Chart', 'Overall performance visualization'),
        ]

        for filename, title, description in chart_files:
            chart_path = self.output_dir / filename
            base64_img = self._embed_image_as_base64(chart_path)

            if base64_img:
                charts.append(f"""
                <div class="chart-container">
                    <h3>{title}</h3>
                    <p style="color: #7f8c8d; margin-bottom: 15px;">{description}</p>
                    <img src="{base64_img}" alt="{title}">
                </div>
                """)

        if charts:
            return f"""
            <h2>Visual Performance Analysis</h2>
            <p style="color: #7f8c8d; font-size: 1.1em; margin-bottom: 30px;">
                Interactive charts showing performance changes across versions
            </p>
            {''.join(charts)}
            """
        return ""

    def _build_sequences_section(self) -> str:
        """Build HTML section with per-sequence comparison tables using sub-tabs."""
        sequences = sorted([s for s in self.comparator.comparison['sequence'].unique()
                          if s != 'COMBINED'])

        if not sequences:
            return ""

        # Build sub-tab navigation buttons
        sub_tab_buttons = []
        sub_tab_contents = []

        for idx, seq in enumerate(sequences):
            # Add emoji based on sequence
            emoji = "🎬"
            if "DPM" in seq:
                emoji = "🟦"
            elif "FRCNN" in seq:
                emoji = "🟩"
            elif "SDP" in seq:
                emoji = "🟨"

            # Create button
            sub_tab_buttons.append(f"""
                <button class="sub-tab-button" onclick="showSubTab('subtab-{idx}')">
                    {emoji} {seq}
                </button>
            """)

            # Get sequence data
            seq_data = self.comparator.comparison[
                self.comparator.comparison['sequence'] == seq
            ].sort_values('delta_pct', ascending=False)

            # Build table rows
            rows = ""
            for _, row in seq_data.iterrows():
                improved_class = "improved" if row['improved'] else "regressed"
                rows += f"""
                <tr class="{improved_class}">
                    <td>{get_metric_display_name(row['metric'])}</td>
                    <td>{row['v1_value']:.4f}</td>
                    <td>{row['v2_value']:.4f}</td>
                    <td>{format_delta(row['delta'], precision=4)}</td>
                    <td>{format_percentage(row['delta_pct']/100, precision=2)}</td>
                </tr>
                """

            # Calculate summary stats for this sequence
            improvements = int(seq_data['improved'].sum())
            regressions = int((~seq_data['improved']).sum())

            # Create sub-tab content
            sub_tab_contents.append(f"""
            <div id="subtab-{idx}" class="sub-tab-content">
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h3 style="margin: 0 0 10px 0; color: #2c3e50;">{emoji} {seq}</h3>
                    <div style="display: flex; gap: 30px; flex-wrap: wrap;">
                        <div>
                            <span style="color: #7f8c8d; font-size: 0.9em;">Improvements:</span>
                            <span class="badge badge-success" style="margin-left: 10px;">{improvements}</span>
                        </div>
                        <div>
                            <span style="color: #7f8c8d; font-size: 0.9em;">Regressions:</span>
                            <span class="badge badge-danger" style="margin-left: 10px;">{regressions}</span>
                        </div>
                        <div>
                            <span style="color: #7f8c8d; font-size: 0.9em;">Total Metrics:</span>
                            <strong style="margin-left: 10px;">{len(seq_data)}</strong>
                        </div>
                    </div>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>{self.comparator.version1}</th>
                            <th>{self.comparator.version2}</th>
                            <th>Delta</th>
                            <th>Change (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
            """)

        return f"""
        <h2>Detailed Sequence Analysis</h2>
        <p style="color: #7f8c8d; font-size: 1.1em; margin-bottom: 20px;">
            Select a sequence to view detailed metric comparison
        </p>

        <!-- Sub-tab Navigation -->
        <div class="sub-tab-navigation">
            {''.join(sub_tab_buttons)}
        </div>

        <!-- Sub-tab Contents -->
        {''.join(sub_tab_contents)}
        """

    def generate_all_reports(self, formats: List[str] = ['terminal', 'json', 'html']):
        """
        Generate reports in all specified formats.

        Args:
            formats: List of formats to generate ('terminal', 'json', 'html')
        """
        if 'terminal' in formats:
            self.generate_terminal_report(detailed=False)

        if 'json' in formats:
            self.generate_json_report()

        if 'html' in formats:
            self.generate_html_report()

        print("\n✓ All reports generated successfully")
