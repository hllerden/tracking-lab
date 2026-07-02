"""
Multi-tracker report generator - terminal, HTML and JSON outputs.

Renders MultiTrackerComparator results. Missing sides (tracker or sequence
present in only one version) are shown as MISS and never compared.
"""

import json
import warnings
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    warnings.warn("rich not available for colored output. Install with: pip install rich")

from .utils import get_metric_display_name, is_higher_better


KEY_METRICS = ['HOTA___AUC', 'MOTA', 'IDF1', 'IDSW', 'Frag']

COUNT_METRICS = {
    'Dets', 'GT_Dets', 'IDs', 'GT_IDs', 'CLR_TP', 'CLR_FN', 'CLR_FP',
    'IDSW', 'Frag', 'MT', 'PT', 'ML',
}

STATUS_LABELS = {
    'both': 'compared',
    'only_v1': 'only in version 1',
    'only_v2': 'only in version 2',
}


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _metric_decimals(metric: str) -> int:
    return 0 if metric in COUNT_METRICS else 4


def _fmt_value(value, metric: str) -> str:
    if _is_missing(value):
        return 'MISS'
    return f"{value:.{_metric_decimals(metric)}f}"


def _fmt_delta(value, metric: str) -> str:
    if _is_missing(value):
        return '-'
    sign = '+' if value > 0 else ''
    return f"{sign}{value:.{_metric_decimals(metric)}f}"


def _fmt_pct(value) -> str:
    if _is_missing(value):
        return '-'
    sign = '+' if value > 0 else ''
    return f"{sign}{value:.2f}%"


class MultiReportGenerator:
    """Generate terminal/HTML/JSON reports from a MultiTrackerComparator."""

    def __init__(self, comparator, output_dir: Optional[str] = None):
        """
        Args:
            comparator: MultiTrackerComparator with compare() already run
            output_dir: Directory for HTML/JSON outputs (optional)
        """
        self.comparator = comparator
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console() if RICH_AVAILABLE else None

    # ---- shared data helpers ----

    def _combined_rows(self, tracker: str) -> pd.DataFrame:
        comparison = self.comparator.results[tracker]['comparison']
        return comparison[comparison['sequence'] == 'COMBINED']

    def _combined_metric(self, tracker: str, metric: str) -> Optional[pd.Series]:
        combined = self._combined_rows(tracker)
        rows = combined[combined['metric'] == metric]
        return rows.iloc[0] if len(rows) else None

    def _sequences(self, tracker: str) -> List[str]:
        comparison = self.comparator.results[tracker]['comparison']
        return sorted(s for s in comparison['sequence'].unique() if s != 'COMBINED')

    def _available_key_metrics(self) -> List[str]:
        available = set()
        for result in self.comparator.results.values():
            available.update(result['comparison']['metric'].unique())
        return [m for m in KEY_METRICS if m in available]

    # ---- terminal report ----

    def generate_terminal_report(self, detailed: bool = False):
        """Print the multi-tracker comparison to the terminal."""
        summary = self.comparator.get_summary()
        if RICH_AVAILABLE:
            self._terminal_rich(summary, detailed)
        else:
            self._terminal_plain(summary, detailed)

    def _terminal_rich(self, summary: Dict, detailed: bool):
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold cyan]TrackEval Multi-Tracker Comparison[/bold cyan]\n"
            f"[yellow]{summary['version1']}[/yellow] vs [green]{summary['version2']}[/green]",
            border_style="bold blue"
        ))

        info = Table(title="Summary", box=box.ROUNDED, show_header=True)
        info.add_column("Item", style="cyan", width=30)
        info.add_column("Value", style="white")
        info.add_row("Trackers Compared", str(summary['trackers_compared']))
        info.add_row("Improvements (COMBINED)", f"[green]{summary['improvements']}[/green]")
        info.add_row("Regressions (COMBINED)", f"[red]{summary['regressions']}[/red]")
        if summary['only_in_v1']:
            info.add_row(f"Only in {summary['version1']}",
                         f"[yellow]{', '.join(summary['only_in_v1'])}[/yellow]")
        if summary['only_in_v2']:
            info.add_row(f"Only in {summary['version2']}",
                         f"[yellow]{', '.join(summary['only_in_v2'])}[/yellow]")
        self.console.print(info)

        self._rich_overview_table(summary)

        for tracker, result in self.comparator.results.items():
            self._rich_tracker_table(tracker, result)
            if detailed:
                self._rich_sequence_tables(tracker, result)

    def _rich_overview_table(self, summary: Dict):
        metrics = self._available_key_metrics()[:3]
        if not metrics:
            return

        table = Table(title="Overview (COMBINED)", box=box.ROUNDED, show_header=True)
        table.add_column("Tracker", style="cyan")
        table.add_column("Status")
        for metric in metrics:
            name = get_metric_display_name(metric).split('(')[0].strip()
            table.add_column(f"{name} v1", justify="right", style="yellow")
            table.add_column(f"{name} v2", justify="right", style="green")
            table.add_column("Delta", justify="right")

        for tracker, result in self.comparator.results.items():
            cells = [tracker, STATUS_LABELS[result['status']]]
            for metric in metrics:
                row = self._combined_metric(tracker, metric)
                if row is None:
                    cells.extend(['[yellow]MISS[/yellow]', '[yellow]MISS[/yellow]', '-'])
                    continue
                cells.append(self._rich_value(row['v1_value'], metric))
                cells.append(self._rich_value(row['v2_value'], metric))
                cells.append(self._rich_delta(row, metric))
            table.add_row(*cells)

        self.console.print()
        self.console.print(table)

    def _rich_value(self, value, metric: str) -> str:
        text = _fmt_value(value, metric)
        return f"[yellow]{text}[/yellow]" if text == 'MISS' else text

    def _rich_delta(self, row, metric: str) -> str:
        text = _fmt_delta(row['delta'], metric)
        if text == '-':
            return text
        color = 'green' if row['improved'] else 'red'
        return f"[{color}]{text}[/{color}]"

    def _rich_tracker_table(self, tracker: str, result: Dict):
        status = STATUS_LABELS[result['status']]
        table = Table(
            title=f"Tracker: {tracker} ({status})",
            box=box.ROUNDED,
            show_header=True
        )
        table.add_column("Metric", style="cyan", width=32)
        table.add_column(self.comparator.version1, justify="right", style="yellow")
        table.add_column(self.comparator.version2, justify="right", style="green")
        table.add_column("Delta", justify="right")
        table.add_column("Change (%)", justify="right")

        for _, row in self._combined_rows(tracker).iterrows():
            metric = row['metric']
            delta_text = self._rich_delta(row, metric)
            pct_text = _fmt_pct(row['delta_pct'])
            if pct_text != '-':
                color = 'green' if row['improved'] else 'red'
                pct_text = f"[{color}]{pct_text}[/{color}]"
            table.add_row(
                get_metric_display_name(metric),
                self._rich_value(row['v1_value'], metric),
                self._rich_value(row['v2_value'], metric),
                delta_text,
                pct_text
            )

        seq_notes = []
        if result['seq_only_in_v1']:
            seq_notes.append(f"sequences only in v1: {', '.join(result['seq_only_in_v1'])}")
        if result['seq_only_in_v2']:
            seq_notes.append(f"sequences only in v2: {', '.join(result['seq_only_in_v2'])}")

        self.console.print()
        self.console.print(table)
        if seq_notes:
            self.console.print(f"[yellow]  Note: {'; '.join(seq_notes)}[/yellow]")

    def _rich_sequence_tables(self, tracker: str, result: Dict):
        comparison = result['comparison']
        for seq in self._sequences(tracker):
            seq_data = comparison[comparison['sequence'] == seq]
            table = Table(title=f"{tracker} / {seq}", box=box.SIMPLE, show_header=True)
            table.add_column("Metric", style="cyan", width=32)
            table.add_column(self.comparator.version1, justify="right")
            table.add_column(self.comparator.version2, justify="right")
            table.add_column("Delta", justify="right")
            for _, row in seq_data.iterrows():
                metric = row['metric']
                table.add_row(
                    get_metric_display_name(metric),
                    self._rich_value(row['v1_value'], metric),
                    self._rich_value(row['v2_value'], metric),
                    self._rich_delta(row, metric)
                )
            self.console.print()
            self.console.print(table)

    def _terminal_plain(self, summary: Dict, detailed: bool):
        print("\n" + "=" * 78)
        print("  TrackEval Multi-Tracker Comparison")
        print(f"  {summary['version1']} vs {summary['version2']}")
        print("=" * 78)

        print("\nSummary:")
        print(f"  Trackers Compared: {summary['trackers_compared']}")
        print(f"  Improvements (COMBINED): {summary['improvements']}")
        print(f"  Regressions (COMBINED): {summary['regressions']}")
        if summary['only_in_v1']:
            print(f"  Only in {summary['version1']}: {', '.join(summary['only_in_v1'])}")
        if summary['only_in_v2']:
            print(f"  Only in {summary['version2']}: {', '.join(summary['only_in_v2'])}")

        for tracker, result in self.comparator.results.items():
            status = STATUS_LABELS[result['status']]
            print(f"\nTracker: {tracker} ({status})")
            print(f"{'Metric':<34} {'v1':>12} {'v2':>12} {'Delta':>12} {'Change':>10}")
            print("-" * 84)
            for _, row in self._combined_rows(tracker).iterrows():
                metric = row['metric']
                print(
                    f"{metric:<34} "
                    f"{_fmt_value(row['v1_value'], metric):>12} "
                    f"{_fmt_value(row['v2_value'], metric):>12} "
                    f"{_fmt_delta(row['delta'], metric):>12} "
                    f"{_fmt_pct(row['delta_pct']):>10}"
                )

            if result['seq_only_in_v1']:
                print(f"  Sequences only in v1: {', '.join(result['seq_only_in_v1'])}")
            if result['seq_only_in_v2']:
                print(f"  Sequences only in v2: {', '.join(result['seq_only_in_v2'])}")

            if detailed:
                comparison = result['comparison']
                for seq in self._sequences(tracker):
                    print(f"\n  {seq}:")
                    seq_data = comparison[comparison['sequence'] == seq]
                    for _, row in seq_data.iterrows():
                        metric = row['metric']
                        print(
                            f"    {metric:<32} "
                            f"{_fmt_value(row['v1_value'], metric):>12} "
                            f"{_fmt_value(row['v2_value'], metric):>12} "
                            f"{_fmt_delta(row['delta'], metric):>12}"
                        )

    # ---- JSON report ----

    def generate_json_report(self, output_path: Optional[str] = None) -> Dict:
        """Write a JSON report; returns the payload."""
        summary = self.comparator.get_summary()
        payload = {
            'generated_at': datetime.now().isoformat(),
            **summary,
            'trackers': {},
        }
        for tracker, result in self.comparator.results.items():
            records = [
                {key: (None if _is_missing(value) else value) for key, value in record.items()}
                for record in result['comparison'].to_dict(orient='records')
            ]
            payload['trackers'][tracker] = {
                'status': result['status'],
                'seq_only_in_v1': result['seq_only_in_v1'],
                'seq_only_in_v2': result['seq_only_in_v2'],
                'comparison': records,
            }

        if output_path:
            output_file = Path(output_path)
        elif self.output_dir:
            output_file = self.output_dir / 'multi_comparison_report.json'
        else:
            output_file = None

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(payload, f, indent=2)
            print(f"Saved multi-tracker JSON report: {output_file.resolve()}")

        return payload

    # ---- HTML report ----

    def generate_html_report(self, output_path: Optional[str] = None):
        """Write a self-contained HTML report."""
        if output_path:
            output_file = Path(output_path)
        elif self.output_dir:
            output_file = self.output_dir / 'multi_comparison_report.html'
        else:
            output_file = Path('multi_comparison_report.html')

        with open(output_file, 'w') as f:
            f.write(self._build_html())
        print(f"Saved multi-tracker HTML report: {output_file.resolve()}")

    def _html_value_cell(self, value, metric: str) -> str:
        if _is_missing(value):
            return '<td><span class="miss-badge">MISS</span></td>'
        return f"<td>{_fmt_value(value, metric)}</td>"

    def _html_comparison_row(self, row) -> str:
        metric = row['metric']
        if row['improved'] is None:
            row_class = 'incomplete'
        elif row['improved']:
            row_class = 'improved'
        else:
            row_class = 'regressed'
        return (
            f'<tr class="{row_class}">'
            f"<td>{escape(get_metric_display_name(metric))}</td>"
            f"{self._html_value_cell(row['v1_value'], metric)}"
            f"{self._html_value_cell(row['v2_value'], metric)}"
            f"<td>{_fmt_delta(row['delta'], metric)}</td>"
            f"<td>{_fmt_pct(row['delta_pct'])}</td>"
            "</tr>"
        )

    def _html_metric_table(self, data: pd.DataFrame) -> str:
        rows = ''.join(self._html_comparison_row(row) for _, row in data.iterrows())
        return f"""
        <table>
            <thead><tr>
                <th>Metric</th>
                <th>{escape(self.comparator.version1)}</th>
                <th>{escape(self.comparator.version2)}</th>
                <th>Delta</th>
                <th>Change (%)</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """

    def _html_overview_tab(self, summary: Dict) -> str:
        metrics = self._available_key_metrics()
        header_cells = ''.join(
            f"<th>{escape(get_metric_display_name(m).split('(')[0].strip())} v1</th>"
            f"<th>v2</th><th>Delta</th>"
            for m in metrics
        )
        body_rows = []
        for tracker, result in self.comparator.results.items():
            cells = [f"<td>{escape(tracker)}</td>",
                     f"<td>{escape(STATUS_LABELS[result['status']])}</td>"]
            for metric in metrics:
                row = self._combined_metric(tracker, metric)
                if row is None:
                    cells.append('<td><span class="miss-badge">MISS</span></td>' * 2 + '<td>-</td>')
                    continue
                cells.append(self._html_value_cell(row['v1_value'], metric))
                cells.append(self._html_value_cell(row['v2_value'], metric))
                delta_class = ''
                if row['improved'] is True:
                    delta_class = ' class="delta-up"'
                elif row['improved'] is False:
                    delta_class = ' class="delta-down"'
                cells.append(f"<td{delta_class}>{_fmt_delta(row['delta'], metric)}</td>")
            body_rows.append(f"<tr>{''.join(cells)}</tr>")

        return f"""
        <h2>Overview (COMBINED)</h2>
        <table>
            <thead><tr><th>Tracker</th><th>Status</th>{header_cells}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
        """

    def _html_tracker_tab(self, tracker: str, result: Dict) -> str:
        notes = []
        if result['status'] == 'only_v1':
            notes.append(f"Tracker exists only in {self.comparator.version1}; values are shown without comparison.")
        elif result['status'] == 'only_v2':
            notes.append(f"Tracker exists only in {self.comparator.version2}; values are shown without comparison.")
        if result['seq_only_in_v1']:
            notes.append(f"Sequences only in {self.comparator.version1}: {', '.join(result['seq_only_in_v1'])}")
        if result['seq_only_in_v2']:
            notes.append(f"Sequences only in {self.comparator.version2}: {', '.join(result['seq_only_in_v2'])}")
        notes_html = ''.join(f'<p class="note">{escape(n)}</p>' for n in notes)

        combined_html = self._html_metric_table(self._combined_rows(tracker))

        comparison = result['comparison']
        seq_sections = []
        for seq in self._sequences(tracker):
            seq_data = comparison[comparison['sequence'] == seq]
            seq_sections.append(f"""
            <details>
                <summary>{escape(seq)}</summary>
                {self._html_metric_table(seq_data)}
            </details>
            """)

        return f"""
        <h2>{escape(tracker)} <span class="status-badge">{escape(STATUS_LABELS[result['status']])}</span></h2>
        {notes_html}
        <h3>Combined Metrics</h3>
        {combined_html}
        <h3>Per-Sequence Metrics</h3>
        {''.join(seq_sections) or '<p class="note">No sequence data.</p>'}
        """

    def _build_html(self) -> str:
        summary = self.comparator.get_summary()

        tab_buttons = ['<button class="tab-button" onclick="showTab(event, \'tab-overview\')">Overview</button>']
        tab_contents = [f'<div id="tab-overview" class="tab-content">{self._html_overview_tab(summary)}</div>']
        for idx, (tracker, result) in enumerate(self.comparator.results.items()):
            tab_buttons.append(
                f'<button class="tab-button" onclick="showTab(event, \'tab-tracker-{idx}\')">{escape(tracker)}</button>'
            )
            tab_contents.append(
                f'<div id="tab-tracker-{idx}" class="tab-content">{self._html_tracker_tab(tracker, result)}</div>'
            )

        only_v1 = ', '.join(summary['only_in_v1']) or '-'
        only_v2 = ', '.join(summary['only_in_v2']) or '-'

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Multi-Tracker Comparison Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
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
    .header h1 {{ font-size: 2.2em; margin-bottom: 10px; }}
    .summary {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px 40px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 20px;
    }}
    .summary-item {{ display: flex; flex-direction: column; gap: 5px; }}
    .summary-label {{ font-size: 0.9em; opacity: 0.9; }}
    .summary-value {{ font-size: 1.2em; font-weight: bold; }}
    .tab-navigation {{ display: flex; background-color: #34495e; overflow-x: auto; }}
    .tab-button {{
        padding: 15px 25px;
        background-color: transparent;
        border: none;
        color: rgba(255,255,255,0.7);
        cursor: pointer;
        font-size: 1em;
        font-weight: 600;
        white-space: nowrap;
        border-bottom: 3px solid transparent;
    }}
    .tab-button:hover {{ background-color: rgba(255,255,255,0.1); color: white; }}
    .tab-button.active {{ background-color: white; color: #2c3e50; border-bottom: 3px solid #3498db; }}
    .tab-content {{ display: none; padding: 40px; }}
    .tab-content.active {{ display: block; }}
    h2 {{ color: #2c3e50; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #ecf0f1; }}
    h3 {{ color: #7f8c8d; margin: 25px 0 15px; }}
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
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
        font-size: 0.9em;
    }}
    td {{ padding: 10px 15px; border-bottom: 1px solid #ecf0f1; }}
    tr:hover {{ background-color: #f8f9fa; }}
    .improved {{ background-color: #d5f4e6; border-left: 4px solid #27ae60; }}
    .regressed {{ background-color: #fadbd8; border-left: 4px solid #e74c3c; }}
    .incomplete {{ background-color: #fef9e7; border-left: 4px solid #f1c40f; }}
    .delta-up {{ color: #27ae60; font-weight: 600; }}
    .delta-down {{ color: #e74c3c; font-weight: 600; }}
    .miss-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        background-color: #f1c40f;
        color: #5d4a00;
        font-size: 0.8em;
        font-weight: 700;
    }}
    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        background-color: #ecf0f1;
        color: #34495e;
        font-size: 0.55em;
        font-weight: 600;
        vertical-align: middle;
    }}
    .note {{ color: #b7791f; background: #fef9e7; padding: 10px 14px; border-radius: 6px; margin-bottom: 10px; }}
    details {{ margin: 12px 0; background: #f8f9fa; border-radius: 8px; padding: 10px 16px; }}
    summary {{ cursor: pointer; font-weight: 600; color: #2c3e50; padding: 6px 0; }}
    .timestamp {{ color: #95a5a6; font-size: 0.9em; margin-top: 30px; text-align: right; font-style: italic; }}
</style>
<script>
    function showTab(evt, tabId) {{
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        evt.currentTarget.classList.add('active');
    }}
    window.addEventListener('DOMContentLoaded', () => document.querySelector('.tab-button').click());
</script>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>TrackEval Multi-Tracker Comparison</h1>
        <p>{escape(summary['version1'])} vs {escape(summary['version2'])}</p>
    </div>
    <div class="summary">
        <div class="summary-item"><span class="summary-label">Version 1</span><span class="summary-value">{escape(summary['version1'])}</span></div>
        <div class="summary-item"><span class="summary-label">Version 2</span><span class="summary-value">{escape(summary['version2'])}</span></div>
        <div class="summary-item"><span class="summary-label">Trackers Compared</span><span class="summary-value">{summary['trackers_compared']}</span></div>
        <div class="summary-item"><span class="summary-label">Only in {escape(summary['version1'])}</span><span class="summary-value">{escape(only_v1)}</span></div>
        <div class="summary-item"><span class="summary-label">Only in {escape(summary['version2'])}</span><span class="summary-value">{escape(only_v2)}</span></div>
    </div>
    <div class="tab-navigation">{''.join(tab_buttons)}</div>
    {''.join(tab_contents)}
    <div class="timestamp" style="padding: 0 40px 30px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
</body>
</html>
"""
