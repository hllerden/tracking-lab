# TrackEval Benchmark Comparison Tool

A comprehensive Python tool for comparing MOT17 TrackEval benchmark results across different versions. Supports flexible metric selection, multiple output formats, and rich visualizations.

## Features

- **OOP Architecture**: Modular, extensible design with clear separation of concerns
- **Configurable Metrics**: Easy-to-customize metric groups and individual metrics
- **Error Handling**: Robust handling of missing versions, sequences, and metrics
- **Multiple Output Formats**: Terminal, JSON, HTML reports
- **Rich Visualizations**: Bar charts, heatmaps, radar charts, improvement/regression plots
- **Sequence Mismatch Handling**: Flexible modes for comparing versions with different sequences

## Installation

### Dependencies

```bash
# Required
pip install pandas

# Optional (for visualizations and rich output)
pip install matplotlib seaborn rich
```

### Quick Start

```bash
# List available versions
python -m scripts.compare_versions_minimal --list-versions

# Basic comparison (terminal output only)
python -m scripts.compare_versions_minimal v1.0.1-minimal v1.0.2-minimal

# Generate beautiful HTML report with embedded charts
python -m scripts.compare_versions_minimal v1.0.1-minimal v1.0.2-minimal \
    --output output/comparison/ \
    --html

# Then open the HTML file in your browser:
# output/comparison/comparison_report.html
#
# The HTML report includes:
# ✓ Summary statistics
# ✓ Combined metrics comparison
# ✓ All charts (bar, heatmap, radar, improvements)
# ✓ Per-sequence detailed tables
# ✓ Beautiful gradient design
# ✓ Single self-contained file (no external dependencies)
```

## Usage

### Command-Line Interface

#### Basic Comparison

```bash
# Compare two versions with primary metrics (HOTA, MOTA, IDF1)
python -m scripts.compare_versions_minimal v1.0.1-minimal v1.0.2-minimal
```

#### Custom Metric Groups

```bash
# Compare specific metric groups
python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 \
    --groups primary detection counts

# Available groups:
#   - primary: HOTA, MOTA, IDF1
#   - detection: DetA, DetRe, DetPr, CLR_Re, CLR_Pr
#   - association: AssA, AssRe, AssPr, IDR, IDP
#   - localization: LocA, MOTP
#   - counts: Dets, GT_Dets, IDs, GT_IDs, CLR_TP, CLR_FN, CLR_FP, IDSW, Frag
#   - advanced: sMOTA, CLR_F1, MTR, PTR, MLR, MT, PT, ML
```

#### Custom Metrics

```bash
# Specify individual metrics
python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 \
    --metrics HOTA MOTA IDF1 Dets GT_Dets IDSW Frag

# Compare all available metrics
python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 --all
```

#### Output Formats

```bash
# Generate HTML report (automatically includes charts if matplotlib available)
python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 \
    --output output/comparison/ \
    --html

# The HTML report includes:
# - Summary statistics
# - COMBINED metrics table
# - Embedded charts (base64 encoded PNG images)
# - Per-sequence detailed tables
# - Responsive design with gradient styling

# Generate JSON report (for automation)
python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 \
    --output output/comparison/ \
    --json

# Generate standalone charts (PNG files)
python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 \
    --output output/comparison/ \
    --charts

# Full report with all outputs
python -m scripts.compare_versions_minimal v1.0.1 v1.0.2 \
    --output output/comparison/ \
    --html \
    --json \
    --detailed

# Note: --html automatically generates and embeds charts
# No need to specify --charts separately for HTML reports
```

#### Sequence Mismatch Handling

```bash
# Only compare sequences present in both versions (default)
python -m scripts.compare_versions_minimal v1.0.0 v1.0.3 --mode intersection

# Include all sequences (NaN for missing)
python -m scripts.compare_versions_minimal v1.0.0 v1.0.3 --mode union

# Fail if sequences don't match
python -m scripts.compare_versions_minimal v1.0.0 v1.0.3 --mode strict
```

### Python API

#### Basic Usage

```python
from scripts.compare_versions_minimal import TrackEvalComparator, MetricsConfig

# Create configuration
config = MetricsConfig(groups=['primary', 'counts'])

# Initialize comparator
comparator = TrackEvalComparator('v1.0.1-minimal', 'v1.0.2-minimal', config)

# Load and compare
comparator.load_data()
comparator.validate_data()
comparison_df = comparator.compare()

# Get summary
summary = comparator.get_summary()
print(f"Improvements: {summary['improvements']}")
print(f"Regressions: {summary['regressions']}")
```

#### Advanced Usage

```python
from scripts.compare_versions_minimal import (
    TrackEvalComparator,
    MetricsConfig,
    ReportGenerator,
    ChartGenerator
)

# Custom metric configuration
config = MetricsConfig(
    groups=['primary', 'detection'],
    custom_metrics=['Dets', 'GT_Dets', 'IDSW'],
    exclude_metrics=['DetPr(0)'],
    include_thresholds=True
)

# Perform comparison
comparator = TrackEvalComparator('v1.0.1', 'v1.0.2', config)
comparator.load_data()
comparator.validate_data(mode='intersection')
comparator.compare()

# Get filtered results
improvements = comparator.get_improvements(threshold=0.01)  # Only >1% changes
regressions = comparator.get_regressions(threshold=0.01)

print(f"Significant improvements: {len(improvements)}")
print(f"Significant regressions: {len(regressions)}")

# Generate reports
reporter = ReportGenerator(comparator, output_dir='output/comparison')
reporter.generate_terminal_report(detailed=True)
reporter.generate_all_reports(formats=['json', 'html'])

# Generate visualizations
visualizer = ChartGenerator(
    comparator.comparison,
    'v1.0.1',
    'v1.0.2',
    output_dir='output/comparison'
)

visualizer.generate_bar_chart(['HOTA(0)', 'MOTA', 'IDF1'])
visualizer.generate_heatmap(['HOTA(0)', 'MOTA', 'IDF1'])
visualizer.generate_radar_chart(['HOTA(0)', 'MOTA', 'IDF1'])
visualizer.generate_improvement_chart(top_n=10)
visualizer.close_all()
```

#### Adding Custom Metrics

```python
from scripts.compare_versions_minimal import MetricsConfig

# Create config with custom group
config = MetricsConfig()
config.add_custom_group('my_metrics', [
    'Dets',
    'GT_Dets',
    'IDs',
    'GT_IDs',
    'IDSW',
    'Frag'
])

# Use the custom group
config.groups = ['primary', 'my_metrics']
metrics = config.get_metrics()
print(f"Selected metrics: {metrics}")
```

## Architecture

### Module Structure

```
scripts/compare_versions_minimal/
├── __init__.py          # Package initialization and exports
├── __main__.py          # CLI entry point
├── config.py            # MetricsConfig class (metric configuration)
├── comparator.py        # TrackEvalComparator class (core logic)
├── visualizer.py        # ChartGenerator class (visualizations)
├── reporter.py          # ReportGenerator class (report generation)
├── utils.py             # Helper functions
└── README.md            # This file
```

### Class Responsibilities

#### MetricsConfig
- Defines available metric groups
- Manages metric selection
- Supports custom metrics and groups
- Handles threshold-based metrics

#### TrackEvalComparator
- Loads CSV data from both versions
- Validates data and checks for mismatches
- Calculates metric differences
- Identifies improvements and regressions
- Provides summary statistics

#### ChartGenerator
- Creates bar charts for metric comparison
- Generates heatmaps for multi-sequence analysis
- Produces radar charts for multi-metric visualization
- Shows top improvements/regressions

#### ReportGenerator
- Terminal output (plain text or Rich-formatted)
- JSON export for automation
- HTML reports with embedded tables
- PDF export (optional, if reportlab installed)

## Output Examples

### Terminal Output

```
====================================
  TrackEval Benchmark Comparison
  v1.0.1-minimal vs v1.0.2-minimal
====================================

Summary:
  Total Sequences Compared: 3
  Total Metrics Compared: 3
  Improvements: 2
  Regressions: 1

Key Metrics (COMBINED):
Metric          v1.0.1-minimal  v1.0.2-minimal  Delta        Change (%)
─────────────────────────────────────────────────────────────────────────
HOTA            0.4116          0.4250          +0.0134      +3.26%
MOTA            0.1825          0.1950          +0.0125      +6.85%
IDF1            0.3472          0.3450          -0.0022      -0.63%
```

### JSON Output

```json
{
  "version1": "v1.0.1-minimal",
  "version2": "v1.0.2-minimal",
  "total_sequences_compared": 3,
  "improvements": 2,
  "regressions": 1,
  "HOTA(0)_v1": 0.4116,
  "HOTA(0)_v2": 0.4250,
  "HOTA(0)_delta": 0.0134,
  "HOTA(0)_delta_pct": 3.26,
  "detailed_comparison": [...],
  "top_improvements": [...],
  "top_regressions": [...]
}
```

### Generated Charts

1. **Bar Chart**: Side-by-side comparison of metric values
2. **Heatmap**: Color-coded metric changes across sequences
3. **Radar Chart**: Multi-metric comparison on polar axes
4. **Improvement Chart**: Top improvements and regressions

All charts saved as high-resolution PNG files (300 DPI).

### HTML Report Features

The HTML report (`comparison_report.html`) includes:

1. **Tab-Based Navigation** 🆕:
   - **📈 Overview Tab**: Combined metrics comparison table
   - **📊 Visualizations Tab**: All embedded charts (4 charts)
   - **📋 Per-Sequence Tab**:
     - **Nested sub-tabs** for each sequence (MOT17-02-DPM, MOT17-02-FRCNN, MOT17-02-SDP)
     - Click sequence name to view detailed metrics
     - Each sub-tab shows improvements/regressions count
     - Color-coded badges for quick overview
   - Smooth animations and transitions
   - Clean, organized interface

2. **Beautiful Design**:
   - Gradient background (purple theme)
   - Responsive tables with hover effects
   - Color-coded improvements (green) and regressions (red)
   - Professional typography and spacing
   - Custom scrollbars

3. **Embedded Charts**:
   - All PNG charts embedded as base64 images
   - No external dependencies - single HTML file
   - Charts displayed in responsive containers
   - 4 chart types: Bar, Improvements/Regressions, Heatmap, Radar

4. **Comprehensive Data**:
   - Summary statistics with badges (always visible)
   - COMBINED metrics table (Overview tab)
   - All visualization charts (Visualizations tab)
   - Per-sequence detailed tables (Per-Sequence tab)
   - Emoji indicators for different sequences

5. **Easy Sharing**:
   - Single self-contained HTML file
   - Can be opened directly in any browser
   - No need to share separate image files
   - Interactive tab navigation
   - Professional presentation ready

## Error Handling

### Missing Version

```
Error: Version directory not found: output/reports/v1.0.99-minimal
Available versions: v1.0.1-minimal, v1.0.2-minimal
```

### Sequence Mismatch

```
Warning: Sequence mismatch detected:
  Common sequences: 3
  Only in v1.0.1-minimal: ['MOT17-04-DPM']
  Only in v1.0.2-minimal: ['MOT17-13-SDP']
```

### Missing Metrics

```
Warning: Metrics not found in v1.0.1-minimal: ['CustomMetric']
These metrics will be skipped.
```

## Best Practices

1. **Always validate data first**: Call `validate_data()` before `compare()`
2. **Use intersection mode** for mismatched sequences (default)
3. **Filter noise**: Use `threshold` parameter in `get_improvements()` and `get_regressions()`
4. **Save outputs**: Always specify `--output` when generating charts
5. **Check warnings**: Pay attention to sequence and metric mismatch warnings

## Extending the Tool

### Adding New Metric Groups

Edit `config.py`:

```python
METRIC_GROUPS['my_group'] = [
    'NewMetric1',
    'NewMetric2',
    'NewMetric3'
]
```

### Custom Visualizations

Extend `ChartGenerator` class:

```python
class MyChartGenerator(ChartGenerator):
    def generate_custom_chart(self, ...):
        # Your custom visualization logic
        pass
```

### New Report Formats

Extend `ReportGenerator` class:

```python
class MyReportGenerator(ReportGenerator):
    def generate_pdf_report(self, output_path):
        # PDF generation logic
        pass
```

## Troubleshooting

### Import Errors

```bash
# Install missing dependencies
pip install pandas matplotlib seaborn rich
```

### Chart Generation Issues

Check matplotlib backend:
```python
import matplotlib
print(matplotlib.get_backend())
```

### Permission Errors

Ensure output directory is writable:
```bash
chmod 755 output/comparison/
```

## Version History

- **v1.0.0** (2024-10-21): Initial release
  - OOP architecture
  - Configurable metrics
  - Multiple output formats
  - Rich visualizations

## License

Part of the opencv-yolo project.

## Support

For issues or questions, please refer to the main project documentation.
