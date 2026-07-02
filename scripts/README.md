# Scripts

This directory contains benchmark helper scripts and TrackEval comparison tools.

## v1.1.0 vs v1.1.1 comparison

The prepared comparison is for tracker-compare outputs:

```bash
./scripts/compare_versions.sh v1.1.0 v1.1.1 tracker-compare
```

The script reads:

- `output/v1.1.0/tracker-compare/benchmark_summary.json`
- `output/v1.1.1/tracker-compare/benchmark_summary.json`

It compares successful/failed test counts, aggregate FPS, total benchmark time, and per tracker/ReID runtime summary. For the current data:

| Metric | v1.1.0 | v1.1.1 | Delta |
|--------|------:|------:|------:|
| Successful tests | 28 | 28 | 0 |
| Failed tests | 0 | 0 | 0 |
| Aggregate FPS | 81.7 | 80.7 | -1.0 (-1.2%) |
| Total benchmark time | 260.4s | 263.6s | +3.2s (+1.2%) |

Per tracker runtime:

| Tracker | v1.1.0 FPS | v1.1.1 FPS | FPS delta | v1.1.0 tracks | v1.1.1 tracks |
|---------|-----------:|-----------:|----------:|--------------:|--------------:|
| botsort-reid-off | 148.1 | 134.0 | -9.5% | 150 | 139 |
| botsort-reid-on | 46.4 | 47.1 | +1.6% | 150 | 134 |
| bytetrack-reid-off | 336.6 | 312.2 | -7.2% | 142 | 142 |
| bytetrack-reid-on | 56.5 | 56.6 | +0.0% | 142 | 142 |

## TrackEval metric comparison

Use `scripts.compare_versions_minimal` for HOTA/MOTA/IDF1 and detailed TrackEval metrics.

### Multi-tracker comparison (version prefix)

Passing version prefixes compares every tracker report discovered under `output/reports` for both versions:

```bash
python3 -m scripts.compare_versions_minimal v1.1.0 v1.1.1 \
    --groups primary counts \
    --output output/comparison/v1.1.0-v1.1.1-all \
    --html --json
```

Behavior:

- A directory belongs to a version when its name is exactly the version or starts with `<version>-` and contains `pedestrian_detailed.csv`. The remainder of the name is the tracker name (e.g. `v1.1.0-tracker-compare-botsort-reid-off` yields `tracker-compare-botsort-reid-off`).
- Trackers present in both versions are compared per sequence (union of sequences).
- Trackers or sequences present in only one version are reported with their values and `MISS` on the other side; no delta is computed for them.
- Scales to any number of trackers and sequences.
- `--trackers <name...>` restricts the comparison to specific tracker names.
- `--detailed` adds per-sequence tables to the terminal output.

Generated reports: `multi_comparison_report.html` and `multi_comparison_report.json` in the `--output` directory.

### Single-report comparison

Report names matching exact directories under `output/reports` keep the original single comparison flow.

Example for BoT-SORT without ReID:

```bash
python3 -m scripts.compare_versions_minimal \
    v1.1.0-tracker-compare-botsort-reid-off \
    v1.1.1-tracker-compare-botsort-reid-off \
    --base-dir output/reports \
    --groups primary detection counts \
    --output output/comparison/v1.1.0-v1.1.1/botsort-reid-off \
    --html --json
```

Current COMBINED deltas:

| Tracker | HOTA delta | MOTA delta | IDF1 delta | Notes |
|---------|-----------:|-----------:|-----------:|-------|
| botsort-reid-off | +0.0156 (+3.23%) | +0.0019 (+0.39%) | +0.0276 (+5.12%) | Main v1.1.1 improvement |
| botsort-reid-on | +0.0029 (+0.59%) | -0.0068 (-1.38%) | +0.0115 (+2.13%) | Better HOTA/IDF1, lower MOTA |
| bytetrack-reid-off | 0.0000 (0.00%) | 0.0000 (0.00%) | 0.0000 (0.00%) | TrackEval metrics unchanged |
| bytetrack-reid-on | 0.0000 (0.00%) | 0.0000 (0.00%) | 0.0000 (0.00%) | TrackEval metrics unchanged |

Generated reports:

- `output/comparison/v1.1.0-v1.1.1/botsort-reid-off/comparison_report.html`
- `output/comparison/v1.1.0-v1.1.1/botsort-reid-off/comparison_report.json`

## Tracker ranking

Rank all tracker variants for a version:

```bash
python3 -m scripts.compare_versions_minimal \
    --rank-prefix v1.1.1-tracker-compare \
    --base-dir output/reports \
    --rank-metric HOTA___AUC \
    --output output/comparison/v1.1.1-tracker-ranking \
    --html --json
```

Current v1.1.1 ranking:

| Rank | Tracker | HOTA | MOTA | IDF1 | IDSW | Frag |
|-----:|---------|-----:|-----:|-----:|-----:|-----:|
| 1 | botsort-reid-off | 0.498 | 0.494 | 0.568 | 716 | 1122 |
| 2 | botsort-reid-on | 0.485 | 0.485 | 0.550 | 864 | 1204 |
| 3 | bytetrack-reid-on | 0.443 | 0.253 | 0.487 | 1113 | 1121 |
| 4 | bytetrack-reid-off | 0.438 | 0.245 | 0.478 | 1177 | 1125 |

Generated reports:

- `output/comparison/v1.1.1-tracker-ranking/tracker_ranking_report.html`
- `output/comparison/v1.1.1-tracker-ranking/tracker_ranking_report.json`

Best by HOTA is `botsort-reid-off`. It wins 3 of the 7 per-sequence comparisons.
