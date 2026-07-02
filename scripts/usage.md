 python3 -m scripts.compare_versions_minimal \
  --rank-prefix v1.1.0-tracker-compare \
  --base-dir output/reports \
  --output output/comparison/tracker-ranking \
  --html \
  --json

# Compare ALL trackers of two versions (missing tracker/sequence -> MISS)
 python3 -m scripts.compare_versions_minimal v1.1.0 v1.1.1 \
  --groups primary counts \
  --output output/comparison/v1.1.0-v1.1.1-all \
  --html \
  --json
