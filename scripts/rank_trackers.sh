#!/usr/bin/env bash
set -euo pipefail

VERSION=${1:-}
METRIC=${2:-HOTA___AUC}

if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <version> [metric]"
  echo "Example: $0 v1.1.0"
  echo "Example: $0 v1.1.0 IDF1"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

python3 -m scripts.compare_versions_minimal \
  --rank-prefix "${VERSION}-tracker-compare" \
  --rank-metric "$METRIC" \
  --base-dir output/reports \
  --output output/comparison/tracker-ranking \
  --html \
  --json
