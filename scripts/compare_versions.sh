#!/bin/bash

###############################################################################
# Version Comparison Script
#
# Compares tracking performance between two benchmark versions
#
# Usage:
#   ./scripts/compare_versions.sh v1.0.0 v1.0.1
#   ./scripts/compare_versions.sh v1.0.0 v1.0.1 tracker-only
###############################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parameters
V1=$1
V2=$2
MODE=${3:-"tracker-only"}

if [ -z "$V1" ] || [ -z "$V2" ]; then
    echo -e "${RED}Usage: $0 <version1> <version2> [mode]${NC}"
    echo -e "Example: $0 v1.0.0 v1.0.1 tracker-only"
    exit 1
fi

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  MOT17 Version Comparison${NC}"
echo -e "${BLUE}  ${V1} vs ${V2}${NC}"
echo -e "${BLUE}  Mode: $MODE${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Check if versions exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
V1_DIR="$PROJECT_DIR/output/$V1/$MODE"
V2_DIR="$PROJECT_DIR/output/$V2/$MODE"

if [ ! -d "$V1_DIR" ]; then
    echo -e "${RED}Error: Version $V1 not found ($V1_DIR)${NC}"
    exit 1
fi

if [ ! -d "$V2_DIR" ]; then
    echo -e "${RED}Error: Version $V2 not found ($V2_DIR)${NC}"
    exit 1
fi

# Extract metrics from JSON summaries
V1_JSON="$V1_DIR/benchmark_summary.json"
V2_JSON="$V2_DIR/benchmark_summary.json"

if [ ! -f "$V1_JSON" ] || [ ! -f "$V2_JSON" ]; then
    echo -e "${RED}Error: Summary JSON files not found${NC}"
    exit 1
fi

# Use jq to parse JSON (install: sudo apt install jq)
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq not installed. Install with: sudo apt install jq${NC}"
    echo -e "${YELLOW}Showing raw JSON summaries instead:${NC}\n"
    echo -e "${CYAN}=== $V1 ===${NC}"
    cat "$V1_JSON"
    echo -e "\n${CYAN}=== $V2 ===${NC}"
    cat "$V2_JSON"
    exit 0
fi

# Extract metrics
V1_TESTS=$(jq -r '.successful_tests' "$V1_JSON")
V2_TESTS=$(jq -r '.successful_tests' "$V2_JSON")
V1_FPS=$(jq -r '.avg_fps' "$V1_JSON")
V2_FPS=$(jq -r '.avg_fps' "$V2_JSON")
V1_TIME=$(jq -r '.total_duration_seconds' "$V1_JSON")
V2_TIME=$(jq -r '.total_duration_seconds' "$V2_JSON")

# Calculate deltas
FPS_DELTA=$(echo "$V2_FPS - $V1_FPS" | bc -l)
FPS_PERCENT=$(echo "scale=2; ($FPS_DELTA / $V1_FPS) * 100" | bc -l)
TIME_DELTA=$(echo "$V2_TIME - $V1_TIME" | bc -l)
TIME_PERCENT=$(echo "scale=2; ($TIME_DELTA / $V1_TIME) * 100" | bc -l)

# Print comparison table
echo -e "${CYAN}Performance Comparison:${NC}\n"
printf "┌─────────────────────┬──────────────┬──────────────┬──────────────┐\n"
printf "│ %-19s │ %-12s │ %-12s │ %-12s │\n" "Metric" "$V1" "$V2" "Delta"
printf "├─────────────────────┼──────────────┼──────────────┼──────────────┤\n"
printf "│ %-19s │ %12s │ %12s │ %12s │\n" "Successful Tests" "$V1_TESTS" "$V2_TESTS" "$(($V2_TESTS - $V1_TESTS))"
printf "│ %-19s │ %10.1f │ %10.1f │ %+10.1f │\n" "Avg FPS" "$V1_FPS" "$V2_FPS" "$FPS_DELTA"
printf "│ %-19s │ %12s │ %12s │ %+11.1f%% │\n" "  (Improvement)" "" "" "$FPS_PERCENT"
printf "│ %-19s │ %10.1fs │ %10.1fs │ %+10.1fs │\n" "Total Time" "$V1_TIME" "$V2_TIME" "$TIME_DELTA"
printf "│ %-19s │ %12s │ %12s │ %+11.1f%% │\n" "  (Change)" "" "" "$TIME_PERCENT"
printf "└─────────────────────┴──────────────┴──────────────┴──────────────┘\n"

echo -e "\n${YELLOW}Note: For HOTA/MOTA/IDF1 metrics, run TrackEval on both versions:${NC}"
echo -e "  cd output && ./run_trackeval.sh $V1 $MODE"
echo -e "  cd output && ./run_trackeval.sh $V2 $MODE"
echo -e "  Then compare the pedestrian_summary.txt files\n"
