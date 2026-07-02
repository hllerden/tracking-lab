#!/bin/bash

###############################################################################
# Version Comparison Script
#
# Compares benchmark_summary.json performance between two benchmark versions.
#
# Usage:
#   ./scripts/compare_versions.sh v1.0.0 v1.0.1
#   ./scripts/compare_versions.sh v1.0.0 v1.0.1 tracker-only
###############################################################################

set -euo pipefail
export LC_NUMERIC=C

RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

V1=${1:-}
V2=${2:-}
MODE=${3:-"tracker-only"}

if [ -z "$V1" ] || [ -z "$V2" ]; then
    echo -e "${RED}Usage: $0 <version1> <version2> [mode]${NC}"
    echo -e "Example: $0 v1.0.0 v1.0.1 tracker-only"
    echo -e "Tracker ranking: ./scripts/rank_trackers.sh v1.1.0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
V1_DIR="$PROJECT_DIR/output/$V1/$MODE"
V2_DIR="$PROJECT_DIR/output/$V2/$MODE"

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  MOT17 Version Comparison${NC}"
echo -e "${BLUE}  ${V1} vs ${V2}${NC}"
echo -e "${BLUE}  Mode: $MODE${NC}"
echo -e "${BLUE}=====================================${NC}\n"

if [ ! -d "$V1_DIR" ]; then
    echo -e "${RED}Error: Version $V1 not found ($V1_DIR)${NC}"
    exit 1
fi

if [ ! -d "$V2_DIR" ]; then
    echo -e "${RED}Error: Version $V2 not found ($V2_DIR)${NC}"
    exit 1
fi

V1_JSON="$V1_DIR/benchmark_summary.json"
V2_JSON="$V2_DIR/benchmark_summary.json"

if [ ! -f "$V1_JSON" ] || [ ! -f "$V2_JSON" ]; then
    echo -e "${RED}Error: Summary JSON files not found${NC}"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq not installed. Showing raw JSON summaries instead.${NC}\n"
    echo -e "${CYAN}=== $V1 ===${NC}"
    cat "$V1_JSON"
    echo -e "\n${CYAN}=== $V2 ===${NC}"
    cat "$V2_JSON"
    exit 0
fi

V1_TESTS=$(jq -r '.successful_tests' "$V1_JSON")
V2_TESTS=$(jq -r '.successful_tests' "$V2_JSON")
V1_FAILED=$(jq -r '.failed_tests // 0' "$V1_JSON")
V2_FAILED=$(jq -r '.failed_tests // 0' "$V2_JSON")

summary_value() {
    local json_file=$1
    local field=$2

    jq -r --arg field "$field" '
        if .[$field] != null then
            .[$field]
        elif (.results // [] | length) > 0 then
            [.results[] | select(.success == true)] as $results |
            if ($results | length) == 0 then
                0
            elif $field == "avg_fps" then
                (($results | map(.total_frames // 0) | add) * 1000) /
                ($results | map(.processing_time_ms // 0) | add)
            elif $field == "total_duration_seconds" then
                ($results | map(.processing_time_ms // 0) | add) / 1000
            else
                0
            end
        else
            0
        end
    ' "$json_file"
}

calc_delta() {
    jq -n --argjson v1 "$1" --argjson v2 "$2" '$v2 - $v1'
}

calc_percent() {
    jq -n --argjson base "$1" --argjson delta "$2" '
        if $base == 0 then 0 else ($delta / $base) * 100 end
    '
}

V1_FPS=$(summary_value "$V1_JSON" "avg_fps")
V2_FPS=$(summary_value "$V2_JSON" "avg_fps")
V1_TIME=$(summary_value "$V1_JSON" "total_duration_seconds")
V2_TIME=$(summary_value "$V2_JSON" "total_duration_seconds")

FPS_DELTA=$(calc_delta "$V1_FPS" "$V2_FPS")
FPS_PERCENT=$(calc_percent "$V1_FPS" "$FPS_DELTA")
TIME_DELTA=$(calc_delta "$V1_TIME" "$V2_TIME")
TIME_PERCENT=$(calc_percent "$V1_TIME" "$TIME_DELTA")

echo -e "${CYAN}Performance Comparison:${NC}\n"
printf "┌─────────────────────┬──────────────┬──────────────┬──────────────┐\n"
printf "│ %-19s │ %-12s │ %-12s │ %-12s │\n" "Metric" "$V1" "$V2" "Delta"
printf "├─────────────────────┼──────────────┼──────────────┼──────────────┤\n"
printf "│ %-19s │ %12s │ %12s │ %12s │\n" "Successful Tests" "$V1_TESTS" "$V2_TESTS" "$(($V2_TESTS - $V1_TESTS))"
printf "│ %-19s │ %12s │ %12s │ %12s │\n" "Failed Tests" "$V1_FAILED" "$V2_FAILED" "$(($V2_FAILED - $V1_FAILED))"
printf "│ %-19s │ %10.1f │ %10.1f │ %+10.1f │\n" "Avg FPS" "$V1_FPS" "$V2_FPS" "$FPS_DELTA"
printf "│ %-19s │ %12s │ %12s │ %+11.1f%% │\n" "  (Improvement)" "" "" "$FPS_PERCENT"
printf "│ %-19s │ %10.1fs │ %10.1fs │ %+10.1fs │\n" "Total Time" "$V1_TIME" "$V2_TIME" "$TIME_DELTA"
printf "│ %-19s │ %12s │ %12s │ %+11.1f%% │\n" "  (Change)" "" "" "$TIME_PERCENT"
printf "└─────────────────────┴──────────────┴──────────────┴──────────────┘\n"

if jq -e '(.results // []) | length > 0' "$V1_JSON" > /dev/null && \
   jq -e '(.results // []) | length > 0' "$V2_JSON" > /dev/null; then
    declare -A V1_GROUP_FPS V2_GROUP_FPS V1_GROUP_TIME V2_GROUP_TIME V1_GROUP_TRACKS V2_GROUP_TRACKS

    load_groups() {
        local json_file=$1
        local prefix=$2

        while IFS=$'\t' read -r name fps time tracks; do
            eval "${prefix}_GROUP_FPS[\"\$name\"]=\"\$fps\""
            eval "${prefix}_GROUP_TIME[\"\$name\"]=\"\$time\""
            eval "${prefix}_GROUP_TRACKS[\"\$name\"]=\"\$tracks\""
        done < <(jq -r '
            [.results[] | select(.success == true)]
            | group_by((.tracker // "unknown") + "|" + (.reid // "unknown"))
            | .[]
            | {
                name: ((.[0].tracker // "unknown") + "-" + (.[0].reid // "unknown")),
                fps: (((map(.total_frames // 0) | add) * 1000) / (map(.processing_time_ms // 0) | add)),
                time: ((map(.processing_time_ms // 0) | add) / 1000),
                tracks: (map(.total_tracks // 0) | add)
              }
            | [.name, .fps, .time, .tracks]
            | @tsv
        ' "$json_file")
    }

    load_groups "$V1_JSON" "V1"
    load_groups "$V2_JSON" "V2"

    echo -e "\n${CYAN}Tracker/ReID Breakdown:${NC}\n"
    printf "┌──────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐\n"
    printf "│ %-20s │ %8s │ %8s │ %8s │ %8s │ %8s │\n" "Tracker" "$V1 FPS" "$V2 FPS" "FPS Δ%" "$V1 trk" "$V2 trk"
    printf "├──────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤\n"

    for name in "${!V1_GROUP_FPS[@]}"; do
        if [[ -z "${V2_GROUP_FPS[$name]:-}" ]]; then
            continue
        fi

        group_delta=$(calc_delta "${V1_GROUP_FPS[$name]}" "${V2_GROUP_FPS[$name]}")
        group_percent=$(calc_percent "${V1_GROUP_FPS[$name]}" "$group_delta")
        printf "│ %-20s │ %8.1f │ %8.1f │ %+7.1f%% │ %8.0f │ %8.0f │\n" \
            "$name" \
            "${V1_GROUP_FPS[$name]}" \
            "${V2_GROUP_FPS[$name]}" \
            "$group_percent" \
            "${V1_GROUP_TRACKS[$name]}" \
            "${V2_GROUP_TRACKS[$name]}"
    done | sort

    printf "└──────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘\n"
fi

echo -e "\n${YELLOW}Note: For HOTA/MOTA/IDF1 metrics, compare matching TrackEval report names:${NC}"
echo -e "  python3 -m scripts.compare_versions_minimal \\"
echo -e "    ${V1}-${MODE}-botsort-reid-off ${V2}-${MODE}-botsort-reid-off \\"
echo -e "    --base-dir output/reports --groups primary detection counts \\"
echo -e "    --html --json --output output/comparison/${V1}-${V2}/botsort-reid-off\n"
