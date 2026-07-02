#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CLEAN=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -clean|--clean)
      CLEAN=true
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo -e "${RED}Error: Bilinmeyen seçenek '$1'${NC}" >&2
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  POSITIONAL_ARGS+=("$@")
fi

set -- "${POSITIONAL_ARGS[@]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACKEVAL_DIR="$PROJECT_DIR/thirdParty/TrackEval"
GT_LINK="$TRACKEVAL_DIR/data/gt/mot_challenge/MOT17-train"
SEQMAPS_FILE="$TRACKEVAL_DIR/data/gt/mot_challenge/seqmaps/MOT17-train.txt"
REPORTS_DIR="$PROJECT_DIR/output/reports"

VERSION="${1:-$(ls -td "$SCRIPT_DIR"/v*/ 2>/dev/null | head -1 | xargs basename 2>/dev/null)}"
MODE="tracker-compare"
SOURCE_DIR="$SCRIPT_DIR/$VERSION/$MODE"

if [[ -z "$VERSION" ]]; then
  echo -e "${RED}Error: output/ altında sürüm bulunamadı.${NC}" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo -e "${RED}HATA: Kaynak dizin bulunamadı: $SOURCE_DIR${NC}" >&2
  echo -e "${YELLOW}Önce çalıştır: build/benchmark/mot17_compare_trackers_benchmark${NC}" >&2
  exit 1
fi

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  TrackEval Tracker Compare Runner${NC}"
echo -e "${BLUE}  Version: ${VERSION}${NC}"
echo -e "${BLUE}=====================================${NC}\n"

if [[ ! -d "$TRACKEVAL_DIR" ]]; then
  echo -e "${RED}HATA: TrackEval bulunamadı: $TRACKEVAL_DIR${NC}" >&2
  exit 1
fi

EXPECTED_GT_TARGET="$PROJECT_DIR/MOT17-Data/MOT17/train"
if [[ ! -e "$GT_LINK/MOT17-02-FRCNN/seqinfo.ini" ]]; then
  echo -e "${YELLOW}Ground truth linki oluşturuluyor...${NC}"
  mkdir -p "$(dirname "$GT_LINK")"
  rm -f "$GT_LINK"
  ln -sf "$EXPECTED_GT_TARGET" "$GT_LINK"
fi

mkdir -p "$(dirname "$SEQMAPS_FILE")"
echo "name" > "$SEQMAPS_FILE"

TRACKER_NAMES=()
shopt -s nullglob
for tracker_source_dir in "$SOURCE_DIR"/*; do
  [[ -d "$tracker_source_dir" ]] || continue

  tracker_short_name="$(basename "$tracker_source_dir")"
  tracker_name="${VERSION}-${MODE}-${tracker_short_name}"
  tracker_parent="$TRACKEVAL_DIR/data/trackers/mot_challenge/MOT17-train/$tracker_name"
  tracker_data_dir="$tracker_parent/data"

  if [[ "$CLEAN" == true && -d "$tracker_parent" ]]; then
    rm -rf "$tracker_parent"
  fi

  mkdir -p "$tracker_data_dir"
  copied=0
  for file in "$tracker_source_dir"/MOT17-*.txt; do
    [[ -f "$file" ]] || continue
    cp "$file" "$tracker_data_dir/$(basename "$file")"
    copied=$((copied + 1))
  done

  if [[ "$copied" -gt 0 ]]; then
    TRACKER_NAMES+=("$tracker_name")
    echo -e "${GREEN}✓ ${tracker_name}: ${copied} dosya${NC}"
  fi
done
shopt -u nullglob

if [[ "${#TRACKER_NAMES[@]}" -eq 0 ]]; then
  echo -e "${RED}HATA: Hiç tracker sonucu bulunamadı.${NC}" >&2
  exit 1
fi

first_tracker_data="$TRACKEVAL_DIR/data/trackers/mot_challenge/MOT17-train/${TRACKER_NAMES[0]}/data"
for file in "$first_tracker_data"/MOT17-*.txt; do
  [[ -f "$file" ]] || continue
  echo "$(basename "$file" .txt)" >> "$SEQMAPS_FILE"
done

cd "$TRACKEVAL_DIR"
if ! python3 -c "import numpy, scipy" 2>/dev/null; then
  echo -e "${YELLOW}Python bağımlılıkları yükleniyor...${NC}"
  pip3 install numpy scipy --quiet
fi

trackers_label="$(IFS=,; echo "${TRACKER_NAMES[*]}")"

echo -e "\n${BLUE}TrackEval başlatılıyor: ${trackers_label}${NC}\n"
python3 scripts/run_mot_challenge.py \
  --BENCHMARK MOT17 \
  --SPLIT_TO_EVAL train \
  --TRACKERS_TO_EVAL "${TRACKER_NAMES[@]}" \
  --METRICS HOTA CLEAR Identity \
  --USE_PARALLEL False \
  --NUM_PARALLEL_CORES 1 \
  --PLOT_CURVES False

echo -e "\n${GREEN}Değerlendirme tamamlandı.${NC}"
echo -e "${BLUE}Sonuç klasörleri:${NC}"
mkdir -p "$REPORTS_DIR"
for tracker_name in "${TRACKER_NAMES[@]}"; do
  tracker_report_dir="$TRACKEVAL_DIR/data/trackers/mot_challenge/MOT17-train/$tracker_name"
  echo "  $tracker_report_dir"
  report_dest="$REPORTS_DIR/$tracker_name"
  rm -rf "$report_dest"
  cp -a "$tracker_report_dir" "$report_dest"
done
echo -e "${GREEN}Raporlar output/reports altına kopyalandı.${NC}"
