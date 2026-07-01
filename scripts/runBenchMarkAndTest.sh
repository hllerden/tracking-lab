#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root based on this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRACKERS_DIR="${REPO_ROOT}/thirdParty/TrackEval/data/trackers/mot_challenge/MOT17-train"
REPORTS_DIR="${REPO_ROOT}/output/reports"
COMPARISON_DIR="${REPO_ROOT}/output/comparison"
OUTPUT_DIR="${REPO_ROOT}/output"
RUN_TRACKEVAL_SCRIPT="${REPO_ROOT}/output/run_trackeval.sh"

echo "[runBenchMarkAndTest] Lütfen benchmark işlemini manuel olarak başlattığınızdan emin olun."

if [[ ! -x "${RUN_TRACKEVAL_SCRIPT}" ]]; then
  echo "[runBenchMarkAndTest] Hata: ${RUN_TRACKEVAL_SCRIPT} çalıştırılabilir değil veya bulunamadı." >&2
  exit 1
fi

shopt -s nullglob
output_version_paths=("${OUTPUT_DIR}"/v*)
shopt -u nullglob

output_versions=()
for version_path in "${output_version_paths[@]}"; do
  [[ -d "${version_path}" ]] || continue
  version_name="$(basename "${version_path}")"
  if [[ "${version_name}" =~ ^v[0-9]+(\.[0-9]+)*$ ]]; then
    output_versions+=("${version_name}")
  fi
done

if [[ ${#output_versions[@]} -eq 0 ]]; then
  echo "[runBenchMarkAndTest] Hata: ${OUTPUT_DIR} içinde geçerli bir sürüm klasörü bulunamadı." >&2
  exit 1
fi

mapfile -t sorted_output_versions < <(printf '%s\n' "${output_versions[@]}" | sort -V)
latest_output_version="${sorted_output_versions[${#sorted_output_versions[@]}-1]}"

echo "[runBenchMarkAndTest] TrackEval ${latest_output_version} sürümü ile çalıştırılıyor..."
"${RUN_TRACKEVAL_SCRIPT}" --clean "${latest_output_version}" --minimal
echo "[runBenchMarkAndTest] TrackEval tamamlandı."

if [[ ! -d "${TRACKERS_DIR}" ]]; then
  echo "[runBenchMarkAndTest] Hata: Beklenen dizin bulunamadı: ${TRACKERS_DIR}" >&2
  exit 1
fi

mkdir -p "${REPORTS_DIR}"

echo "[runBenchMarkAndTest] TrackEval çıktıları rapor klasörüne kopyalanıyor..."
shopt -s nullglob
version_paths=("${TRACKERS_DIR}"/*)
shopt -u nullglob

if [[ ${#version_paths[@]} -eq 0 ]]; then
  echo "[runBenchMarkAndTest] Kopyalanacak sürüm dizini bulunamadı." >&2
  exit 1
fi

for version_path in "${version_paths[@]}"; do
  if [[ -d "${version_path}" ]]; then
    version_name="$(basename "${version_path}")"
    dest_path="${REPORTS_DIR}/${version_name}"
    rm -rf "${dest_path}"
    cp -a "${version_path}" "${dest_path}"
  fi
done
echo "[runBenchMarkAndTest] Rapor kopyalama tamamlandı."

shopt -s nullglob
minimal_version_paths=("${TRACKERS_DIR}"/v*-minimal)
shopt -u nullglob

minimal_versions=()
for version_path in "${minimal_version_paths[@]}"; do
  [[ -d "${version_path}" ]] || continue
  version_name="$(basename "${version_path}")"
  if [[ "${version_name}" =~ ^v[0-9]+(\.[0-9]+)*-minimal$ ]]; then
    minimal_versions+=("${version_name}")
  fi
done

if [[ ${#minimal_versions[@]} -lt 2 ]]; then
  echo "[runBenchMarkAndTest] Karşılaştırma için en az iki sürüme ihtiyaç var." >&2
  exit 1
fi

mapfile -t sorted_minimal_versions < <(printf '%s\n' "${minimal_versions[@]}" | sort -V)
prev_index=$((${#sorted_minimal_versions[@]} - 2))
latest_index=$((${#sorted_minimal_versions[@]} - 1))
previous_minimal_version="${sorted_minimal_versions[$prev_index]}"
latest_minimal_version="${sorted_minimal_versions[$latest_index]}"

mkdir -p "${COMPARISON_DIR}"

echo "[runBenchMarkAndTest] Sürümler karşılaştırılıyor: ${previous_minimal_version} -> ${latest_minimal_version}"
python3 -m scripts.compare_versions_minimal "${previous_minimal_version}" "${latest_minimal_version}" \
  --output "${COMPARISON_DIR}" \
  --html \
  --json \
  --groups primary detection counts
echo "[runBenchMarkAndTest] Sürüm karşılaştırması tamamlandı."

echo "[runBenchMarkAndTest] Tüm adımlar başarıyla tamamlandı."
