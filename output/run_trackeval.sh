#!/bin/bash

###############################################################################
# TrackEval Runner Script for MOT17 Evaluation (VERSION SUPPORT)
#
# Bu script tracker sonuçlarını TrackEval ile değerlendirir.
# Kullanım:
#   ./run_trackeval.sh                              # Latest version, tracker-only, CIOU
#   ./run_trackeval.sh -clean                       # Temizledikten sonra latest version
#   ./run_trackeval.sh v1.0.0                       # Specific version, tracker-only, CIOU
#   ./run_trackeval.sh v1.0.0 tracker-only          # Explicit mode, CIOU
#   ./run_trackeval.sh v1.0.0 tracker-only CIOU     # Specific IoU type
#   ./run_trackeval.sh v1.0.0 end-to-end CIOU       # End-to-end mode, CIOU
#   ./run_trackeval.sh v1.0.0 --minimal             # Minimal benchmark mode
#   ./run_trackeval.sh --minimal                    # Latest version, minimal mode
#   ./run_trackeval.sh -clean v1.0.0 tracker-only   # Temizlik + explicit mode
###############################################################################

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parametreler
CLEAN=false
MINIMAL_MODE=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -clean|--clean)
            CLEAN=true
            shift
            ;;
        --minimal)
            MINIMAL_MODE=true
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo -e "${RED}Error: Bilinmeyen seçenek '$1'${NC}"
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

VERSION=${1:-$(ls -td v*/ 2>/dev/null | head -1 | xargs basename 2>/dev/null)}

# Minimal mode overrides
if [ "$MINIMAL_MODE" = true ]; then
    MODE="minimal"
    IOU_TYPE="NONE"  # No IoU suffix in minimal mode
else
    MODE=${2:-"tracker-only"}
    IOU_TYPE=${3:-"CIOU"}
fi

# Parametre kontrolü
if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: No version found in output/ directory${NC}"
    echo -e "${YELLOW}Run benchmark first: cd build/benchmark && ./mot17_benchmark${NC}"
    exit 1
fi

if [ "$MODE" != "tracker-only" ] && [ "$MODE" != "end-to-end" ] && [ "$MODE" != "minimal" ]; then
    echo -e "${RED}Error: Invalid mode '$MODE'${NC}"
    echo -e "${YELLOW}Valid modes: tracker-only, end-to-end, minimal${NC}"
    exit 1
fi

# IoU type validation (skip for minimal mode)
if [ "$MODE" != "minimal" ]; then
    VALID_IOU_TYPES="IOU GIOU DIOU CIOU SIOU AIOU"
    if [[ ! " $VALID_IOU_TYPES " =~ " $IOU_TYPE " ]]; then
        echo -e "${RED}Error: Invalid IoU type '$IOU_TYPE'${NC}"
        echo -e "${YELLOW}Valid IoU types: $VALID_IOU_TYPES${NC}"
        exit 1
    fi
fi

# Dizin yapısı
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$SCRIPT_DIR"
TRACKEVAL_DIR="$PROJECT_DIR/thirdParty/TrackEval"
GT_LINK="$TRACKEVAL_DIR/data/gt/mot_challenge/MOT17-train"

# Versiyonlu tracker adı (minimal mode için özel format)
if [ "$MODE" = "minimal" ]; then
    TRACKER_NAME="${VERSION}-minimal"
else
    TRACKER_NAME="${VERSION}-${MODE}-${IOU_TYPE}"
fi
TRACKER_DIR="$TRACKEVAL_DIR/data/trackers/mot_challenge/MOT17-train/$TRACKER_NAME/data"
SEQMAPS_FILE="$TRACKEVAL_DIR/data/gt/mot_challenge/seqmaps/MOT17-train.txt"

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  TrackEval Runner for MOT17${NC}"
echo -e "${BLUE}  Version: ${VERSION}${NC}"
echo -e "${BLUE}  Mode: ${MODE}${NC}"
if [ "$MODE" != "minimal" ]; then
    echo -e "${BLUE}  IoU Type: ${IOU_TYPE}${NC}"
else
    echo -e "${BLUE}  IoU Type: Hardcoded in code${NC}"
fi
echo -e "${BLUE}=====================================${NC}\n"

# Dizin kontrolü
echo -e "${YELLOW}[1/6] Dizinleri kontrol ediliyor...${NC}"
if [ ! -d "$TRACKEVAL_DIR" ]; then
    echo -e "${RED}HATA: TrackEval bulunamadı: $TRACKEVAL_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ TrackEval bulundu${NC}"

# Ground truth kontrolü
echo -e "\n${YELLOW}[2/6] Ground truth kontrolü...${NC}"
if [ ! -L "$GT_LINK" ] && [ ! -d "$GT_LINK" ]; then
    echo -e "${YELLOW}Ground truth linki oluşturuluyor...${NC}"
    mkdir -p "$(dirname "$GT_LINK")"
    ln -sf "$PROJECT_DIR/MOT17-Data/MOT17/train" "$GT_LINK"
    echo -e "${GREEN}✓ Ground truth linki oluşturuldu${NC}"
else
    echo -e "${GREEN}✓ Ground truth mevcut${NC}"
fi

# Tracker dizini oluştur
echo -e "\n${YELLOW}[3/6] Tracker dizini hazırlanıyor ($TRACKER_NAME)...${NC}"
TRACKER_PARENT_DIR="$(dirname "$TRACKER_DIR")"
if [ "$CLEAN" = true ] && [ -d "$TRACKER_PARENT_DIR" ]; then
    echo -e "${YELLOW}Temizleme aktif: mevcut tracker dizini siliniyor...${NC}"
    rm -rf "$TRACKER_PARENT_DIR"
fi
mkdir -p "$TRACKER_DIR"
echo -e "${GREEN}✓ Tracker dizini hazır${NC}"

# Sonuç dosyalarını kopyala (versiyonlu, IoU-specific veya minimal)
echo -e "\n${YELLOW}[4/6] Tracker sonuçları kopyalanıyor...${NC}"
echo -e "  Hedef pwd: $(cd "$TRACKER_DIR" && pwd)"
SOURCE_DIR="$OUTPUT_DIR/$VERSION/$MODE"

if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}HATA: Kaynak dizin bulunamadı: $SOURCE_DIR${NC}"
    if [ "$MODE" = "minimal" ]; then
        echo -e "${YELLOW}Minimal benchmark çalıştırıldı mı? cd build/benchmark && ./mot17_minimal_benchmark${NC}"
    else
        echo -e "${YELLOW}Benchmark çalıştırıldı mı? cd build/benchmark && ./mot17_benchmark${NC}"
    fi
    exit 1
fi

COPIED_COUNT=0

if [ "$MODE" = "minimal" ]; then
    # Minimal mode: Dosyalar zaten doğru formatta (MOT17-02-DPM.txt)
    for file in "$SOURCE_DIR"/MOT17-*.txt; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            cp "$file" "$TRACKER_DIR/$filename"
            echo -e "  → $filename"
            COPIED_COUNT=$((COPIED_COUNT + 1))
        fi
    done
else
    # Normal mode: Sadece belirtilen IoU tipindeki dosyaları kopyala ve adını düzelt
    # MOT17-02-DPM-CIOU.txt → MOT17-02-DPM.txt (TrackEval için)
    for file in "$SOURCE_DIR"/*-${IOU_TYPE}.txt; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            # IoU suffix'ini çıkar
            # MOT17-02-DPM-CIOU.txt → MOT17-02-DPM.txt
            new_filename=$(echo "$filename" | sed "s/-${IOU_TYPE}\.txt$/.txt/")
            cp "$file" "$TRACKER_DIR/$new_filename"
            echo -e "  → $filename → $new_filename"
            COPIED_COUNT=$((COPIED_COUNT + 1))
        fi
    done
fi

if [ $COPIED_COUNT -eq 0 ]; then
    echo -e "${RED}HATA: Hiç tracker sonucu bulunamadı!${NC}"
    echo -e "${YELLOW}output/ dizininde MOT17-*.txt dosyaları olmalı${NC}"
    exit 1
fi
echo -e "${GREEN}✓ $COPIED_COUNT dosya kopyalandı${NC}"

# Seqmaps dosyası oluştur
echo -e "\n${YELLOW}[5/6] Seqmaps dosyası oluşturuluyor...${NC}"
mkdir -p "$(dirname "$SEQMAPS_FILE")"
echo "name" > "$SEQMAPS_FILE"

# Kopyalanan dosyalardan sequence isimlerini çıkar
# Artık dosya isimleri düzgün: MOT17-{SEQ}-{DET}.txt
for file in "$TRACKER_DIR"/MOT17-*.txt; do
    if [ -f "$file" ]; then
        seq_name=$(basename "$file" .txt)
        echo "$seq_name" >> "$SEQMAPS_FILE"
    fi
done

SEQ_COUNT=$(( $(wc -l < "$SEQMAPS_FILE") - 1 ))
echo -e "${GREEN}✓ $SEQ_COUNT sequence için seqmaps hazırlandı${NC}"

# Python bağımlılıkları kontrolü
echo -e "\n${YELLOW}[6/6] Python bağımlılıkları kontrol ediliyor...${NC}"
cd "$TRACKEVAL_DIR"
if ! python3 -c "import numpy, scipy" 2>/dev/null; then
    echo -e "${YELLOW}Bağımlılıklar yükleniyor...${NC}"
    pip3 install numpy scipy --quiet
fi
echo -e "${GREEN}✓ Python bağımlılıkları hazır${NC}"

# TrackEval çalıştır
echo -e "\n${BLUE}=====================================${NC}"
echo -e "${BLUE}  TrackEval Başlatılıyor...${NC}"
echo -e "${BLUE}=====================================${NC}\n"

python3 scripts/run_mot_challenge.py \
    --BENCHMARK MOT17 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL "$TRACKER_NAME" \
    --METRICS HOTA CLEAR Identity \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1 \
    --PLOT_CURVES False

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}=====================================${NC}"
    echo -e "${GREEN}  Değerlendirme Tamamlandı!${NC}"
    echo -e "${GREEN}  Version: ${VERSION}, Mode: ${MODE}${NC}"
    echo -e "${GREEN}=====================================${NC}"
    echo -e "\n${BLUE}Sonuç dosyaları:${NC}"
    echo -e "  ${YELLOW}Summary:${NC} $TRACKEVAL_DIR/data/trackers/mot_challenge/MOT17-train/$TRACKER_NAME/pedestrian_summary.txt"
    echo -e "  ${YELLOW}Detailed CSV:${NC} $TRACKEVAL_DIR/data/trackers/mot_challenge/MOT17-train/$TRACKER_NAME/pedestrian_detailed.csv"
else
    echo -e "\n${RED}=====================================${NC}"
    echo -e "${RED}  Değerlendirme Başarısız!${NC}"
    echo -e "${RED}=====================================${NC}"
    exit $EVAL_EXIT_CODE
fi
if [ $EVAL_EXIT_CODE -eq 0 ]; then
# ==============================================================================
# TRACKEVAL RAPOR METRİKLERİ AÇIKLAMALARI
# ==============================================================================
echo -e "\n${CYAN}===================================================================${NC}"
echo -e "${CYAN}  TRACKEVAL RAPOR METRİKLERİ AÇIKLAMALARI${NC}"
echo -e "${CYAN}===================================================================${NC}"

echo -e "\n${YELLOW}--- HOTA (Higher Order Tracking Accuracy) Metrikleri ---${NC}"
echo -e "  ${GREEN}HOTA:${NC} Genel takip doğruluğu. Tespit (Detection) ve ilişkilendirme (Association) performansını birleştirir.${RED}[İstenir: Yüksek. Genellikle > 50% iyi kabul edilir.]"
echo -e "  ${GREEN}DetA:${NC} Tespit doğruluğu. Algoritmanın nesne tespit performansını özetler.${RED}[İstenir: Yüksek.]"
echo -e "  ${GREEN}AssA:${NC} İlişkilendirme doğruluğu. Takip sırasında kimliklerin sürekliliğini ölçer.${RED}[İstenir: Yüksek.]"
echo -e "  ${GREEN}DetRe (Recall):${NC} Bulunması gereken nesnelerin ne kadarının tespit edildiği. ${RED}[İstenir: Yüksek (> 80%).]"
echo -e "  ${GREEN}DetPr (Precision):${NC} Yapılan tespitlerin ne kadarının gerçek nesne olduğu. ${RED}[İstenir: Yüksek (> 80%).]"
echo -e "  ${GREEN}AssRe:${NC} Tüm takip edilmesi gereken anlarda doğru kimlik atamalarının oranını gösterir (İlişkilendirme Geri Çağırma).${RED}[İstenir: Yüksek.]"
echo -e "  ${GREEN}AssPr:${NC} Atanan kimlik eşleşmelerinin ne kadarının doğru olduğunu gösterir (İlişkilendirme Kesinliği).${RED}[İstenir: Yüksek.]"
echo -e "  ${GREEN}LocA:${NC} Lokalizasyon doğruluğu. Sınırlayıcı kutuların (bounding box) yer doğruluğunu ölçer."
echo -e "  ${GREEN}OWTA:${NC} Tek yönlü takip doğruluğu. HOTA'nın bir varyantıdır."
echo -e "  ${GREEN}HOTA(0):${NC} HOTA puanının IoU eşiği 0 için hesaplanmış hali (eşiğe bağlı davranışın anlaşılması için)."
echo -e "  ${GREEN}LocA(0):${NC} Lokalizasyon doğruluğunun IoU eşiği 0 için gösterimi. ${RED}[İstenir: Yüksek.]"
echo -e "  ${GREEN}HOTALocA(0):${NC} HOTA içerisinde lokalizasyona yönelik bileşenin IoU=0 eşiği için değeri."

echo -e "\n${YELLOW}--- CLEAR Metrikleri ---${NC}"
echo -e "  ${GREEN}MOTA:${NC} Geleneksel ana metrik. Tespit hataları, yanlış pozitifler ve kimlik değişimleri göz önüne alınarak hesaplanır.${RED}[İstenir: Yüksek (> 50-60% iyi kabul edilir, > 80% mükemmeldir). Negatif olabilir.]"
echo -e "  ${GREEN}MOTP:${NC} Konumlandırma hassasiyeti. Doğru bulunan nesnelerin konum doğruluğunu özetler.${RED}[İstenir: Yüksek (0'a yakın hata iyi).] "
echo -e "  ${GREEN}MODA:${NC} Çoklu Nesne Tespit Doğruluğu. Sadece tespit hatalarını dikkate alır (IDSW hariç)."
echo -e "  ${GREEN}CLR_Re/CLR_Pr:${NC} CLEAR Geri Çağırma/Kesinlik (Tespit performansını ölçer).(Re/Pr için yüksek istenir)."
echo -e "  ${GREEN}CLR_TP/FN/FP:${NC} Doğru Pozitif/Yanlış Negatif/Yanlış Pozitif tespit sayıları.(FN/FP için düşük istenir)."
echo -e "  ${GREEN}IDSW:${NC} Kimlik değişimleri (ID switches). Bir nesnenin takip kimliğinin yanlış değiştiği olay sayısı.${RED}[İstenir: Düşük (Mümkünse 0'a yakın olmalı)."
echo -e "  ${GREEN}MTR (Mostly Tracked):${NC} %80'den fazla süreyle doğru takip edilen ground-truth nesnelerin oranı.${RED}[İstenir: Yüksek.]"
echo -e "  ${GREEN}PTR (Partially Tracked):${NC} %20 ile %80 arası süreyle takip edilen ground-truth nesnelerin oranı."
echo -e "  ${GREEN}MLR (Mostly Lost):${NC} %20'den az süreyle takip edilen ground-truth nesnelerin oranı. ${RED}[İstenir: Düşük.]"
echo -e "  ${GREEN}Frag (Fragmentation):${NC} Bir ground-truth nesnesinin takip kayıtlarının bölünmesi/yeniden başlaması sayısı.${RED}[İstenir: Düşük.]"
echo -e "  ${GREEN}sMOTA:${NC} Standartlaştırılmış MOTA (Genellikle konum hassasiyeti etkisini içerir)."

echo -e "\n${YELLOW}--- Identity (Kimlik) Metrikleri ---${NC}"
echo -e "  ${GREEN}IDF1:${NC} Kimlik doğruluğunun dengeli F1 skoru (ID tabanlı eşlemelerde performans).${RED}[İstenir: Yüksek (> 50% iyi kabul edilir).]"
echo -e "  ${GREEN}IDR (Recall):${NC} Doğru kimlik eşleşmelerinin geri çağırması (recall).${RED}[İstenir: Yüksek.]"
echo -e "  ${GREEN}IDP (Precision):${NC} Atanan kimlik eşleşmelerinin kesinliği (precision).${RED}[İstenir: Yüksek.]"
echo -e "  ${GREEN}IDTP/IDFN/IDFP:${NC} Doğru/yanlış negatif/yanlış pozitif kimlik eşleştirme sayıları.(FN/FP için düşük istenir)"

echo -e "\n${YELLOW}--- Count (Sayım) Metrikleri ---${NC}"
echo -e "  ${GREEN}Dets:${NC} Algoritmanın ürettiği toplam tespit sayısı.${RED}[Dets'in GT_Dets'e yakın olması beklenir.]"
echo -e "  ${GREEN}GT_Dets:${NC} Ground-truth verisindeki toplam nesne (detection) sayısı."
echo -e "  ${GREEN}IDs:${NC} Algoritmanın oluşturduğu benzersiz takip ID sayısı.${RED}[IDs'in GT_IDs'e yakın olması beklenir.]"
echo -e "  ${GREEN}GT_IDs:${NC} Ground-truth verisindeki benzersiz nesne kimlik sayısı."
echo -e "\n"
fi




