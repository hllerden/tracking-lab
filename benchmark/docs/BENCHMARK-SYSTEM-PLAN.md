# MOT17 Benchmark System - Detaylı Mimari ve Plan

**Oluşturulma Tarihi:** 2025-01-20
**Versiyon:** 1.0.0
**Durum:** Implementation Planı

---

## 🎯 Amaç

MOT17 veri seti üzerinde tracking algoritmalarını sistematik olarak test etmek ve versiyon bazlı performans karşılaştırması yapmak için headless (görsel arayüz olmayan) bir benchmark sistemi oluşturmak.

### Temel Hedefler

1. **Dual-Mode Support**: Hem det.txt (tracker-only) hem de YOLO (end-to-end) modları desteklenmeli
2. **Version Management**: Her geliştirme için ayrı version klasörleri (`v1.0.0`, `v1.0.1`, vb.)
3. **Comprehensive Testing**: Tüm IoU tipleri ve detector kombinasyonlarını otomatik test etme
4. **TrackEval Integration**: Benchmark sonuçlarının otomatik değerlendirilmesi
5. **Version Comparison**: Farklı versiyonların performans karşılaştırması

---

## 📊 Test Matrisi

### Mode 1: Tracker-Only (det.txt)

**Amaç**: Sadece tracking algoritmasını test etmek (detection sabit)

```
Test Kombinasyonları:
- 7 sequences × 3 detectors × 6 IoU types = 126 test

Sequences:
  - MOT17-02, MOT17-04, MOT17-05, MOT17-09
  - MOT17-10, MOT17-11, MOT17-13

Detectors:
  - DPM    (Deformable Part Model)
  - FRCNN  (Faster R-CNN)
  - SDP    (SDP detector)

IoU Types:
  - IOU    (Standard Intersection over Union)
  - GIOU   (Generalized IoU - better for low overlap)
  - DIOU   (Distance IoU - penalizes center distance)
  - CIOU   (Complete IoU - includes aspect ratio)
  - SIOU   (Soft IoU - smoothed exponential)
  - AIOU   (Alpha IoU - tunable sensitivity)

Tahmini Süre: 10-15 dakika (tüm testler)
Ortalama FPS: ~300 FPS (GPU ile)
```

**Örnek Test Çıktısı:**
```
output/v1.0.0/tracker-only/
├── MOT17-02-DPM-IOU.txt
├── MOT17-02-DPM-GIOU.txt
├── MOT17-02-DPM-DIOU.txt
├── MOT17-02-DPM-CIOU.txt
├── MOT17-02-DPM-SIOU.txt
├── MOT17-02-DPM-AIOU.txt
├── MOT17-02-FRCNN-IOU.txt
├── ... (120 daha)
└── benchmark_summary.json
```

---

### Mode 2: End-to-End (YOLO)

**Amaç**: Tüm sistemi test etmek (detection + tracking)

```
Test Kombinasyonları:
- 7 sequences × 1 detector × 6 IoU types = 42 test

Detector:
  - YOLO (yolov9s.onnx)

IoU Types: (aynı 6 tip)

Tahmini Süre: 60-90 dakika (GPU ile)
Ortalama FPS: ~40-50 FPS (GPU ile)
```

**Örnek Test Çıktısı:**
```
output/v1.0.0/end-to-end/
├── MOT17-02-YOLO-IOU.txt
├── MOT17-02-YOLO-GIOU.txt
├── ... (40 daha)
└── benchmark_summary.json
```

---

## 📁 Dosya ve Klasör Yapısı

### Proje Dizin Yapısı

```
opencv-yolo/
├── CMakeLists.txt                      (güncellenecek)
├── benchmark_config.h                  (YENİ)
├── mot17_benchmark.cpp                 (YENİ)
├── kalman_iou_tracker.h/cpp           (mevcut)
├── mot_sequence_reader.h/cpp          (mevcut)
├── mot_detection_reader.h/cpp         (mevcut)
├── inference.h/cpp                     (mevcut)
├── impression.h/cpp                    (mevcut)
├── TrackerTelemtryLogger/
│   └── MotChallengeLogger.h/cpp       (mevcut)
├── output/
│   ├── v1.0.0/
│   │   ├── tracker-only/
│   │   │   ├── MOT17-02-DPM-IOU.txt
│   │   │   ├── MOT17-02-DPM-GIOU.txt
│   │   │   ├── ... (126 dosya)
│   │   │   └── benchmark_summary.json
│   │   ├── end-to-end/
│   │   │   ├── MOT17-02-YOLO-IOU.txt
│   │   │   ├── ... (42 dosya)
│   │   │   └── benchmark_summary.json
│   │   └── combined_report.json
│   ├── v1.0.1/
│   │   └── ... (aynı yapı)
│   └── run_trackeval.sh                (güncellenecek)
├── scripts/
│   ├── compare_versions.sh             (YENİ)
│   └── analyze_results.py              (gelecek - opsiyonel)
├── docs/
│   ├── BENCHMARK-SYSTEM-PLAN.md        (bu dosya)
│   ├── BENCHMARK-TODO.md               (yapılacaklar)
│   └── BENCHMARK-CODE-TEMPLATES.md     (kod şablonları)
└── models/
    └── yolov9s.onnx
```

---

## 🔧 Teknik Mimari

### 1. Version Management

**CMakeLists.txt'de Version Tanımlama:**

```cmake
# Project version
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

# Version compile-time define
add_compile_definitions(PROJECT_VERSION="${PROJECT_VERSION}")
```

**Version Değiştirme:**
```bash
# 1. CMakeLists.txt'de PATCH'i artır
set(PROJECT_VERSION_PATCH 1)

# 2. Rebuild
cd build && cmake .. && make

# 3. Yeni benchmark
./mot17_benchmark
# → output/v1.0.1/ oluşur
```

---

### 2. Benchmark Configuration

**benchmark_config.h Yapısı:**

```cpp
enum class BenchmarkMode {
    TRACKER_ONLY,   // det.txt kullan
    END_TO_END,     // YOLO kullan
    BOTH            // Her ikisini de çalıştır
};

struct BenchmarkConfig {
    std::string projectBasePath;
    std::string outputVersion = PROJECT_VERSION;
    BenchmarkMode mode = BenchmarkMode::BOTH;

    // Test edilecek sequence'ler
    std::vector<std::string> sequences = {
        "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13"
    };

    // Detector'ler (tracker-only için)
    std::vector<std::string> detectors = {"DPM", "FRCNN", "SDP"};

    // Test edilecek IoU tipleri
    std::vector<KalmanIoUConfig::IoUType> iouTypes = {
        IoUType::IOU, IoUType::GIOU, IoUType::DIOU,
        IoUType::CIOU, IoUType::SIOU, IoUType::AIOU
    };

    // YOLO ayarları
    std::string yoloModelPath = "/models/yolov9s.onnx";
    bool runOnGPU = true;

    // Tracker ayarları
    float iouThreshold = 0.2f;
    int maxLostFrames = 20;
    bool usePredictionInLost = true;
    bool removeOutOfBounds = true;
};
```

---

### 3. Benchmark Runner

**mot17_benchmark.cpp Ana Sınıf:**

```cpp
class BenchmarkRunner {
public:
    BenchmarkRunner(const BenchmarkConfig& config);

    // Ana çalıştırma fonksiyonu
    void runAllTests();

private:
    // Mode'lara göre test çalıştırma
    void runTrackerOnlyMode();
    void runEndToEndMode();

    // Tek test fonksiyonları
    void runSingleTest_TrackerOnly(
        const std::string& sequence,
        const std::string& detector,
        KalmanIoUConfig::IoUType iouType
    );

    void runSingleTest_EndToEnd(
        const std::string& sequence,
        KalmanIoUConfig::IoUType iouType
    );

    // Yardımcı fonksiyonlar
    void createOutputDirectories();
    void printProgress(int current, int total, const std::string& testName);
    void saveSummaryJSON(const std::string& mode);
    void saveCombinedReport();

    // İç veri yapıları
    BenchmarkConfig config_;
    std::vector<TestResult> trackerOnlyResults_;
    std::vector<TestResult> endToEndResults_;
};

struct TestResult {
    std::string sequence;
    std::string detector;
    std::string iouType;
    std::string outputFile;
    int totalFrames;
    double processingTimeMs;
    double fps;
    int totalDetections;
    int totalTracks;
    bool success;
    std::string errorMessage;
};
```

---

### 4. Test Execution Flow

**Tracker-Only Mode:**

```
FOR each sequence IN [MOT17-02, ..., MOT17-13]:
    FOR each detector IN [DPM, FRCNN, SDP]:
        FOR each iouType IN [IOU, GIOU, DIOU, CIOU, SIOU, AIOU]:

            1. Initialize MotSequenceReader(sequence)
            2. Initialize MotDetectionReader(sequence/det/det.txt)
            3. Initialize KalmanIoUTracker
            4. Configure tracker with iouType
            5. Initialize MotChallengeLogger(output_path)

            6. FOR each frame:
                - Read frame from MotSequenceReader
                - Get detections from MotDetectionReader
                - Update tracker
                - Log results to MotChallengeLogger

            7. Save results
            8. Collect statistics
```

**End-to-End Mode:**

```
FOR each sequence IN [MOT17-02, ..., MOT17-13]:
    FOR each iouType IN [IOU, GIOU, DIOU, CIOU, SIOU, AIOU]:

        1. Initialize MotSequenceReader(sequence)
        2. Initialize Impression (YOLO + Tracker)
        3. Configure tracker with iouType
        4. Initialize MotChallengeLogger(output_path)

        5. FOR each frame:
            - Read frame from MotSequenceReader
            - Run YOLO inference (Impression)
            - Update tracker
            - Log results

        6. Save results
        7. Collect statistics
```

---

### 5. Output Format

**benchmark_summary.json:**

```json
{
  "version": "1.0.0",
  "mode": "tracker-only",
  "timestamp": "2025-01-20T10:30:00Z",
  "git_commit": "f0c0056",
  "total_tests": 126,
  "successful_tests": 126,
  "failed_tests": 0,
  "total_duration_seconds": 623.4,
  "avg_fps": 305.2,
  "config": {
    "iou_threshold": 0.2,
    "max_lost_frames": 20,
    "use_prediction_in_lost": true,
    "remove_out_of_bounds": true
  },
  "results": [
    {
      "test_id": 1,
      "sequence": "MOT17-02",
      "detector": "DPM",
      "iou_type": "IOU",
      "output_file": "MOT17-02-DPM-IOU.txt",
      "total_frames": 600,
      "processing_time_ms": 3204,
      "fps": 187.3,
      "total_detections": 12450,
      "total_tracks": 89,
      "success": true,
      "error_message": ""
    },
    ...
  ]
}
```

**combined_report.json:**

```json
{
  "version": "1.0.0",
  "timestamp": "2025-01-20T10:30:00Z",
  "tracker_only": {
    "total_tests": 126,
    "avg_fps": 305.2,
    "total_duration_seconds": 623.4
  },
  "end_to_end": {
    "total_tests": 42,
    "avg_fps": 41.2,
    "total_duration_seconds": 1834.2
  },
  "summary": {
    "total_tests": 168,
    "total_duration_seconds": 2457.6,
    "total_duration_minutes": 40.96
  }
}
```

---

## 🚀 Kullanım Senaryoları

### Senaryo 1: Full Benchmark (Default)

```bash
cd build
cmake .. && make
./mot17_benchmark

# Çıktı:
# output/v1.0.0/tracker-only/   (126 test, ~10 dakika)
# output/v1.0.0/end-to-end/     (42 test, ~60 dakika)
```

---

### Senaryo 2: Sadece Tracker-Only (Hızlı)

```bash
./mot17_benchmark --mode tracker-only

# Çıktı:
# output/v1.0.0/tracker-only/   (126 test, ~10 dakika)
```

---

### Senaryo 3: Tek Sequence Test

```bash
./mot17_benchmark \
  --sequences MOT17-02 \
  --detectors DPM \
  --iou-types CIOU

# Çıktı:
# output/v1.0.0/tracker-only/MOT17-02-DPM-CIOU.txt (1 test)
```

---

### Senaryo 4: Version Karşılaştırma

```bash
# 1. İlk baseline oluştur
./mot17_benchmark
# → output/v1.0.0/

# 2. Geliştirme yap ve version'ı artır
# (CMakeLists.txt: PATCH = 1)
cd build && cmake .. && make
./mot17_benchmark
# → output/v1.0.1/

# 3. Karşılaştır
cd output
../scripts/compare_versions.sh v1.0.0 v1.0.1

# Çıktı:
# ┌──────────────┬──────────┬──────────┬─────────┐
# │ Metric       │ v1.0.0   │ v1.0.1   │ Delta   │
# ├──────────────┼──────────┼──────────┼─────────┤
# │ HOTA         │ 26.31%   │ 28.45%   │ +2.14%  │
# │ MOTA         │ 0.91%    │ 3.21%    │ +2.30%  │
# │ IDF1         │ 34.12%   │ 36.89%   │ +2.77%  │
# └──────────────┴──────────┴──────────┴─────────┘
```

---

### Senaryo 5: TrackEval Değerlendirmesi

```bash
# Tracker-only sonuçlarını değerlendir
cd output
./run_trackeval.sh v1.0.0 tracker-only

# End-to-end sonuçlarını değerlendir
./run_trackeval.sh v1.0.0 end-to-end

# Sonuçlar:
# thirdParty/TrackEval/data/trackers/.../pedestrian_summary.txt
```

---

## 📈 Performans Optimizasyonu

### Hız İyileştirmeleri

1. **Tracker-Only Mode**: ~10x daha hızlı (YOLO olmadan)
2. **GPU Kullanımı**: CUDA ile ~5x hızlanma
3. **Paralel İşleme**: (gelecek) Farklı sequence'leri paralel çalıştırma
4. **Caching**: (gelecek) Detection cache sistemi

### Bellek Yönetimi

- Trajectory history sınırlı tutulmalı (maxTrajectoryLength = 50)
- Frame-by-frame işleme (tüm video memory'de değil)
- RAII pattern ile otomatik cleanup

---

## 🐛 Hata Yönetimi

### Exception Handling

```cpp
try {
    runSingleTest_TrackerOnly(sequence, detector, iouType);
    result.success = true;
} catch (const std::exception& e) {
    result.success = false;
    result.errorMessage = e.what();
    std::cerr << "ERROR: " << e.what() << std::endl;
    // Continue with next test (don't crash entire benchmark)
}
```

### Hata Senaryoları

1. **Dosya bulunamadı**: Sequence veya det.txt eksik
2. **Model yükleme hatası**: YOLO modeli eksik/bozuk
3. **Out of memory**: Frame çok büyük veya çok fazla tracker
4. **OpenCV hataları**: Video decode problemi

**Çözüm**: Her test try-catch ile korunmalı, hata loglanmalı, benchmark devam etmeli.

---

## 🔍 Debugging ve Monitoring

### Progress Tracking

```
=== MOT17 Benchmark v1.0.0 ===
Running Tracker-Only Mode...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[  1/126] MOT17-02-DPM-IOU      ... Done (2.3s, 261 FPS)
[  2/126] MOT17-02-DPM-GIOU     ... Done (2.4s, 250 FPS)
[  3/126] MOT17-02-DPM-DIOU     ... Done (2.3s, 261 FPS)
...
[ 45/126] MOT17-05-FRCNN-CIOU   ... Done (3.1s, 194 FPS)
...
[126/126] MOT17-13-SDP-AIOU     ... Done (3.0s, 200 FPS)

Summary:
  ✓ Total: 126 tests
  ✓ Success: 126 (100%)
  ✗ Failed: 0 (0%)
  ⏱ Duration: 623.4s (10.4 minutes)
  ⚡ Avg FPS: 305.2
```

### Logging Levels

- **INFO**: Test başlangıcı, bitiş, özet
- **WARNING**: Düşük FPS, fazla IDSW
- **ERROR**: Test başarısız, exception

---

## 📚 Referanslar

### İlgili Dosyalar

- [BENCHMARK-TODO.md](BENCHMARK-TODO.md): Checkbox'lu yapılacaklar listesi
- [BENCHMARK-CODE-TEMPLATES.md](BENCHMARK-CODE-TEMPLATES.md): Hazır kod şablonları
- [MOT17_USAGE.md](../MOT17_USAGE.md): MOT17 kullanım kılavuzu
- [CLAUDE.md](../CLAUDE.md): Proje mimari dokümantasyonu

### External Links

- [MOTChallenge](https://motchallenge.net/)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [HOTA Metric Paper](https://link.springer.com/article/10.1007/s11263-020-01375-2)

---

## 🎯 Version History Plan

```
v1.0.0: Initial baseline
  - Current KalmanIoU implementation
  - Default parameters (iouThreshold=0.2, maxLostFrames=20)
  - CIOU as default IoU type

v1.0.1: Kalman filter tuning
  - Optimize process/measurement noise
  - Test different prediction models

v1.0.2: IoU threshold optimization
  - Dynamic threshold based on detection confidence
  - Adaptive IoU per sequence

v1.0.3: Hungarian algorithm optimization
  - Faster assignment algorithm
  - Better cost matrix calculation

v1.0.4: Lost track handling
  - Improved re-identification
  - Better occlusion handling
```

---

## ✅ Success Criteria

Benchmark sistemi başarılı sayılır eğer:

1. ✅ **Tüm testler çalışıyor**: 168 test hatasız tamamlanıyor
2. ✅ **Headless**: İmshow yok, tamamen otomatik
3. ✅ **Versiyonlu**: Her geliştirme için ayrı klasör
4. ✅ **TrackEval entegre**: Otomatik değerlendirme çalışıyor
5. ✅ **Karşılaştırılabilir**: Version comparison script çalışıyor
6. ✅ **Dokümante**: Tüm kullanım senaryoları açık
7. ✅ **Hata toleranslı**: Tek test başarısız olsa da devam ediyor

---

## 📝 Notlar

- Bu plan implementation sırasında güncellenebilir
- Tüm değişiklikler git commit'lerde loglanmalı
- Her yeni version için CHANGELOG.md güncellenmeli
- Performance regression testleri düzenli yapılmalı

---

**Son Güncelleme:** 2025-01-20
**Sorumlu:** Halil Erden
**Durum:** ✅ Plan Tamamlandı, Implementation Başlayacak
