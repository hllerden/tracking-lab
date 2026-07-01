# MOT17 Benchmark System - Implementation Checklist

**Başlangıç Tarihi:** 2025-01-20
**Tahmini Süre:** 2-3 saat
**Durum:** 🚧 In Progress

---

## 📋 Genel İlerleme

```
┌─────────────────────────────────────────────────┐
│ Tamamlanan: 1 / 11                              │
│ Progress: ███░░░░░░░░░░░░░░░░░░░░░░░░ 9%       │
└─────────────────────────────────────────────────┘
```

---

## 🎯 Faz 1: Dokümantasyon (Öncelik: ⚡️ Yüksek)

### Doküman Oluşturma
- [x] **BENCHMARK-SYSTEM-PLAN.md** oluştur
  - [x] Mimari ve test matrisi
  - [x] Dosya yapısı
  - [x] Kullanım senaryoları
  - [x] Technical details

- [ ] **BENCHMARK-TODO.md** oluştur (bu dosya)
  - [x] Checkbox listesi yapısı
  - [ ] Tüm maddeleri ekle
  - [ ] Tahmini süreler

- [ ] **BENCHMARK-CODE-TEMPLATES.md** oluştur
  - [ ] CMakeLists.txt template
  - [ ] benchmark_config.h template
  - [ ] mot17_benchmark.cpp template
  - [ ] Script templates

---

## 🔧 Faz 2: Temel Yapı (Öncelik: ⚡️ Yüksek)

### 2.1 CMakeLists.txt Güncellemesi
**Dosya:** `CMakeLists.txt`

- [ ] Version tanımlamaları ekle
  - [ ] `set(PROJECT_VERSION_MAJOR 1)`
  - [ ] `set(PROJECT_VERSION_MINOR 0)`
  - [ ] `set(PROJECT_VERSION_PATCH 0)`
  - [ ] `set(PROJECT_VERSION "${MAJOR}.${MINOR}.${PATCH}")`

- [ ] Project satırını güncelle
  - [ ] `project(yoloExercise VERSION ${PROJECT_VERSION} LANGUAGES CXX)`

- [ ] Compile definitions ekle
  - [ ] `add_compile_definitions(PROJECT_VERSION="${PROJECT_VERSION}")`

- [ ] `mot17_benchmark` executable ekle
  - [ ] Source files listesi (benchmark_config.h, mot17_benchmark.cpp, vb.)
  - [ ] `target_link_libraries(mot17_benchmark ${OpenCV_LIBS} Eigen3::Eigen)`

**Tahmini Süre:** 10 dakika
**Bağımlılık:** Yok
**Test:** `cd build && cmake .. && make mot17_benchmark`

---

### 2.2 benchmark_config.h Oluşturma
**Dosya:** `benchmark/benchmark_config.h` (YENİ)

- [ ] Header guard ekle (`#ifndef BENCHMARK_CONFIG_H`)
- [ ] Include'lar ekle
  - [ ] `<string>`, `<vector>`
  - [ ] `"kalman_iou_tracker.h"`

- [ ] `BenchmarkMode` enum tanımla
  - [ ] `TRACKER_ONLY`
  - [ ] `END_TO_END`
  - [ ] `BOTH`

- [ ] `BenchmarkConfig` struct tanımla
  - [ ] `projectBasePath` (string)
  - [ ] `outputVersion` (string, default: PROJECT_VERSION)
  - [ ] `mode` (BenchmarkMode, default: BOTH)
  - [ ] `sequences` (vector<string>)
  - [ ] `detectors` (vector<string>)
  - [ ] `iouTypes` (vector<IoUType>)
  - [ ] `yoloModelPath` (string)
  - [ ] `runOnGPU` (bool, default: true)
  - [ ] Tracker parametreleri (iouThreshold, maxLostFrames, vb.)

- [ ] `TestResult` struct tanımla
  - [ ] `sequence`, `detector`, `iouType` (string)
  - [ ] `outputFile` (string)
  - [ ] `totalFrames`, `processingTimeMs`, `fps` (numeric)
  - [ ] `totalDetections`, `totalTracks` (int)
  - [ ] `success` (bool), `errorMessage` (string)

- [ ] Helper fonksiyonlar
  - [ ] `iouTypeToString(IoUType)` → string
  - [ ] `stringToIoUType(string)` → IoUType

**Tahmini Süre:** 15 dakika
**Bağımlılık:** Yok
**Test:** `#include "benchmark_config.h"` compile oluyor mu?

---

### 2.3 mot17_benchmark.cpp Oluşturma (İskelet)
**Dosya:** `benchmark/mot17_benchmark.cpp` (YENİ)

- [ ] Include'lar ekle
  - [ ] Standard library (`<iostream>`, `<fstream>`, `<chrono>`, `<filesystem>`)
  - [ ] Project headers (`benchmark_config.h`, `mot_sequence_reader.h`, vb.)

- [ ] `BenchmarkRunner` sınıfı tanımla
  - [ ] Constructor: `BenchmarkRunner(const BenchmarkConfig&)`
  - [ ] Public metodlar:
    - [ ] `void runAllTests()`
  - [ ] Private metodlar:
    - [ ] `void runTrackerOnlyMode()`
    - [ ] `void runEndToEndMode()`
    - [ ] `void runSingleTest_TrackerOnly(...)`
    - [ ] `void runSingleTest_EndToEnd(...)`
    - [ ] `void createOutputDirectories()`
    - [ ] `void printProgress(int current, int total, const std::string& testName)`
    - [ ] `void saveSummaryJSON(const std::string& mode)`
    - [ ] `void saveCombinedReport()`
  - [ ] Private members:
    - [ ] `BenchmarkConfig config_`
    - [ ] `std::vector<TestResult> trackerOnlyResults_`
    - [ ] `std::vector<TestResult> endToEndResults_`

- [ ] `main()` fonksiyonu
  - [ ] Command line parsing (optional)
  - [ ] Config oluştur ve başlat
  - [ ] `BenchmarkRunner runner(config);`
  - [ ] `runner.runAllTests();`
  - [ ] Return 0

**Tahmini Süre:** 20 dakika (sadece iskelet)
**Bağımlılık:** benchmark_config.h
**Test:** Compile oluyor mu? `make mot17_benchmark`

---

## 🎨 Faz 3: Core Implementation (Öncelik: ⚡️ Yüksek)

### 3.1 BenchmarkRunner::runTrackerOnlyMode()
**Dosya:** `mot17_benchmark.cpp`

- [ ] Output dizinini oluştur
  - [ ] `output/v{VERSION}/tracker-only/`

- [ ] Test sayısını hesapla
  - [ ] `totalTests = sequences.size() × detectors.size() × iouTypes.size()`

- [ ] İç içe döngüler
  - [ ] `for (sequence in sequences)`
  - [ ] `  for (detector in detectors)`
  - [ ] `    for (iouType in iouTypes)`
  - [ ] `      runSingleTest_TrackerOnly(sequence, detector, iouType)`

- [ ] Progress tracking
  - [ ] Her test sonrası `printProgress(testIndex, totalTests, testName)`

- [ ] Summary kaydetme
  - [ ] `saveSummaryJSON("tracker-only")`

**Tahmini Süre:** 30 dakika
**Bağımlılık:** `runSingleTest_TrackerOnly()` implementasyonu
**Test:** Çalıştır ve progress çıktısını kontrol et

---

### 3.2 BenchmarkRunner::runSingleTest_TrackerOnly()
**Dosya:** `mot17_benchmark.cpp`

- [ ] Try-catch bloğu ekle
- [ ] Sequence path oluştur
  - [ ] `projectBasePath + "/MOT17-Data/MOT17/train/" + sequence + "-" + detector`

- [ ] Output path oluştur
  - [ ] `output/v{VERSION}/tracker-only/{sequence}-{detector}-{iouType}.txt`

- [ ] Components başlat
  - [ ] `MotSequenceReader reader(sequencePath)`
  - [ ] `MotDetectionReader detReader(sequencePath + "/det/det.txt")`
  - [ ] `KalmanIoUTracker tracker`
  - [ ] Configure tracker (iouType, threshold, vb.)
  - [ ] `MotChallengeLogger logger(outputPath, true)`
  - [ ] `tracker.setTelemetry(logger)`

- [ ] Frame loop
  - [ ] `while (reader.nextFrame(frame))`
  - [ ] `  detections = detReader.getDetectionsForFrame(frameIndex)`
  - [ ] `  tracker.update(detections)`
  - [ ] Timing ölçümü (startTime, endTime, FPS hesapla)

- [ ] TestResult oluştur ve döndür
  - [ ] `result.success = true`
  - [ ] `result.fps = totalFrames / totalTime`
  - [ ] `result.totalDetections = ...`

- [ ] Exception handling
  - [ ] `catch (std::exception& e)` → `result.success = false`, `result.errorMessage = e.what()`

**Tahmini Süre:** 40 dakika
**Bağımlılık:** Mevcut sınıflar (MotSequenceReader, vb.)
**Test:** Tek sequence ile test et

---

### 3.3 BenchmarkRunner::runEndToEndMode()
**Dosya:** `mot17_benchmark.cpp`

- [ ] Output dizinini oluştur
  - [ ] `output/v{VERSION}/end-to-end/`

- [ ] Test sayısını hesapla
  - [ ] `totalTests = sequences.size() × iouTypes.size()`

- [ ] İç içe döngüler
  - [ ] `for (sequence in sequences)`
  - [ ] `  for (iouType in iouTypes)`
  - [ ] `    runSingleTest_EndToEnd(sequence, iouType)`

- [ ] Progress tracking
- [ ] Summary kaydetme

**Tahmini Süre:** 15 dakika (runTrackerOnlyMode'a benzer)
**Bağımlılık:** `runSingleTest_EndToEnd()` implementasyonu

---

### 3.4 BenchmarkRunner::runSingleTest_EndToEnd()
**Dosya:** `mot17_benchmark.cpp`

- [ ] Try-catch bloğu ekle
- [ ] Sequence path oluştur
  - [ ] `projectBasePath + "/MOT17-Data/MOT17/train/" + sequence + "-DPM"` (herhangi biri)

- [ ] Output path oluştur
  - [ ] `output/v{VERSION}/end-to-end/{sequence}-YOLO-{iouType}.txt`

- [ ] Components başlat
  - [ ] `MotSequenceReader reader(sequencePath)`
  - [ ] `Impression impression`
  - [ ] `impression.createInfrance(yoloModelPath, ...)`
  - [ ] Configure impression tracker (iouType)
  - [ ] `MotChallengeLogger logger(outputPath, true)`
  - [ ] `impression.setTelemetry(logger)`

- [ ] Frame loop
  - [ ] `while (reader.nextFrame(frame))`
  - [ ] `  outputFrame = impression.stalkImageAdvance(frame)`
  - [ ] Timing ölçümü

- [ ] TestResult oluştur ve döndür
- [ ] Exception handling

**Tahmini Süre:** 30 dakika
**Bağımlılık:** Impression sınıfı, YOLO model
**Test:** Tek sequence ile test et

---

### 3.5 Yardımcı Fonksiyonlar
**Dosya:** `mot17_benchmark.cpp`

- [ ] `createOutputDirectories()`
  - [ ] `std::filesystem::create_directories(output/v{VERSION}/tracker-only)`
  - [ ] `std::filesystem::create_directories(output/v{VERSION}/end-to-end)`

- [ ] `printProgress(int current, int total, const std::string& testName)`
  - [ ] Progress bar çiz: `[45/126] MOT17-04-FRCNN-CIOU ... Done (3.2s, 289 FPS)`
  - [ ] ANSI renk kodları kullan (opsiyonel)

- [ ] `saveSummaryJSON(const std::string& mode)`
  - [ ] JSON formatında summary oluştur
  - [ ] Dosyaya yaz: `output/v{VERSION}/{mode}/benchmark_summary.json`

- [ ] `saveCombinedReport()`
  - [ ] Her iki mode'un özetini birleştir
  - [ ] Dosyaya yaz: `output/v{VERSION}/combined_report.json`

**Tahmini Süre:** 40 dakika
**Bağımlılık:** JSON library kullanılacaksa ekle (yoksa manuel string concatenation)

---

## 📜 Faz 4: Script Güncellemeleri (Öncelik: 🟡 Orta)

### 4.1 run_trackeval.sh Güncelleme
**Dosya:** `output/run_trackeval.sh`

- [ ] Version parametresi ekle
  - [ ] `VERSION=${1:-$(ls -t output/ | grep "^v" | head -1)}`
  - [ ] Default: En son version

- [ ] Mode parametresi ekle
  - [ ] `MODE=${2:-"tracker-only"}`
  - [ ] Seçenekler: tracker-only, end-to-end

- [ ] TRACKER_DIR'i güncelle
  - [ ] `$TRACKEVAL_DIR/.../opencv-yolo-$VERSION-$MODE/data`

- [ ] Copy komutunu güncelle
  - [ ] `cp output/$VERSION/$MODE/*.txt "$TRACKER_DIR/"`

- [ ] TrackEval çağrısını güncelle
  - [ ] `--TRACKERS_TO_EVAL "opencv-yolo-$VERSION-$MODE"`

**Tahmini Süre:** 20 dakika
**Bağımlılık:** Yok
**Test:** `./run_trackeval.sh v1.0.0 tracker-only`

---

### 4.2 compare_versions.sh Oluşturma
**Dosya:** `scripts/compare_versions.sh` (YENİ)

- [ ] Dosyayı oluştur ve executable yap
  - [ ] `touch scripts/compare_versions.sh && chmod +x scripts/compare_versions.sh`

- [ ] Shebang ve parametre alma
  - [ ] `#!/bin/bash`
  - [ ] `V1=$1`, `V2=$2`
  - [ ] Parametre kontrolü (eksikse hata ver)

- [ ] JSON'lardan metrikleri çıkar
  - [ ] `output/v{V1}/tracker-only/benchmark_summary.json`
  - [ ] `output/v{V2}/tracker-only/benchmark_summary.json`
  - [ ] `jq` veya `python` ile parse et

- [ ] Karşılaştırma tablosu bas
  - [ ] HOTA, MOTA, IDF1 için delta hesapla
  - [ ] ASCII tablo formatı kullan

- [ ] (Opsiyonel) Grafik oluştur
  - [ ] `python3 scripts/analyze_results.py --plot ...`

**Tahmini Süre:** 30 dakika
**Bağımlılık:** jq veya Python
**Test:** `./scripts/compare_versions.sh v1.0.0 v1.0.1`

---

## 🧪 Faz 5: Test ve Doğrulama (Öncelik: ⚡️ Yüksek)

### 5.1 İlk Build
- [ ] CMake configure
  - [ ] `cd build && cmake ..`
  - [ ] Hata kontrolü (version tanımlandı mı?)

- [ ] Compile
  - [ ] `cmake --build .`
  - [ ] `mot17_benchmark` executable oluştu mu?

- [ ] Executable çalıştır (help)
  - [ ] `./mot17_benchmark --help` (eğer arg parsing eklediysen)

**Tahmini Süre:** 5 dakika
**Bağımlılık:** Tüm kod dosyaları tamamlanmış olmalı

---

### 5.2 Küçük Test (Tek Sequence)
- [ ] Tek sequence ile test
  - [ ] `./mot17_benchmark --sequences MOT17-02 --detectors DPM --iou-types CIOU`

- [ ] Output kontrolü
  - [ ] `output/v1.0.0/tracker-only/MOT17-02-DPM-CIOU.txt` oluştu mu?
  - [ ] Dosya boyutu makul mü? (> 0 byte)
  - [ ] MOTChallenge formatında mı? (virgülle ayrılmış)

- [ ] Benchmark summary kontrolü
  - [ ] `output/v1.0.0/tracker-only/benchmark_summary.json` oluştu mu?
  - [ ] JSON valid mi? (`jq . < benchmark_summary.json`)
  - [ ] FPS değeri makul mü? (~200-300 FPS)

**Tahmini Süre:** 10 dakika
**Bağımlılık:** 5.1 tamamlanmış olmalı

---

### 5.3 Orta Ölçekli Test (3 Sequence, 3 IoU)
- [ ] Daha fazla kombinasyon test et
  - [ ] `--sequences MOT17-02,MOT17-04,MOT17-05`
  - [ ] `--iou-types IOU,CIOU,SIOU`
  - [ ] Toplam: 3 seq × 3 det × 3 IoU = 27 test

- [ ] Progress tracking çalışıyor mu?
  - [ ] `[1/27] MOT17-02-DPM-IOU ... Done`
  - [ ] Her test log çıktısı düzgün mü?

- [ ] Hata handling test et
  - [ ] Yanlış sequence ismi ver → crash etmemeli
  - [ ] Eksik det.txt → error message loglamalı

**Tahmini Süre:** 20 dakika
**Bağımlılık:** 5.2 başarılı olmalı

---

### 5.4 Full Benchmark (Tracker-Only)
- [ ] Tracker-only mode full test
  - [ ] `./mot17_benchmark --mode tracker-only`
  - [ ] 126 test çalışıyor mu?
  - [ ] Tahmini süre: ~10-15 dakika

- [ ] Sonuç kontrolü
  - [ ] `ls -l output/v1.0.0/tracker-only/*.txt | wc -l` → 126 dosya?
  - [ ] benchmark_summary.json'da 126 result var mı?

- [ ] TrackEval ile değerlendir
  - [ ] `cd output && ./run_trackeval.sh v1.0.0 tracker-only`
  - [ ] HOTA, MOTA, IDF1 sonuçları çıkıyor mu?

**Tahmini Süre:** 20 dakika (çalıştırma + değerlendirme)
**Bağımlılık:** 5.3 başarılı olmalı

---

### 5.5 Full Benchmark (End-to-End)
- [ ] End-to-end mode full test
  - [ ] `./mot17_benchmark --mode end-to-end`
  - [ ] 42 test çalışıyor mu?
  - [ ] Tahmini süre: ~60-90 dakika (GPU ile)

- [ ] Sonuç kontrolü
  - [ ] `ls -l output/v1.0.0/end-to-end/*.txt | wc -l` → 42 dosya?
  - [ ] benchmark_summary.json'da 42 result var mı?

- [ ] TrackEval ile değerlendir
  - [ ] `cd output && ./run_trackeval.sh v1.0.0 end-to-end`

**Tahmini Süre:** 90+ dakika (çoğu süre YOLO inference)
**Bağımlılık:** 5.4 başarılı olmalı, YOLO model hazır

---

### 5.6 Version Comparison Test
- [ ] İkinci version oluştur
  - [ ] CMakeLists.txt'de `PATCH = 1` yap
  - [ ] Rebuild: `cd build && cmake .. && make`
  - [ ] `./mot17_benchmark --mode tracker-only --sequences MOT17-02`
  - [ ] `output/v1.0.1/` oluştu mu?

- [ ] Version karşılaştırma
  - [ ] `./scripts/compare_versions.sh v1.0.0 v1.0.1`
  - [ ] Tablo çıktısı düzgün mü?
  - [ ] Delta hesaplaması doğru mu?

**Tahmini Süre:** 15 dakika
**Bağımlılık:** 5.4 tamamlanmış olmalı

---

## 📚 Faz 6: Dokümantasyon Güncellemesi (Öncelik: 🟢 Düşük)

### 6.1 CLAUDE.md Güncelleme
**Dosya:** `CLAUDE.md`

- [ ] Benchmark System bölümü ekle
  - [ ] `mot17_benchmark` executable açıklaması
  - [ ] Version yönetimi nasıl yapılır
  - [ ] Kullanım örnekleri

- [ ] Model Paths bölümünü güncelle
  - [ ] Version'a göre output path bilgisi ekle

**Tahmini Süre:** 10 dakika

---

### 6.2 README.md Güncelleme (Eğer varsa)
**Dosya:** `README.md`

- [ ] "Benchmark System" başlığı ekle
- [ ] Quick start guide
  - [ ] Build
  - [ ] Run benchmark
  - [ ] Evaluate results

**Tahmini Süre:** 15 dakika

---

### 6.3 MOT17_USAGE.md Güncelleme
**Dosya:** `MOT17_USAGE.md`

- [ ] Benchmark bölümü ekle
  - [ ] Mode 5: Benchmark Mode
  - [ ] Kullanım örnekleri
  - [ ] Version comparison örnekleri

**Tahmini Süre:** 10 dakika

---

## 🎉 Faz 7: Final Commit ve Tag (Öncelik: ⚡️ Yüksek)

### 7.1 Git Commit
- [ ] Tüm değişiklikleri stage et
  - [ ] `git add CMakeLists.txt benchmark_config.h mot17_benchmark.cpp`
  - [ ] `git add output/run_trackeval.sh scripts/compare_versions.sh`
  - [ ] `git add docs/BENCHMARK-*.md`

- [ ] Commit message yaz
  - [ ] `feat: MOT17 benchmark sistemi eklendi`
  - [ ] Detaylı açıklama: dual-mode, versiyonlu, TrackEval entegre

- [ ] Commit yap
  - [ ] `git commit -m "..."`

**Tahmini Süre:** 5 dakika
**Bağımlılık:** Tüm testler başarılı olmalı

---

### 7.2 Git Tag
- [ ] Version tag oluştur
  - [ ] `git tag -a v1.0.0 -m "Initial benchmark system release"`

- [ ] Tag'i push et (eğer remote varsa)
  - [ ] `git push origin v1.0.0`

**Tahmini Süre:** 2 dakika

---

### 7.3 İlk Baseline Oluştur
- [ ] Full benchmark çalıştır
  - [ ] `./mot17_benchmark`
  - [ ] output/v1.0.0/ tamamlandı mı?

- [ ] TrackEval sonuçlarını kaydet
  - [ ] Tracker-only results
  - [ ] End-to-end results

- [ ] Baseline olarak işaretle
  - [ ] `docs/BENCHMARK-BASELINES.md` dosyası oluştur
  - [ ] v1.0.0 sonuçlarını dokümante et

**Tahmini Süre:** 100+ dakika (full benchmark)

---

## 📊 Toplam İlerleme Özeti

### Zaman Tahminleri
```
Faz 1: Dokümantasyon           → 30 dakika
Faz 2: Temel Yapı              → 45 dakika
Faz 3: Core Implementation     → 175 dakika (~3 saat)
Faz 4: Script Güncellemeleri   → 50 dakika
Faz 5: Test ve Doğrulama       → 160 dakika (~2.5 saat)
Faz 6: Dokümantasyon Güncelle  → 35 dakika
Faz 7: Final Commit            → 107+ dakika

Toplam: ~9-10 saat (full benchmark dahil)
```

### Kritik Path
```
1. benchmark_config.h (15 dk)
2. mot17_benchmark.cpp iskelet (20 dk)
3. runSingleTest_TrackerOnly() (40 dk)
4. runTrackerOnlyMode() (30 dk)
5. Küçük test (10 dk)
6. Full benchmark (10+ dk)

Minimum viable: ~2 saat (sadece tracker-only)
```

---

## ✅ Başarı Kriterleri

### Must Have (Minimum)
- [x] Dokümantasyon tamamlandı
- [ ] CMakeLists.txt güncel
- [ ] benchmark_config.h çalışıyor
- [ ] mot17_benchmark executable build oluyor
- [ ] Tracker-only mode çalışıyor (126 test)
- [ ] TrackEval entegrasyonu çalışıyor
- [ ] Version management çalışıyor

### Should Have (İstenilen)
- [ ] End-to-end mode çalışıyor (42 test)
- [ ] Progress tracking güzel görünüyor
- [ ] JSON summary doğru formatda
- [ ] compare_versions.sh çalışıyor
- [ ] Hata handling sağlam

### Nice to Have (Opsiyonel)
- [ ] ANSI renk kodları progress bar'da
- [ ] Grafik çıktıları (matplotlib vb.)
- [ ] Paralel test execution
- [ ] Web dashboard (gelecek)

---

## 🐛 Bilinen Sorunlar ve Çözümler

### Sorun 1: JSON serialization
**Problem:** C++'ta JSON işleme zor
**Çözüm:** Manuel string concatenation veya `nlohmann/json` library kullan

### Sorun 2: YOLO model yükleme yavaş
**Problem:** Her test için model yüklemek ~5s
**Çözüm:** Aynı IoU tipi testlerinde model'i cache'le

### Sorun 3: Out of memory
**Problem:** Çok fazla trajectory history
**Çözüm:** `maxTrajectoryLength = 50` sınırlı tut

---

## 📝 Notlar

- Her maddedeki tahmini süre **ortalama developer** içindir
- **Bloke durumları**: Bir madde başarısız olursa, ona bağımlı maddeler beklemeli
- **Testing stratejisi**: Her faz sonunda test yap, hata varsa düzelt, devam et
- **Git commits**: Her faz sonunda commit yap (incremental progress)
- **Yarım kalırsa**: Bu dosyaya bak, kaldığın yerden devam et

---

**Son Güncelleme:** 2025-01-20
**Güncelleyen:** Halil Erden / Claude
**Durum:** 🚧 In Progress (1/11 tamamlandı)
