# Tracker Results & Evaluation

Bu dizin tracker sonuçlarını ve TrackEval değerlendirme scriptini içerir.

## Dosyalar

- `MOT17-*.txt` - Tracker çıktı dosyaları (MOTChallenge formatında)
- `run_trackeval.sh` - TrackEval değerlendirme scripti
- `README.md` - Bu dosya

## Kullanım

### 1. Tüm Sonuçları Değerlendir

```bash
cd output
./run_trackeval.sh
```

Bu komut `output/` dizinindeki tüm `MOT17-*.txt` dosyalarını otomatik olarak bulup değerlendirir.

### 2. Belirli Sequence'i Değerlendir

```bash
cd output
./run_trackeval.sh MOT17-02-DPM.txt
```

### 3. Birden Fazla Belirli Sequence

```bash
cd output
./run_trackeval.sh MOT17-02-DPM.txt MOT17-04-DPM.txt
```

## Script Ne Yapar?

`run_trackeval.sh` scripti otomatik olarak:

1. ✅ TrackEval dizin yapısını kontrol eder
2. ✅ Ground truth sembolik bağlantısını oluşturur (yoksa)
3. ✅ Tracker sonuçlarını doğru dizine kopyalar
4. ✅ Seqmaps dosyasını oluşturur
5. ✅ Python bağımlılıklarını kontrol eder
6. ✅ TrackEval değerlendirmesini çalıştırır
7. ✅ Sonuç dosyalarının yerini gösterir

## Değerlendirme Sonuçları

Sonuçlar şu dizinde kaydedilir:
```
thirdParty/TrackEval/data/trackers/mot_challenge/MOT17-train/opencv-yolo/
├── pedestrian_summary.txt      # Özet metrikler
├── pedestrian_detailed.csv     # Detaylı frame-by-frame sonuçlar
└── data/
    └── MOT17-*.txt             # Tracker girdileri
```

## Metrikler

### HOTA (Higher Order Tracking Accuracy) - **ÖNERİLEN**
- **HOTA**: Genel tracking kalitesi (0-100%)
- **DetA**: Detection accuracy
- **AssA**: Association accuracy
- **LocA**: Localization accuracy

### CLEAR (Traditional MOT Metrics)
- **MOTA**: Multiple Object Tracking Accuracy
- **MOTP**: Multiple Object Tracking Precision
- **MT**: Mostly Tracked tracks
- **ML**: Mostly Lost tracks
- **FP/FN**: False Positives / False Negatives
- **IDSW**: ID Switches

### Identity Metrics
- **IDF1**: ID F1 Score
- **IDR**: ID Recall
- **IDP**: ID Precision

## Örnek Çıktı

```
HOTA: opencv-yolo-pedestrian    HOTA    DetA    AssA    ...
MOT17-02-DPM                    26.31   17.14   41.23   ...
MOT17-04-DPM                    XX.XX   XX.XX   XX.XX   ...
COMBINED                        XX.XX   XX.XX   XX.XX   ...

CLEAR: opencv-yolo-pedestrian   MOTA    MOTP    MT      ML      ...
MOT17-02-DPM                    0.91    77.78   7       40      ...
...
```

## Sorun Giderme

### "No such file or directory"
```bash
# Script'i output/ dizininden çalıştırın
cd output
./run_trackeval.sh
```

### "Permission denied"
```bash
# Executable iznini verin
chmod +x run_trackeval.sh
./run_trackeval.sh
```

### "Ground truth bulunamadı"
```bash
# MOT17-Data sembolik linkini kontrol edin
ls -la ../MOT17-Data
```

### "Tracker sonucu bulunamadı"
```bash
# output/ dizininde MOT17-*.txt dosyaları olmalı
ls MOT17-*.txt
```

## Manuel TrackEval Çalıştırma

Script kullanmak istemezseniz:

```bash
cd thirdParty/TrackEval

# Sonuçları kopyala
cp ../../output/MOT17-02-DPM.txt data/trackers/mot_challenge/MOT17-train/opencv-yolo/data/

# Değerlendirmeyi çalıştır
python3 scripts/run_mot_challenge.py \
    --BENCHMARK MOT17 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL opencv-yolo \
    --METRICS HOTA CLEAR Identity \
    --USE_PARALLEL False \
    --PLOT_CURVES False
```

## Yeni Sequence Ekleme

1. Tracker'ı çalıştır ve sonuçları kaydet:
```cpp
processMOT17Sequence(
    projectBasePath + "/MOT17-Data/MOT17/train/MOT17-04-DPM",
    projectBasePath + "/output/MOT17-04-DPM.txt",
    projectBasePath + "/models/yolov9s.onnx",
    false, runOnGPU
);
```

2. TrackEval çalıştır:
```bash
cd output
./run_trackeval.sh MOT17-04-DPM.txt
```

## İyileştirme İpuçları

Düşük metrik skorları için:

1. **Düşük MOTA (<50%)**:
   - Detection threshold düşür (conf > 0.2)
   - Daha güçlü YOLO modeli kullan
   - Input resolution artır

2. **Yüksek ID Switches**:
   - IoU threshold optimize et
   - Predict frame limit artır
   - Re-ID feature ekle

3. **Düşük Recall**:
   - NMS threshold ayarla
   - Detection confidence threshold düşür

4. **Yüksek False Positives**:
   - Confidence threshold yükselt
   - Post-processing ekle

## Referanslar

- [TrackEval GitHub](https://github.com/JonathonLuiten/TrackEval)
- [MOTChallenge](https://motchallenge.net/)
- [HOTA Metrics Paper](https://link.springer.com/article/10.1007/s11263-020-01375-2)
