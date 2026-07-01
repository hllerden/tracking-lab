# TrackEval Hızlı Başlangıç Kılavuzu

## 🚀 Tek Komut ile Değerlendirme

```bash
cd output
./run_trackeval.sh
```

Bu kadar! Script otomatik olarak her şeyi halleder.

## 📋 Adım Adım Kullanım

### 1️⃣ Tracker Sonuçlarını Üret

```bash
# Projeyi derle
cd build
cmake --build .

# main.cpp'de MOT17 processing kodunu aktif et
# Satır 456-462 arası yorum işaretlerini kaldır
./yoloExercise

# Sonuç: output/MOT17-02-DPM.txt oluşturulur
```

### 2️⃣ Değerlendirmeyi Çalıştır

```bash
cd output
./run_trackeval.sh
```

### 3️⃣ Sonuçları İncele

Console çıktısında metrikleri göreceksin:

```
HOTA: 26.31%  ← Genel tracking kalitesi
MOTA: 0.91%   ← Multiple Object Tracking Accuracy
IDF1: 26.91%  ← Identity F1 Score
```

Detaylı sonuçlar:
- `thirdParty/TrackEval/.../pedestrian_summary.txt`
- `thirdParty/TrackEval/.../pedestrian_detailed.csv`

## 🎯 Farklı Kullanım Senaryoları

### Sadece Belirli Sequence'i Test Et

```bash
./run_trackeval.sh MOT17-02-DPM.txt
```

### Birden Fazla Sequence

```bash
./run_trackeval.sh MOT17-02-DPM.txt MOT17-04-DPM.txt
```

### Tüm Sequence'leri Toplu İşle

```bash
# 1. Tüm sequence'leri işleyecek kodu main.cpp'ye ekle
# (Satır 465-473 arası)

# 2. Çalıştır
./yoloExercise

# 3. Hepsini değerlendir
cd output
./run_trackeval.sh
```

## 📊 Metrik Anlamları

| Metrik | Anlamı | İyi Değer |
|--------|--------|-----------|
| **HOTA** | Genel tracking kalitesi | >50% |
| **MOTA** | Tracking doğruluğu | >50% |
| **IDF1** | Identity eşleştirme | >60% |
| **MT** | Çoğunlukla takip edilen | Yüksek |
| **ML** | Çoğunlukla kaybolan | Düşük |
| **IDSW** | ID değişim sayısı | Düşük |
| **FP** | Yanlış pozitif | Düşük |
| **FN** | Yanlış negatif | Düşük |

## 🔧 Sorun Giderme

### Script çalışmıyor

```bash
# Executable iznini ver
chmod +x run_trackeval.sh

# Doğru dizinden çalıştır
cd output
./run_trackeval.sh
```

### MOT17-Data bulunamadı

```bash
# Sembolik link kontrolü
ls -la ../MOT17-Data

# Yoksa oluştur
ln -s /path/to/MOT17-Data ../MOT17-Data
```

### Python hatası

```bash
# Bağımlılıkları yükle
pip3 install numpy scipy
```

## 💡 İyileştirme İpuçları

### Düşük MOTA (<10%)
**Sorun**: Çok fazla nesne kaçırılıyor
**Çözüm**:
```cpp
// inference.cpp'de confidence threshold düşür
float confThreshold = 0.2; // 0.5 yerine
```

### Yüksek ID Switches
**Sorun**: Nesneler sık sık ID değiştiriyor
**Çözüm**:
```cpp
// impression.h'de predict frame limit artır
settings.predictFrameLimit = 40; // 20 yerine
```

### Düşük Recall
**Sorun**: Nesnelerin çoğu tespit edilemiyor
**Çözüm**:
- Daha güçlü model kullan: `yolov9m.onnx`
- Input resolution artır: `Size(1280, 1280)`
- NMS threshold ayarla

## 📈 Benchmark Karşılaştırması

| Tracker | HOTA | MOTA | IDF1 |
|---------|------|------|------|
| **State-of-art** | 63.1 | 79.6 | 76.2 |
| **Good** | 50+ | 60+ | 65+ |
| **Medium** | 35-50 | 40-60 | 50-65 |
| **Başlangıç** | <35 | <40 | <50 |
| **opencv-yolo (mevcut)** | 26.3 | 0.9 | 26.9 |

Hedef: **Medium** seviyeye çıkmak!

## 🎓 Sonraki Adımlar

1. **Detection İyileştir**
   - Daha güçlü YOLO modeli (`yolov9m`, `yolov9e`)
   - Confidence threshold optimizasyonu
   - Multi-scale testing

2. **Tracking İyileştir**
   - IoU threshold grid search
   - Farklı IoU tipleri dene (CIOU, DIOU, GIOU)
   - Predict frame limit optimizasyonu

3. **Re-ID Ekle**
   - Appearance-based matching
   - DeepSORT entegrasyonu
   - Feature embedding

4. **Tüm Sequence'lerde Test**
   - 7 train sequence'i değerlendir
   - Average metriklere bak
   - En zor sequence'leri belirle

## 📚 Kaynaklar

- [TrackEval Docs](../thirdParty/TrackEval/docs/)
- [MOT17_USAGE.md](../MOT17_USAGE.md) - Detaylı kılavuz
- [README.md](README.md) - Tam dokümantasyon
- [CLAUDE.md](../CLAUDE.md) - Proje mimari bilgisi

## ✅ Checklist

- [ ] MOT17-Data linki var mı?
- [ ] output/*.txt dosyaları var mı?
- [ ] run_trackeval.sh executable mı?
- [ ] Python bağımlılıkları yüklü mü?
- [ ] İlk değerlendirmeyi çalıştırdın mı?
- [ ] Sonuçları inceledi mi?
- [ ] İyileştirme hedefleri belirledin mi?

---

**İpucu**: İlk kez kullanıyorsan önce tek bir sequence ile test et, sonra toplu işleme geç!
