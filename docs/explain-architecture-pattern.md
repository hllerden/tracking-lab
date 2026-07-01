# Mimari Desen Analiz Raporu
## OpenCV-YOLO Nesne Tespit ve Takip Sistemi

**Rapor Tarihi:** 2025-10-17
**Proje:** opencv-yolo
**Analiz Kapsamı:** Sistem Mimarisi, Tasarım Desenleri ve Bileşen Etkileşimleri

---

## Yönetici Özeti

Bu doküman, OpenCV-YOLO nesne tespit ve takip sisteminde kullanılan mimari desenleri ve tasarım ilkelerini kapsamlı biçimde inceler. Sistem; bilgisayarlı görü çıkarımını gelişmiş takip algoritmalarıyla birleştiren iyi yapılandırılmış, modüler bir mimariye sahiptir.

---

## 1. Genel Mimari Desen

### 1.1 Katmanlı Mimari

Sistem, **Üç Katmanlı Mimari** yaklaşımını izler:

```
┌─────────────────────────────────────────────┐
│         Uygulama Katmanı (main.cpp)         │
│         UI, Video G/Ç, Orkestrasyon         │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│     İş Mantığı Katmanı (Impression)         │
│    Üst Düzey API, Boru Hattı Yönetimi       │
└─────────────────────────────────────────────┘
                      ↓
┌──────────────────┬──────────────────────────┐
│  Inference       │         KalmanIoUTracker          │
│  (Tespit)        │         (Takip)          │
└──────────────────┴──────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│    Altyapı Katmanı (OpenCV, CUDA)           │
│  DNN Modülü, Kalman Filtresi, Algoritmalar  │
└─────────────────────────────────────────────┘
```

**Faydalar:**
- Sorumlulukların net ayrımı
- Bağımsız modül geliştirme ve test imkanı
- Bakımı ve genişletilmesi kolay
- Her katmanda teknoloji yığını değiştirilebilir

---

## 2. Çekirdek Tasarım Desenleri

### 2.1 Facade Deseni

**Uygulama:** `Impression` sınıfı ([impression.h](impression.h))

`Impression` sınıfı, karmaşık alt sistemleri sadeleştiren bir fasat görevi görür:

```cpp
class Impression {
    KalmanIoUTracker *stalker;       // Takip alt sistemi
    Inference *inference;   // Tespit alt sistemi
    ImpressionSettings settingsParams;

    Mat stalkImageDefault(const cv::Mat frame);
    Mat stalkImageAdvance(const cv::Mat inputFrame);
};
```

**Amaç:**
- Karmaşık alt sistemlere tek ve basit bir arayüz sunmak
- Inference ve KalmanIoUTracker koordinasyonundaki karmaşıklığı gizlemek
- Tespit ve takip boru hattının tamamını yönetmek
- Kullanıcı kodunun iç koordinasyonu bilmesini gereksiz kılmak

**Örnek Kullanım:**
```cpp
Impression impression;
impression.createInfrance(modelPath);
Mat result = impression.stalkImageAdvance(frame);
```

---

### 2.2 Strateji Deseni

**Uygulama:** `KalmanIoUTracker` içinde IoU tipi seçimi ([kalman_iou_tracker.h](kalman_iou_tracker.h):31-79)

Sistem, çalışma zamanında değiştirilebilen birden fazla IoU hesaplama stratejisi sunar:

```cpp
enum class IOUTYPE {
    IOU,    // Standart Kesişim / Birleşim
    GIOU,   // Genelleştirilmiş IoU
    DIOU,   // Mesafe IoU
    CIOU,   // Tam IoU (varsayılan)
    SIOU,   // Yumuşak IoU
    AIOU    // Alfa IoU
};

IOUTYPE selectedIoUType = IOUTYPE::CIOU;
double computeIoU(IOUTYPE type, const cv::Rect& boxA, const cv::Rect& boxB);
```

**Faydalar:**
- Çalışma zamanında algoritma seçimi
- Yeni IoU varyantlarının kolay eklenmesi
- Stratejiler arası performans karşılaştırması
- Yapılandırma esnekliği

**Uygulama Ayrıntıları:**
- Varsayılan strateji: CIOU (en kapsamlısı)
- Strateji seçimi `setSelectedIoUType()` üzerinden yapılır
- Her stratejinin farklı kullanım alanları belgelenmiştir

---

### 2.3 State Deseni

**Uygulama:** `TrackedObject::State` ([kalman_iou_tracker.h](kalman_iou_tracker.h):10-25)

Takip edilen nesneler, davranışlarını etkileyen bir durumu korur:

```cpp
struct TrackedObject {
    enum State {
        TRACKING,  // Nesne aktif olarak tespit ediliyor
        LOST       // Nesne tespit edilmedi, tahmin kullanılıyor
    };
    State state = TRACKING;
    int missedFrames = 0;
    // ... duruma bağlı davranış
};
```

**Durum Geçişleri:**
```
TRACKING ──[Tespit Kayboldu]──> LOST
    ↑                             │
    └────[Tespit Bulundu]─────────┘
                                  │
                           [Limit Aşıldı]
                                  ↓
                              [SİLİNDİ]
```

**Durum Bazlı Davranışlar:**
- **TRACKING:** Tespit verileri kullanılır, `missedFrames` sıfırlanır
- **LOST:** Kalman tahmini kullanılır, `missedFrames` artırılır
- Görselleştirme duruma göre değişir

---

### 2.4 Gözlemci Deseni (Örtük)

**Uygulama:** Takip boru hattı

Sistem, şu akışta gözlemci benzeri davranış sergiler:

1. `Inference` nesneleri tespit eder (yayıncı)
2. `KalmanIoUTracker` tespitleri alır (abone)
3. İzleyiciler yeni verilere göre güncellenir

```cpp
// Tespit aşaması (yayın)
std::vector<Detection> output = inference->runInference(frame);

// İşleme aşaması (abonelik)
std::vector<TrackedObject> trackers = stalker->processDetectionsWithHungarianTrackers(boxes);
```

---

### 2.5 Şablon Yöntem Deseni

**Uygulama:** Takip süreç akışı

Takip iş akışı, özelleştirilebilir adımları olan tanımlı bir algoritmayı izler:

```cpp
std::vector<TrackedObject> processDetectionsWithHungarianTrackers(
    const std::vector<cv::Rect> &detections) {

    // Adım 1: Tahmin (Kalman Filtresi)
    for (auto& tracker : trackers) {
        predictNextState(tracker);
    }

    // Adım 2: Eşleştirme (Macar Algoritması)
    performAssociation(detections);

    // Adım 3: Güncelleme (eşleşen izleyiciler)
    updateTrackers(matches);

    // Adım 4: Başlatma (yeni tespitler)
    addNewTrackers(unmatched);

    // Adım 5: Temizlik (kayıp izleyiciler)
    removeLostTrackers();

    return trackers;
}
```

---

### 2.6 Factory Deseni (Örtük)

**Uygulama:** Kalman filtresi oluşturma ([kalman_iou_tracker.cpp](kalman_iou_tracker.cpp):30-46)

```cpp
cv::KalmanFilter KalmanIoUTracker::createKalmanFilter() {
    // 6 durum: x, y, vx, vy, w, h
    // 6 ölçüm: x, y, vx, vy, w, h
    cv::KalmanFilter kf(6, 6, 0);

    // Geçiş matrisini yapılandır
    // Gürültü matrislerini yapılandır
    // Yapılandırılmış örneği döndür

    return kf;
}
```

**Amaç:**
- Karmaşık Kalman filtresi yapılandırmasını kapsüllemek
- Tutarlı başlatma sağlamak
- Oluşturma karmaşıklığını gizlemek

---

## 3. Mimari İlkeler

### 3.1 Tek Sorumluluk İlkesi (SRP)

Her sınıf, iyi tanımlanmış tek bir amaca sahiptir:

| Sınıf | Sorumluluk |
|-------|------------|
| **Inference** | YOLO model çıkarımı ve tespiti |
| **KalmanIoUTracker** | Nesne takibi ve yörünge yönetimi |
| **Impression** | Boru hattı orkestrasyonu |
| **HungarianAlgorithm** | Optimal atama çözümü |
| **TrackedObject** | Tek bir izlenen nesnenin durumu ve verileri |

---

### 3.2 Açık/Kapalı İlkesi

**Genişletmeye açık, değiştirmeye kapalı:**

1. **IoU Stratejileri:** Yeni IoU tipleri mevcut kodu değiştirmeden eklenebilir
   ```cpp
   // Yeni IoU tipi ekleme:
   enum class IOUTYPE {
       // ... mevcut tipler
       NEW_IOU_TYPE  // Sadece buraya ekleyin
   };
   ```

2. **Tespit Modelleri:** Sistem herhangi bir YOLO sürümüyle (v5, v8, v9, v10, v11) çalışır
   - Model yolu ve yapılandırma dışsallaştırılmıştır
   - Yeni modeller için kod değişikliği gerekmez

---

### 3.3 Bağımlılıkların Ters Çevrilmesi İlkesi

**Üst düzey modüller soyutlamalara bağlıdır:**

```cpp
class Impression {
private:
    KalmanIoUTracker *stalker;      // Bağımlılık enjeksiyonu
    Inference *inference;  // Bağımlılık enjeksiyonu
};
```

- `Impression`, bağımlılıkları doğrudan oluşturmaz
- Tam DIP uyumluluğu için arayüzler kullanılarak genişletilebilir

---

### 3.4 Sorumluluk Ayrımı

Mimari, farklı sorumlulukları net biçimde ayırır:

1. **Tespit Sorumluluğu:** `Inference` sınıfı
   - Model yükleme
   - Önişleme
   - Sinir ağı çıkarımı
   - Son işleme (NMS)

2. **Takip Sorumluluğu:** `KalmanIoUTracker` sınıfı
   - Hareket tahmini
   - Veri eşleştirme
   - Durum yönetimi
   - Yörünge yönetimi

3. **Sunum Sorumluluğu:** Uygulama katmanı
   - Görselleştirme
   - Kullanıcı etkileşimi
   - Video G/Ç

---

## 4. Temel Algoritmalar ve Teknikler

### 4.1 Optimal Atama için Macar Algoritması

**Konum:** [HungarianAlgorithm.cpp](HungarianAlgorithm.cpp)
**Amaç:** Tespit ve izleyici eşleştirmesi için atama problemini çözmek

**Matematiksel Temel:**
- Minimum maliyetli atamayı O(n³) zamanda çözer
- IoU mesafelerine dayalı maliyet matrisi kullanır
- Optimal eşleşmeyi garanti eder

**Entegrasyon:**
```cpp
// Maliyet matrisini oluştur (mesafe için 1 - IoU)
std::vector<std::vector<double>> costMatrix;
for (tracker : trackers) {
    for (detection : detections) {
        double iou = computeIoU(selectedIoUType, tracker.box, detection);
        costMatrix[i][j] = 1.0 - iou;  // Maliyete dönüştür
    }
}

// Atamayı çöz
HungarianAlgorithm hungarian;
std::vector<int> assignment;
hungarian.Solve(costMatrix, assignment);
```

---

### 4.2 Hareket Tahmini için Kalman Filtresi

**Durum Vektörü (6B):**
```
x  = [x, y, vx, vy, w, h]ᵀ
     │  │  │   │   │  └─ Yükseklik
     │  │  │   │   └──── Genişlik
     │  │  │   └──────── Y hızı
     │  │  └──────────── X hızı
     │  └─────────────── Y konumu
     └────────────────── X konumu
```

**Yapılandırma:**
- Süreç gürültüsü: 0.01 (düşük, akıcı hareket varsayar)
- Ölçüm gürültüsü: 0.5 (orta, tespit değişkenliğini kapsar)
- Sabit hız hareket modeli

**Kullanım Kalıbı:**
```cpp
// Tahmin adımı (her kare)
cv::Mat prediction = tracker.kalmanFilter.predict();

// Düzeltme adımı (tespit mevcutsa)
cv::Mat measurement = (cv::Mat_<float>(6,1) << x, y, vx, vy, w, h);
tracker.kalmanFilter.correct(measurement);
```

---

### 4.3 Birden Fazla IoU Varyantı

Sistem, farklı senaryolar için 6 adet IoU hesaplama yöntemi uygular:

| IoU Tipi | Formül | Kullanım Senaryosu |
|----------|--------|--------------------|
| **IOU** | Intersection / Union | Genel amaçlı, temel |
| **GIOU** | IoU - \|C-(A∪B)\| / \|C\| | Düşük örtüşme senaryoları |
| **DIOU** | IoU - d²/c² | Konum duyarlı eşleştirme |
| **CIOU** | DIoU - αv | Konum + en-boy oranı (varsayılan) |
| **SIOU** | IoU / (IoU + s * (1 - IoU)) | Yumuşak, pürüzsüz eşleşme (s slack faktörü) |
| **AIOU** | IoU^α | Ayarlanabilir duyarlılık (α üssü) |

**Varsayılan tip CIOU'dur** ([kalman_iou_tracker.h](kalman_iou_tracker.h):133) çünkü en kapsamlı eşleştirme kriterlerini sunar.

---

## 5. Veri Akışı Mimarisi

### 5.1 Ana İşleme Boru Hattı

```
Giriş Karesi
    │
    ├──> [Inference Modülü]
    │         │
    │         ├─> Önişleme (yeniden boyutlandır, normalize et)
    │         ├─> DNN İleri Geçişi (CUDA hızlandırmalı)
    │         └─> Son İşleme (NMS, filtreleme)
    │              │
    │              └─> std::vector<Detection>
    │                       │
    ├────────────────────────┘
    │
    └──> [KalmanIoUTracker Modülü]
              │
              ├─> Kalman Tahmini (tüm izleyiciler)
              ├─> Maliyet Matrisi Oluşturma (IoU)
              ├─> Macar Ataması
              ├─> Eşleşen İzleyicileri Güncelle
              ├─> Yeni İzleyiciler Oluştur
              ├─> Kayıp İzleyicileri Ele Al
              └─> Ölü İzleyicileri Kaldır
                   │
                   └─> std::vector<TrackedObject>
                            │
                            └──> Görselleştirme ve Çıktı
```

---

### 5.2 Takip Durum Makinesi

```
Yeni Tespit
     │
     ├──[Eşleşme Yok]──> Yeni İzleyici Oluştur ──> TRACKING
     │                       │
     └──[Eşleşme Bulundu]────┘
              │
              ├──[Tespit Tamam]──> Durumu Güncelle ──> TRACKING
              │
              └──[Tespit Kayboldu]──> Tahmini Kullan ──> LOST
                        │
                        ├──[Yeniden Bulundu]─────────> TRACKING
                        │
                        ├──[Sınır İçinde]──> Tahmine Devam Et
                        │
                        └──[Limit Aşıldı]──> İzleyiciyi Sil
```

---

## 6. Yapılandırma ve Genişletilebilirlik

### 6.1 ImpressionSettings Yapılandırması

```cpp
struct ImpressionSettings {
    bool usePredict = true;           // Tespit başarısız olduğunda Kalman kullan
    u_int8_t predictFrameLimit = 20;  // Tespit olmadan izin verilen maksimum kare
    bool printTrajectory = false;     // Nesne yollarını çiz
    bool printCenter = true;          // Merkez noktaları çiz
};
```

**Tasarım Faydaları:**
- Merkezi yapılandırma
- Çalışma zamanında yeniden yapılandırma mümkün
- Varsayılan değerler net
- Kolay kullanım için özel operatör aşırı yükleme

---

### 6.2 KalmanIoUTracker Yapılandırması

```cpp
class KalmanIoUTracker {
    IOUTYPE selectedIoUType = IOUTYPE::CIOU;
    bool removeOutOfBounds = true;
    int predictUpdateFrameCountInLostTracks = 20;
    bool usePredictionInLostTargets = true;
    cv::Rect cameraBounds;
};
```

**Esneklik:**
- Tüm parametrelerin getter/setter'ı vardır
- Kullanım senaryosuna göre ayarlanabilir
- Varsayılan değerler üretime hazırdır

---

## 7. Performans Hususları

### 7.1 GPU Hızlandırması

```cpp
Inference(const std::string &onnxModelPath,
          const cv::Size &modelInputShape = {640, 640},
          const std::string &classesTxtFile = "",
          const bool &runWithCuda = true);  // GPU hızlandırma bayrağı
```

**CUDA Entegrasyonu:**
- CUDA arka ucu ile OpenCV DNN
- Çıkarım için belirgin hız artışı
- CUDA yoksa CPU'ya geri dönüş

---

### 7.2 Algoritma Karmaşıklığı

| Bileşen | Karmaşıklık | Notlar |
|---------|-------------|--------|
| YOLO Çıkarımı | Kare başına O(1) | GPU hızlandırmalı, sabit süre |
| Kalman Tahmini | O(n) | n = izleyici sayısı |
| Maliyet Matrisi | O(n×m) | n = izleyiciler, m = tespitler |
| Macar Algoritması | O(n³) | Çok sayıda nesne için darboğaz |
| İzleyicileri Güncelle | O(n) | Eşleşen çiftler kadar doğrusal |

**Optimizasyon Notları:**
- Macar algoritması hesaplama açısından darboğazdır
- Maksimum izleyici sayısını sınırlamak performansı iyileştirir
- Gerçek zamanlı çalışma için GPU çıkarımı kritiktir

---

### 7.3 Bellek Yönetimi

**Tasarım Özellikleri:**
- **Dinamik ayrım:** İzleyiciler ihtiyaç oldukça yaratılır/silinir
- **Yörünge depolama:** Verimli push/pop için `std::deque`
- **Vektör kullanımı:** Dinamik tespit listeleri için `std::vector`
- **Kalman filtresi:** İzleyici başına önceden ayrılmış matrisler

**Bellek Büyümesi:**
- Yörünge deque'leri sınırsız büyüyebilir
- Uzun süreli uygulamalarda yörünge uzunluğu sınırı eklenmesi düşünülmelidir

---

## 8. Genişletilebilirlik Noktaları

### 8.1 Yeni IoU Tipleri Ekleme

**Konum:** [kalman_iou_tracker.h](kalman_iou_tracker.h):31-79, [kalman_iou_tracker.cpp](kalman_iou_tracker.cpp)

```cpp
// 1. enum değerini ekle
enum class IOUTYPE {
    // ... mevcut tipler
    NEW_IOU
};

// 2. Hesaplamayı uygula
double KalmanIoUTracker::calculateNewIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    // Uygulama
}

// 3. switch durumunu ekle
double KalmanIoUTracker::computeIoU(IOUTYPE type, const cv::Rect& boxA, const cv::Rect& boxB) {
    switch (type) {
        // ... mevcut durumlar
        case IOUTYPE::NEW_IOU:
            return calculateNewIoU(boxA, boxB);
    }
}
```

---

### 8.2 Özel Takip Algoritmaları

Sistem, alternatif takip yöntemlerini destekleyecek şekilde genişletilebilir:

```cpp
// Mevcut: Kalman ile Macar
std::vector<TrackedObject> processDetectionsWithHungarianTrackers(...);

// Olası genişletmeler:
std::vector<TrackedObject> processDetectionsWithDeepSort(...);
std::vector<TrackedObject> processDetectionsWithCenterTrack(...);
std::vector<TrackedObject> processDetectionsWithByteTrack(...);
```

**DeepSort Entegrasyonu:**
- `thirdParty/deepsort/` dizininde hazır
- Şu anda [CMakeLists.txt](CMakeLists.txt):38-58 içinde yorum satırında
- Yorumları kaldırıp ekleri güncelleyerek etkinleştirilebilir

---

### 8.3 Model Esnekliği

Sistem, herhangi bir YOLO sürümünü destekler:

```cpp
// YOLOv5
Inference inf(basePath + "/models/yolov5s.onnx", Size(640, 640), "", true);

// YOLOv8
Inference inf(basePath + "/models/yolov8s.onnx", Size(640, 640), "", true);

// YOLOv9
Inference inf(basePath + "/models/yolov9s.onnx", Size(640, 640), "", true);

// YOLOv11
Inference inf(basePath + "/models/yolov11n-face.onnx", Size(640, 640), "", true);

// Özel eğitilmiş model
Inference inf(basePath + "/models/best.onnx", Size(640, 640), "", true);
```

---

## 9. Kaçınılan Anti-Desenler

### 9.1 God Object Anti-Deseni (Kaçınıldı)

- Sistem, odaklı sınıflara düzgün biçimde ayrılmıştır
- Her sınıfın net ve sınırlı sorumlulukları vardır
- Hiçbir sınıf her işi yapmaya çalışmaz

### 9.2 Sıkı Bağlılık (Kaçınıldı)

- `Impression` bir arabulucu görevi görür
- Bileşenler iyi tanımlanmış arayüzler üzerinden iletişim kurar
- Uygulamalar (ör. farklı IoU stratejileri) kolayca değiştirilebilir

### 9.3 Sert Kodlanmış Yapılandırma (Kaçınıldı)

- Tüm kritik parametreler yapılandırılabilir
- Model yolları dışarıdan yönetilir
- Eşikler ve limitler ayarlanabilir
- Varsayılan değerler mantıklıdır ancak değiştirilebilir

---

## 10. Öneriler

### 10.1 Gelecekteki Mimari İyileştirmeler

1. **Arayüz Çıkarma:**
   ```cpp
   class ITracker {
       virtual std::vector<TrackedObject> track(const std::vector<cv::Rect>&) = 0;
   };

   class KalmanIoUTracker : public ITracker { ... };
   class DeepSortTracker : public ITracker { ... };
   ```

2. **Yapılandırma Yönetimi:**
   - Yapılandırma dosyası desteği (YAML/JSON) ekleyin
   - Çalışma zamanı yapılandırma yeniden yükleme
   - Profile dayalı yapılandırmalar (düşük gecikme, yüksek doğruluk vb.)

3. **Eklenti Mimarisi:**
   - Takip algoritmalarının dinamik yüklenmesi
   - Özel IoU implementasyonları eklenti olarak
   - Model formatı dönüştürücüleri

4. **Performans İzleme:**
   - Zamanlama metriklerinin toplanması
   - FPS takibi
   - Bileşen bazlı profil çıkarma

---

### 10.2 Kod Kalitesi İyileştirmeleri

1. **İstisna Yönetimi:**
   - Kapsamlı hata yakalama ekleyin
   - Kaynak temizliğini garanti altına alın
   - Kademeli bozulma sağlanması

2. **Günlükleme Çerçevesi:**
   - `std::cout` yerine yapılandırılmış günlükleme kullanın
   - Günlük seviyeleri (DEBUG, INFO, WARN, ERROR)
   - Performans etkisi minimum tutulur

3. **Birim Testi:**
   - IoU implementasyonlarını ayrı ayrı test edin
   - Kalman filtresi tahminlerini test edin
   - Macar algoritması doğruluğunu test edin

---

## 11. Sonuç

OpenCV-YOLO nesne tespit ve takip sistemi, sağlam yazılım mühendisliği prensipleri ile tasarım desenlerinin etkin kullanımını sergiler. Mimari şu özelliklere sahiptir:

**Güçlü Yanlar:**
- Sorumlulukların net ayrıldığı iyi yapılandırılmış tasarım
- Esnek ve genişletilebilir yapı
- Birden fazla algoritma seçeneği (IoU tipleri)
- Performans odaklı (GPU hızlandırması)
- Bakımı kolay kod tabanı

**Mimari Öne Çıkanlar:**
- **Facade Deseni** karmaşık alt sistemleri basitleştirir
- **Strateji Deseni** algoritma seçimini etkin kılar
- **State Deseni** izleyici yaşam döngüsünü yönetir
- **Katmanlı Mimari** modülerliği teşvik eder

**Üretim Hazırlığı:**
- Farklı senaryolar için yapılandırılabilir
- Birden fazla YOLO sürümünü destekler
- GPU hızlandırması ile gerçek zamanlı çalışabilir
- Kenar durumlarını (kayıp izler, sınır dışı) ele alır

Sistem, sağlam nesne tespiti ve takibi gerektiren bilgisayarlı görü uygulamaları için güçlü bir temel sunar.

---

## Ek: Önemli Dosya Referansları

| Dosya | Amaç | İlgili Satırlar |
|-------|------|-----------------|
| [inference.h](inference.h) | YOLO tespit arayüzü | 16-23 (Detection yapısı), 25-57 (Inference sınıfı) |
| [kalman_iou_tracker.h](kalman_iou_tracker.h) | Takip sistemi arayüzü | 10-25 (TrackedObject), 31-79 (IoU tipleri), 82-114 (API) |
| [impression.h](impression.h) | Üst düzey API | 34-57 (ImpressionSettings), 58-84 (Impression sınıfı) |
| [HungarianAlgorithm.h](HungarianAlgorithm.h) | Atama çözücü | 8-24 (Hungarian API'si) |
| [CMakeLists.txt](CMakeLists.txt) | Derleme yapılandırması | 14-26 (Bağımlılıklar), 30-61 (Kaynaklar) |

---

**Doküman Sürümü:** 1.0
**Son Güncelleme:** 2025-10-17
**Sorumlu Ekip:** Mimari Dokümantasyon Ekibi
