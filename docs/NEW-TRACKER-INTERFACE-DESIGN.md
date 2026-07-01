# Yeni Tracker Arayüz Tasarımı

**Tarih:** 17 Ekim 2025
**Yazar:** Halil Erden & Claude Code

## Motivasyon

Eski `ITracker` arayüzü **KalmanIoUTracker'a aşırı spesifik** bir tasarımdı:
- IoUType enum sadece IoU-tabanlı trackerlar için anlamlı
- drawPredictions(), drawCenters(), drawDirections() gibi Kalman-spesifik görselleştirme metodları
- Tüm tracker parametreleri arayüz seviyesinde expose edilmişti
- DeepSORT, ByteTrack gibi farklı trackerlar bu arayüze uyarlanamıyor

**Problem:** Her yeni tracker için yeni bir arayüz yazmak gerekiyordu, ya da anlamsız metodları implement etmek zorunda kalıyorduk.

## Yeni Tasarım: Katmanlı Arayüz Mimarisi

### Temel Felsefe: Separation of Concerns

```
┌─────────────────────────────────────────┐
│     IObjectTracker (Minimal Core)       │  ← Zorunlu, temel tracking
└─────────────────────────────────────────┘
                    ↑
         ┌──────────┴──────────┐
         │                     │
┌─────────────────┐   ┌─────────────────┐
│ IVisualTracker  │   │ IConfigurable   │  ← Opsiyonel yetenekler
└─────────────────┘   └─────────────────┘
         │                     │
         └──────────┬──────────┘
                    ↑
         ┌──────────┴──────────┐
         │                     │
┌──────────────────┐  ┌──────────────────┐
│ KalmanIoUTracker │  │ DeepSortTracker  │  ← Concrete implementations
└──────────────────┘  └──────────────────┘
```

### Katman 1: IObjectTracker (Core Interface)

**Dosya:** [i_object_tracker.h](../i_object_tracker.h)

Bu, tüm trackerların implement etmesi **zorunlu** olan minimal arayüz:

```cpp
class IObjectTracker {
public:
    struct Detection {
        cv::Rect bbox;
        float confidence;
        int classId;
        cv::Mat featureVector; // Opsiyonel (DeepSORT için)
    };

    struct TrackedResult {
        int trackId;
        cv::Rect bbox;
        cv::Point2f velocity;
        float confidence;
        int classId;
        bool isPredicted;
        int age;
        int missedFrames;
    };

    // CORE CONTRACT
    virtual std::vector<TrackedResult> update(const std::vector<Detection>& detections) = 0;
    virtual void reset() = 0;
    virtual int getActiveTrackCount() const = 0;
    virtual int getTotalTrackCount() const = 0;
    virtual std::optional<TrackState> getTrackState(int trackId) const = 0;
};
```

**Avantajları:**
- ✅ **Algoritma-agnostik**: IoU, feature-based, hybrid trackerlar için uygun
- ✅ **Minimal**: Sadece temel tracking operasyonları
- ✅ **Standardize Detection/Result**: Her tracker aynı input/output formatını kullanır

### Katman 2: IVisualTracker (Görselleştirme)

**Dosya:** [i_visual_tracker.h](../i_visual_tracker.h)

**Opsiyonel** görselleştirme arayüzü. Trackerlar bunu implement etmek zorunda değil:

```cpp
class IVisualTracker {
public:
    struct VisualizationConfig {
        bool showBoundingBox = true;
        bool showTrackId = true;
        bool showTrajectory = false;
        bool showVelocity = false;
        bool showCenter = false;
        bool colorByState = false;
        // ... daha fazla config
    };

    virtual void draw(cv::Mat& frame, const VisualizationConfig& config) = 0;
    virtual std::vector<std::vector<cv::Point>> getTrajectories() const = 0;
    virtual std::vector<cv::Point> getTrajectory(int trackId) const = 0;
    virtual void clearTrajectories() = 0;
    virtual void setTrajectoryLength(int length) = 0;
};
```

**Avantajları:**
- ✅ **Separate Concern**: Görselleştirme tracking logic'ten ayrı
- ✅ **Opsiyonel**: Sunucu tarafı trackerlar implement etmeyebilir
- ✅ **Esnek Config**: Runtime'da görselleştirme ayarları değiştirilebilir

### Katman 3: IConfigurableTracker (Parametreler)

**Dosya:** [i_configurable_tracker.h](../i_configurable_tracker.h)

**Opsiyonel** konfigürasyon arayüzü. Her tracker kendi config struct'ını tanımlayabilir:

```cpp
template<typename ConfigType>
class IConfigurableTracker {
public:
    virtual ConfigType getConfig() const = 0;
    virtual void setConfig(const ConfigType& config) = 0;
    virtual void setParameter(const std::string& key, const std::any& value) = 0;
    virtual std::any getParameter(const std::string& key) const = 0;
    virtual bool validateConfig(const ConfigType& config) const = 0;
};
```

**Avantajları:**
- ✅ **Type-safe**: Her tracker kendi strongly-typed config'ine sahip
- ✅ **Batch veya Individual**: Config toptan veya parametre-parametre değiştirilebilir
- ✅ **Runtime tuning**: GUI/CLI üzerinden parametre ayarı mümkün

## Concrete Implementation: KalmanIoUTracker

**Dosyalar:** [kalman_iou_tracker.h](../kalman_iou_tracker.h), [kalman_iou_tracker.cpp](../kalman_iou_tracker.cpp)

### Kalman-Spesifik Config

```cpp
struct KalmanIoUConfig {
    enum class IoUType { IOU, GIOU, DIOU, CIOU, SIOU, AIOU };

    IoUType iouType = IoUType::CIOU;
    float iouThreshold = 0.3f;
    bool usePredictionInLost = true;
    int maxLostFrames = 20;
    bool removeOutOfBounds = true;
    cv::Rect cameraBounds = cv::Rect(0, 0, 640, 480);

    float processNoise = 0.01f;
    float measurementNoise = 0.5f;
    float errorCovPost = 0.1f;

    int maxTrajectoryLength = 50;
};
```

### Implementation

```cpp
class KalmanIoUTracker : public IObjectTracker,
                          public IVisualTracker,
                          public IConfigurableTracker<KalmanIoUConfig> {
public:
    // IObjectTracker
    std::vector<TrackedResult> update(const std::vector<Detection>& detections) override;
    void reset() override;
    int getActiveTrackCount() const override;

    // IVisualTracker
    void draw(cv::Mat& frame, const VisualizationConfig& config) override;
    std::vector<std::vector<cv::Point>> getTrajectories() const override;

    // IConfigurableTracker
    KalmanIoUConfig getConfig() const override;
    void setConfig(const KalmanIoUConfig& config) override;
    void setParameter(const std::string& key, const std::any& value) override;
};
```

## Impression Sınıfında Kullanım

**Dosyalar:** [impression.h](../impression.h), [impression.cpp](../impression.cpp)

Impression sınıfı artık arayüzler üzerinden çalışır:

```cpp
class Impression {
private:
    IObjectTracker *tracker = nullptr;
    IVisualTracker *visualTracker = nullptr; // Dynamic cast ile kontrol edilir
    Inference *inference = nullptr;

public:
    Impression() {
        auto* kalmanTracker = new KalmanIoUTracker();
        tracker = kalmanTracker;
        visualTracker = kalmanTracker; // Aynı nesne
    }

    cv::Mat stalkImage(const cv::Mat& frame) {
        // YOLO inference
        auto yoloDetections = inference->runInference(frame);

        // Convert to IObjectTracker::Detection
        std::vector<IObjectTracker::Detection> detections;
        for (const auto& det : yoloDetections) {
            detections.push_back({det.box, det.confidence, det.class_id});
        }

        // Update tracker
        auto trackedResults = tracker->update(detections);

        // Visualize if supported
        if (visualTracker) {
            IVisualTracker::VisualizationConfig vizConfig;
            vizConfig.showBoundingBox = true;
            vizConfig.showTrackId = true;
            vizConfig.showTrajectory = settingsParams.printTrajectory;

            visualTracker->draw(frame, vizConfig);
        }

        return frame;
    }
};
```

## Gelecekte DeepSORT Entegrasyonu

Şimdi DeepSORT'u entegre etmek **çok kolay**:

```cpp
struct DeepSortConfig {
    float maxCosineDistance = 0.2f;
    int nnBudget = 100;
    float maxIouDistance = 0.7f;
    int maxAge = 30;
    int nInit = 3;
};

class DeepSortTracker : public IObjectTracker,
                        public IVisualTracker,
                        public IConfigurableTracker<DeepSortConfig> {
public:
    // IObjectTracker implementation
    std::vector<TrackedResult> update(const std::vector<Detection>& detections) override {
        // DeepSORT-specific logic
        // Feature extraction kullan (Detection.featureVector)
        // Cosine distance hesapla
        // ...
    }

    // IVisualTracker implementation
    void draw(cv::Mat& frame, const VisualizationConfig& config) override {
        // DeepSORT-specific visualization
    }

    // IConfigurableTracker implementation
    DeepSortConfig getConfig() const override { return config; }
    void setConfig(const DeepSortConfig& cfg) override { config = cfg; }
    // ...
};
```

**Impression sınıfında sadece constructor değişir:**

```cpp
Impression::Impression() {
    auto* deepsortTracker = new DeepSortTracker();
    tracker = deepsortTracker;
    visualTracker = deepsortTracker;
}
```

## Karşılaştırma: Eski vs Yeni

| **Özellik** | **Eski ITracker** | **Yeni Tasarım** |
|------------|------------------|-----------------|
| **Spesifiklik** | KalmanIoU'ya özel | Algoritma-agnostik |
| **IoUType enum** | Arayüzde ❌ | Config struct'ta ✅ |
| **Görselleştirme** | Zorunlu metodlar | Opsiyonel arayüz |
| **Konfigürasyon** | Getter/setter'lar | Type-safe config struct |
| **DeepSORT uyumu** | Uyumsuz ❌ | Uyumlu ✅ |
| **ByteTrack uyumu** | Uyumsuz ❌ | Uyumlu ✅ |
| **Esneklik** | Düşük | Yüksek |

## Yeni Tracker Ekleme Rehberi

1. **Config struct oluştur:**
```cpp
struct MyTrackerConfig {
    // Tracker-specific parameters
};
```

2. **Tracker sınıfını implement et:**
```cpp
class MyTracker : public IObjectTracker,
                   public IVisualTracker,  // Opsiyonel
                   public IConfigurableTracker<MyTrackerConfig> {  // Opsiyonel
    // Implement required methods
};
```

3. **update() metodunu implement et:**
```cpp
std::vector<TrackedResult> update(const std::vector<Detection>& detections) override {
    // Your tracking logic
}
```

4. **Impression'da kullan:**
```cpp
auto* myTracker = new MyTracker();
tracker = myTracker;
visualTracker = dynamic_cast<IVisualTracker*>(myTracker); // Null olabilir
```

## Sonuç

✅ **Başarıyla tamamlandı:**
- 3 katmanlı, modüler arayüz tasarımı
- KalmanIoUTracker refactor edildi
- Impression sınıfı güncellendi
- Kod başarıyla derlendi
- Backward compatibility (deprecated methods)

✅ **Gelecek için hazır:**
- DeepSORT
- ByteTrack
- OC-SORT
- SORT
- Custom trackerlar

**Bu tasarım SOLID prensiplerine uygun:**
- **S**ingle Responsibility: Her arayüz tek bir concern
- **O**pen/Closed: Yeni trackerlar eklenebilir, mevcut kod değişmez
- **L**iskov Substitution: Herhangi bir IObjectTracker kullanılabilir
- **I**nterface Segregation: Trackerlar sadece ihtiyaç duydukları arayüzleri implement eder
- **D**ependency Inversion: Impression somut sınıflara değil arayüzlere bağımlı
