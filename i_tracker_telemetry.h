#ifndef I_TRACKER_TELEMETRY_H
#define I_TRACKER_TELEMETRY_H

#include "i_object_tracker.h"
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

/**
 * @file i_tracker_telemetry.h
 * @brief Tracker bağımsız telemetri arayüzü
 *
 * Trackerdan bağımsız olarak metrik toplamayı ve farklı çıktı formatlarına
 * yönlendirmeyi sağlar. Her kare `update` çağrısı ile ilişkilidir.
 */
class ITrackerTelemetry {
public:
    struct FrameMetadata {
        int frameIndex = 0;                  ///< Sıfır tabanlı kare index'i
        std::optional<cv::Size> sourceSize;  ///< Orijinal kare boyutu
        std::optional<cv::Size> processedSize; ///< İşlenen (ör. resize) kare boyutu
        std::optional<double> timestamp;     ///< Saniye cinsinden zaman damgası (varsa)
    };

    virtual ~ITrackerTelemetry() = default;

    /**
     * @brief Kare işlemesi başlamadan çağrılır.
     * @param metadata Kareye ait meta bilgiler.
     */
    virtual void onFrameStart(const FrameMetadata& metadata) = 0;

    /**
     * @brief Her iz sonucu için çağrılır.
     * @param frameIndex İlgili karenin index'i.
     * @param result İz sonucu.
     */
    virtual void onTrackResult(int frameIndex, const IObjectTracker::TrackedResult& result) = 0;

    /**
     * @brief Kare tamamlandığında çağrılır.
     * @param frameIndex İlgili karenin index'i.
     */
    virtual void onFrameEnd(int frameIndex) = 0;

    /**
     * @brief Biriktirilmiş verileri kalıcı hale getirir.
     */
    virtual void flush() = 0;
};

#endif // I_TRACKER_TELEMETRY_H
