#ifndef KALMAN_IOU_TRACKER_H
#define KALMAN_IOU_TRACKER_H

#include "../helpers/HungarianAlgorithm.h"
#include "../../i_configurable_tracker.h"
#include "../../i_object_tracker.h"
#include "../../i_tracker_telemetry.h"
#include "../../i_visual_tracker.h"
#include <algorithm>
#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

/**
 * @file kalman_iou_tracker.h
 * @brief Kalman Filter + IoU-based object tracker
 * @author Halil Erden
 * @date 17.10.2025
 *
 * This tracker combines Kalman filtering for motion prediction with
 * various IoU metrics for detection-to-track association. It supports
 * multiple IoU variants (IOU, GIOU, DIOU, CIOU, SIOU, AIOU) and uses
 * the Hungarian algorithm for optimal matching.
 */

#include "kalman_iou_config.h"

/**
 * @brief Internal tracked object representation for 8-state tracker
 * This is Kalman-specific and not exposed through IObjectTracker interface
 */
namespace KalmanIoU8State {
struct TrackedObject {
    enum State {
        TRACKING,   ///< Active tracking
        LOST        ///< Lost (no detection)
    };

    int id;                                    ///< Unique track ID
    cv::Rect boundingBox;                     ///< Current bounding box
    cv::KalmanFilter kalmanFilter;            ///< Kalman filter instance
    int missedFrames = 0;                     ///< Consecutive missed detections
    int age = 0;                              ///< Total frames tracked
    cv::Point lastPredictedCenter;            ///< Last predicted center
    cv::Point lastActualCenter;               ///< Last measured center
    std::deque<cv::Point> trajectory;         ///< Path history
    cv::Scalar color;                         ///< Visualization color
    cv::Point2f velocity;                     ///< vx, vy (pixels/frame)
    cv::Point2f sizeVelocity;                 ///< vw, vh (size change)
    State state = TRACKING;                   ///< Current state
    float detectionConfidence = 1.0f;         ///< Last detection confidence (for telemetry)
};
} // namespace KalmanIoU8State

/**
 * @brief Kalman + IoU based tracker implementation
 *
 * Implements all three optional interfaces:
 * - IObjectTracker: Core tracking functionality
 * - IVisualTracker: Drawing and trajectory visualization
 * - IConfigurableTracker: Runtime configuration
 */
class KalmanIoUTracker : public IObjectTracker,
                          public IVisualTracker,
                          public IConfigurableTracker<KalmanIoUConfig> {
public:
    KalmanIoUTracker();
    explicit KalmanIoUTracker(const KalmanIoUConfig& config);

    // ========== IObjectTracker Implementation ==========
    std::vector<TrackedResult> update(const std::vector<Detection>& detections) override;
    void reset() override;
    int getActiveTrackCount() const override;
    int getTotalTrackCount() const override;
    std::optional<TrackState> getTrackState(int trackId) const override;

    // ========== IVisualTracker Implementation ==========
    void draw(cv::Mat& frame, const VisualizationConfig& config) override;
    std::vector<std::vector<cv::Point>> getTrajectories() const override;
    std::vector<cv::Point> getTrajectory(int trackId) const override;
    void clearTrajectories() override;
    void setTrajectoryLength(int length) override;

    // ========== IConfigurableTracker Implementation ==========
    KalmanIoUConfig getConfig() const override;
    void setConfig(const KalmanIoUConfig& config) override;
    void setParameter(const std::string& key, const std::any& value) override;
    std::any getParameter(const std::string& key) const override;
    bool validateConfig(const KalmanIoUConfig& config) const override;

    // ========== Telemetry ==========
    void setTelemetry(const std::shared_ptr<ITrackerTelemetry>& telemetryLogger);
    void clearTelemetry();
    void setFrameSizeHint(const cv::Size& sourceSize, const cv::Size& processedSize);
    void clearFrameSizeHint();

    // ========== Legacy Methods (Deprecated - for backward compatibility) ==========
    [[deprecated("Use update() instead")]]
    std::vector<std::pair<int, cv::Rect>> processDetections(const std::vector<cv::Rect>& detections);

    [[deprecated("Use update() instead")]]
    std::vector<std::pair<int, cv::Rect>> processDetectionsWithHungarian(const std::vector<cv::Rect>& detections);

    [[deprecated("Use update() and access internal trackers differently")]]
    std::vector<KalmanIoU8State::TrackedObject> processDetectionsWithHungarianTrackers(const std::vector<cv::Rect>& detections);

private:
    // ========== Internal State ==========
    std::vector<KalmanIoU8State::TrackedObject> trackers;
    int nextId;
    KalmanIoUConfig config;
    std::shared_ptr<ITrackerTelemetry> telemetry;
    struct FrameSizeInfo {
        cv::Size source;
        cv::Size processed;
    };
    std::optional<FrameSizeInfo> frameSizeHint;
    int frameIndexCounter = 0;

    // ========== Internal Helper Methods ==========
    cv::Scalar generateRandomColor();
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    cv::KalmanFilter createKalmanFilter();
    void updateTrackersWithHungarian(const std::vector<cv::Rect>& detections, const std::vector<float>& confidences = {});
    void updateTrackers(const std::vector<cv::Rect>& detections, std::vector<bool>& matchedDetections);
    void addNewTrackers(const std::vector<cv::Rect>& detections, const std::vector<bool>& matchedDetections, const std::vector<float>& confidences = {});
    void removeLostTrackers();
    double computeIoU(KalmanIoUConfig::IoUType type, const cv::Rect& boxA, const cv::Rect& boxB);
    double calculateGIoU(const cv::Rect& boxA, const cv::Rect& boxB);
    double calculateDIoU(const cv::Rect& boxA, const cv::Rect& boxB);
    double calculateCIoU(const cv::Rect& boxA, const cv::Rect& boxB);
    double calculateSoftIoU(const cv::Rect& boxA, const cv::Rect& boxB);
    double calculateAlphaIoU(const cv::Rect& boxA, const cv::Rect& boxB);
    uint64_t timeSinceEpochMillisec();

    // Helper to convert internal state to IObjectTracker state
    TrackState toTrackState(KalmanIoU8State::TrackedObject::State state) const;

    // Find tracker by ID
    const KalmanIoU8State::TrackedObject* findTracker(int trackId) const;
};

#endif // KALMAN_IOU_TRACKER_H
