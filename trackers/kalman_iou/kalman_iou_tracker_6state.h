#ifndef KALMAN_IOU_TRACKER_6State_H
#define KALMAN_IOU_TRACKER_6State_H

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
 * @file kalman_iou_tracker_6state.h
 * @brief Kalman Filter + IoU-based object tracker (6-state)
 * @author Halil Erden
 * @date 17.10.2025
 *
 * @deprecated This tracker is superseded by KalmanIoUByteTrack which implements
 *             the full ByteTrack two-stage association pipeline with optional ReID
 *             support. KalmanIoUTracker6State will not receive further development.
 *             Existing call sites are preserved for backward compatibility.
 */

/**
 * @brief Configuration for KalmanIoUTracker
 */
#ifndef KALMAN_IOU_CONFIG_DEFINED
#define KALMAN_IOU_CONFIG_DEFINED
struct KalmanIoUConfig {
    /**
     * @brief IoU calculation method
     */
    enum class IoUType {
        IOU,    ///< Standard Intersection over Union
        GIOU,   ///< Generalized IoU (better for low overlap)
        DIOU,   ///< Distance IoU (penalizes center distance)
        CIOU,   ///< Complete IoU (includes aspect ratio)
        SIOU,   ///< Soft IoU (IoU with slack weighting)
        AIOU    ///< Alpha IoU (tunable exponent)
    };

    IoUType iouType = IoUType::CIOU;          ///< IoU calculation method
    float iouThreshold = 0.2f;                ///< Minimum IoU for matching (lowered from 0.3 to match original code)
    bool usePredictionInLost = true;          ///< Use predictions for lost tracks
    int maxLostFrames = 20;                   ///< Max frames without detection before deletion
    bool removeOutOfBounds = true;            ///< Remove tracks outside frame
    cv::Rect cameraBounds = cv::Rect(0, 0, 640, 480);  ///< Camera frame bounds

    // Kalman filter parameters
    float processNoise = 0.01f;               ///< Process noise covariance
    float measurementNoise = 0.5f;            ///< Measurement noise covariance
    float errorCovPost = 0.1f;                ///< Initial error covariance

    // Trajectory parameters
    int maxTrajectoryLength = 50;             ///< Maximum trajectory points to keep

    // IoU variants tuning
    float softIoUSlack = 0.5f;                ///< Slack factor (<1 boosts moderate IoUs)
    float alphaIoUExponent = 0.6f;            ///< Exponent for Alpha IoU (>0)

    // Track state management
    int lostStateThreshold = 2;               ///< Missed frames before switching to LOST state

    // Bounding box size constraints for prediction clamping
    float minBBoxWidth = 10.0f;               ///< Minimum predicted bbox width (pixels)
    float minBBoxHeight = 20.0f;              ///< Minimum predicted bbox height (pixels)
    float maxBBoxWidth = 500.0f;              ///< Maximum predicted bbox width (pixels)
    float maxBBoxHeight = 600.0f;             ///< Maximum predicted bbox height (pixels)
    float sizeVelocityClamp = 0.10f;          ///< Max size velocity as fraction of bbox dimension

    // ByteTrack two-stage association thresholds (ignored by 6-state and 8-state trackers)
    float highConfThreshold = 0.6f;           ///< High-confidence detection threshold (stage 1)
    float lowConfFloor = 0.1f;                ///< Low-confidence detection floor (stage 2)
    float lowConfIouThreshold = 0.15f;        ///< IoU threshold for low-confidence track revival

    // ReID appearance matching (ByteTrack only)
    bool  useReId         = false;            ///< Enable hybrid IoU + cosine distance cost
    float reidAlpha       = 0.65f;            ///< Weight of IoU term (1-reidAlpha → appearance term)
    float featureEmaDecay = 0.9f;             ///< EMA momentum for track appearance update

    KalmanIoUConfig() = default;
};
#endif // KALMAN_IOU_CONFIG_DEFINED

/**
 * @brief Internal tracked object representation for 6-state tracker
 * This is Kalman-specific and not exposed through IObjectTracker interface
 */
namespace [[deprecated("KalmanIoU6State is superseded by KalmanIoUByteTrackNS")]] KalmanIoU6State {
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
    cv::Point velocity;                       ///< Current velocity
    State state = TRACKING;                   ///< Current state
    float detectionConfidence = 1.0f;         ///< Last detection confidence (for telemetry)
};
} // namespace KalmanIoU6State

/**
 * @brief Kalman + IoU based tracker implementation
 *
 * Implements all three optional interfaces:
 * - IObjectTracker: Core tracking functionality
 * - IVisualTracker: Drawing and trajectory visualization
 * - IConfigurableTracker: Runtime configuration
 */
class [[deprecated("KalmanIoUTracker6State is superseded by KalmanIoUByteTrack; do not use in new code")]]
KalmanIoUTracker6State : public IObjectTracker,
                         public IVisualTracker,
                         public IConfigurableTracker<KalmanIoUConfig> {
public:
    KalmanIoUTracker6State();
    explicit KalmanIoUTracker6State(const KalmanIoUConfig& config);

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
    std::vector<KalmanIoU6State::TrackedObject> processDetectionsWithHungarianTrackers(const std::vector<cv::Rect>& detections);

private:
    // ========== Internal State ==========
    std::vector<KalmanIoU6State::TrackedObject> trackers;
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
    TrackState toTrackState(KalmanIoU6State::TrackedObject::State state) const;

    // Find tracker by ID
    const KalmanIoU6State::TrackedObject* findTracker(int trackId) const;
};

#endif // KALMAN_IOU_TRACKER_6State_H
