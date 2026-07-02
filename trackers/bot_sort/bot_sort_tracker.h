#ifndef BOT_SORT_TRACKER_H
#define BOT_SORT_TRACKER_H

#include "../helpers/HungarianAlgorithm.h"
#include "../../i_configurable_tracker.h"
#include "../../i_object_tracker.h"
#include "../../i_tracker_telemetry.h"
#include "../../i_visual_tracker.h"
#include "bot_sort_config.h"
#include "cmc_module.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

namespace BotSortNS {

struct BotSortTrack {
    enum State { TRACKING, LOST };

    int id;
    cv::Rect boundingBox;
    cv::KalmanFilter kalmanFilter;
    int missedFrames  = 0;
    int age           = 0;
    cv::Point lastPredictedCenter;
    cv::Point lastActualCenter;
    std::deque<cv::Point> trajectory;
    cv::Scalar color;
    cv::Point2f velocity;
    cv::Point2f sizeVelocity;
    State state                  = TRACKING;
    float detectionConfidence    = 1.0f;
    cv::Mat featureEmbedding;
    float lastReidSimilarity     = -1.f;
    cv::Rect cmcPredictedBox;   // CMC-warped prediction used for matching
};

} // namespace BotSortNS

/**
 * @brief BoT-SORT: ByteTrack + Camera Motion Compensation (CMC).
 *
 * Extends the ByteTrack two-stage association algorithm with a CMC step
 * that warps all Kalman-predicted states to compensate for global camera
 * motion before Hungarian matching. This improves track continuity on
 * sequences with moving cameras (pan, tilt, zoom).
 *
 * Usage:
 * @code
 *   BotSortTracker tracker(config);
 *   while (reader.nextFrame(frame)) {
 *       tracker.setFrame(frame);   // must be called before update()
 *       auto results = tracker.update(detections);
 *   }
 * @endcode
 */
class BotSortTracker : public IObjectTracker,
                       public IVisualTracker,
                       public IConfigurableTracker<BotSortConfig> {
public:
    BotSortTracker();
    explicit BotSortTracker(const BotSortConfig& config);

    // ========== IObjectTracker ==========
    std::vector<TrackedResult> update(const std::vector<Detection>& detections) override;
    void reset() override;
    int getActiveTrackCount() const override;
    int getTotalTrackCount() const override;
    std::optional<TrackState> getTrackState(int trackId) const override;

    // ========== IVisualTracker ==========
    void draw(cv::Mat& frame, const VisualizationConfig& config) override;
    std::vector<std::vector<cv::Point>> getTrajectories() const override;
    std::vector<cv::Point> getTrajectory(int trackId) const override;
    void clearTrajectories() override;
    void setTrajectoryLength(int length) override;

    // ========== IConfigurableTracker<BotSortConfig> ==========
    BotSortConfig getConfig() const override;
    void setConfig(const BotSortConfig& config) override;
    void setParameter(const std::string& key, const std::any& value) override;
    std::any getParameter(const std::string& key) const override;
    bool validateConfig(const BotSortConfig& config) const override;

    // ========== Telemetry (same pattern as KalmanIoUByteTrack) ==========
    void setTelemetry(const std::shared_ptr<ITrackerTelemetry>& telemetryLogger);
    void clearTelemetry();
    void setFrameSizeHint(const cv::Size& sourceSize, const cv::Size& processedSize);
    void clearFrameSizeHint();

    /**
     * @brief Provide the current video frame for Camera Motion Compensation.
     *
     * Must be called BEFORE update() on each frame. Internally converts the
     * frame to grayscale and stores it. After update(), the current frame
     * becomes the previous frame for the next call.
     *
     * If setFrame() is not called, CMC is skipped for that frame and the
     * tracker behaves identically to ByteTrack.
     */
    void setFrame(const cv::Mat& frame);

private:
    std::vector<BotSortNS::BotSortTrack> tracks_;
    int nextId_            = 0;
    int frameIndexCounter_ = 0;
    BotSortConfig config_;
    CmcModule cmc_;

    cv::Mat prevGray_;  // Previous frame grayscale (set at end of update())
    cv::Mat currGray_;  // Current frame grayscale (set by setFrame())

    std::shared_ptr<ITrackerTelemetry> telemetry_;
    struct FrameSizeInfo { cv::Size source; cv::Size processed; };
    std::optional<FrameSizeInfo> frameSizeHint_;

    // ---- Helpers ----
    cv::KalmanFilter createKalmanFilter();

    // Predict all tracks, apply CMC warp to statePre, fill cmcPredictedBox
    std::vector<cv::Rect> predictAllAndApplyCMC(const cv::Mat& warp);

    struct AssociationResult {
        std::vector<int> unmatchedTrackIdx;
        std::vector<bool> matchedDetections;
    };

    // Stage 1: match high-confidence detections to TRACKING tracks
    AssociationResult associateHighConf(const std::vector<cv::Rect>& dets,
                                        const std::vector<float>& confs,
                                        const std::vector<cv::Mat>& feats);

    // Stage 2: match low-confidence detections to Stage-1-unmatched TRACKING tracks
    AssociationResult matchTrackingWithLowConf(const std::vector<int>& trackIdx,
                                               const std::vector<cv::Rect>& dets,
                                               const std::vector<float>& confs,
                                               const std::vector<cv::Mat>& feats);

    void markTrackingTracksLost(const std::vector<int>& trackIdx);
    void ageExistingLostTracks(const std::vector<int>& lostIdx);

    // Stage 3/4: revive LOST tracks with unmatched detections (IoU + ReID,
    // ReID-only fallback gated by reidAppearanceThresh). Returns per-detection
    // matched mask.
    std::vector<bool> reviveLostTracks(const std::vector<cv::Rect>& dets,
                                       const std::vector<float>& confs,
                                       const std::vector<cv::Mat>& feats);

    // Initialize new tracks from unmatched high-confidence detections
    void initNewTracks(const std::vector<cv::Rect>& dets,
                       const std::vector<bool>& matched,
                       const std::vector<float>& confs,
                       const std::vector<cv::Mat>& feats);

    // Remove tracks that exceeded maxLostFrames or went out of bounds
    void pruneLostTracks();

    // Kalman correct() with adaptive measurement noise
    void correctTrack(BotSortNS::BotSortTrack& t,
                      const cv::Rect& det,
                      float conf,
                      const cv::Mat& feat);

    double computeIoU(BotSortConfig::IoUType type,
                      const cv::Rect& a, const cv::Rect& b);
    double calcIoU(const cv::Rect& a, const cv::Rect& b);
    double calcGIoU(const cv::Rect& a, const cv::Rect& b);
    double calcDIoU(const cv::Rect& a, const cv::Rect& b);
    double calcCIoU(const cv::Rect& a, const cv::Rect& b);
    double calcSoftIoU(const cv::Rect& a, const cv::Rect& b);
    double calcAlphaIoU(const cv::Rect& a, const cv::Rect& b);

    static double cosineDistance(const cv::Mat& a, const cv::Mat& b);

    cv::Scalar randomColor();
    TrackState mapState(BotSortNS::BotSortTrack::State s) const;
    const BotSortNS::BotSortTrack* findTrack(int trackId) const;

    uint64_t timeSinceEpochMs() const;
};

#endif // BOT_SORT_TRACKER_H
