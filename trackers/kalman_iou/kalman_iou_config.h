#ifndef KALMAN_IOU_CONFIG_H
#define KALMAN_IOU_CONFIG_H

#include <opencv2/opencv.hpp>

/**
 * @brief Shared configuration for the KalmanIoU tracker family
 * (KalmanIoUTracker, KalmanIoUTracker6State, KalmanIoUByteTrack).
 *
 * Single definition to guarantee an identical layout in every translation
 * unit. Per-tracker defaults (e.g. iouThreshold, maxLostFrames) are applied
 * by TrackerFactory; the values here are only the common fallback.
 */
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
    float iouThreshold = 0.25f;               ///< Minimum IoU for matching
    bool usePredictionInLost = true;          ///< Use predictions for lost tracks
    int maxLostFrames = 30;                   ///< Max frames without detection before deletion
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
    float reidAppearanceThresh = 0.25f;       ///< LOST recovery: max cosine distance for ReID-only match

    KalmanIoUConfig() = default;
};

#endif // KALMAN_IOU_CONFIG_H
