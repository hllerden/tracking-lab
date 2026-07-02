#ifndef BOT_SORT_CONFIG_H
#define BOT_SORT_CONFIG_H

#include <opencv2/opencv.hpp>

struct BotSortConfig {
    // ---- IoU matching (mirrors KalmanIoUConfig fields) ----
    enum class IoUType { IOU, GIOU, DIOU, CIOU, SIOU, AIOU };
    IoUType iouType             = IoUType::CIOU;
    float   iouThreshold        = 0.25f;
    bool    usePredictionInLost = true;
    int     maxLostFrames       = 30;
    bool    removeOutOfBounds   = true;
    cv::Rect cameraBounds       = cv::Rect(0, 0, 640, 480);

    // Track state management
    int lostStateThreshold  = 2;

    // Kalman filter parameters
    float processNoise     = 0.01f;
    float measurementNoise = 0.5f;
    float errorCovPost     = 0.1f;

    // Trajectory
    int maxTrajectoryLength = 50;

    // IoU variant tuning
    float softIoUSlack     = 0.5f;
    float alphaIoUExponent = 0.6f;

    // Bounding box size constraints
    float minBBoxWidth       = 10.0f;
    float minBBoxHeight      = 20.0f;
    float maxBBoxWidth       = 500.0f;
    float maxBBoxHeight      = 600.0f;
    float sizeVelocityClamp  = 0.10f;

    // ByteTrack two-stage thresholds
    float highConfThreshold   = 0.6f;
    float lowConfFloor        = 0.1f;
    float lowConfIouThreshold = 0.15f;

    // ReID appearance matching
    bool  useReId         = false;
    float reidAlpha       = 0.65f;
    float featureEmaDecay = 0.9f;

    // ---- BoT-SORT CMC parameters ----
    enum class CmcMethod { NONE, SPARSE_LK, ECC };
    CmcMethod cmcMethod        = CmcMethod::SPARSE_LK;

    // SPARSE_LK (goodFeaturesToTrack + calcOpticalFlowPyrLK + estimateAffinePartial2D)
    int   cmcMaxCorners      = 300;
    float cmcQualityLevel    = 0.01f;
    float cmcMinDistance     = 8.0f;
    int   cmcBlockSize       = 5;
    int   cmcLkMaxLevel      = 2;
    int   cmcRansacMaxIter   = 100;
    float cmcRansacThresh    = 3.0f;
    float cmcConfidence      = 0.99f;
    float cmcDownscale       = 0.5f;   // estimate CMC on smaller frames, scale translation back

    // ECC (findTransformECC)
    int    eccMaxIter = 50;
    double eccEps     = 1e-4;

    // CMC quality gates
    bool  cmcMaskTrackedObjects  = true;   // mask current track boxes before goodFeaturesToTrack
    float cmcMinInlierRatio      = 0.30f;  // reject warp if RANSAC inliers/total < this
    float cmcIdentityTransPx     = 1.5f;   // skip warp if |tx|,|ty| both below this (pixels)
    float cmcIdentityScaleEps    = 0.005f; // skip warp if |scale-1| below this
    float cmcIdentityAngleDeg    = 0.30f;  // skip warp if |rotation| below this (degrees)

    BotSortConfig() = default;
};

#endif // BOT_SORT_CONFIG_H
