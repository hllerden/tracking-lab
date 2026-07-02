#ifndef CMC_MODULE_H
#define CMC_MODULE_H

#include <opencv2/opencv.hpp>
#include "bot_sort_config.h"

/**
 * @brief Camera Motion Compensation module for BoT-SORT.
 *
 * Estimates the global camera warp (affine, 4-DOF: rotation + uniform scale +
 * translation) between consecutive frames. The returned 2x3 CV_64F matrix can
 * be applied to Kalman-predicted bounding box states before Hungarian matching
 * to compensate for camera pan/tilt/zoom.
 *
 * Returns an empty Mat when motion estimation fails or NONE mode is selected;
 * callers must check warp.empty() and skip CMC in that case.
 */
class CmcModule {
public:
    explicit CmcModule(const BotSortConfig& cfg = BotSortConfig{});

    void setConfig(const BotSortConfig& cfg);

    /**
     * @brief Estimate affine warp from prevGray -> currGray.
     *
     * @param exclusionRects Bounding boxes of tracked objects to mask out before
     *                       feature detection. Prevents pedestrian motion from
     *                       corrupting the background-based camera motion estimate.
     * @return 2x3 CV_64F affine matrix, or cv::Mat() on failure / NONE mode /
     *         near-identity motion (static camera).
     */
    cv::Mat estimate(const cv::Mat& prevGray, const cv::Mat& currGray,
                     const std::vector<cv::Rect>& exclusionRects = {});

private:
    BotSortConfig cfg_;

    cv::Mat estimateSparseLK(const cv::Mat& prev, const cv::Mat& curr,
                             const cv::Mat& mask);
    cv::Mat estimateECC(const cv::Mat& prev, const cv::Mat& curr);

    // Returns empty Mat if warp is within identity thresholds (static camera noise gate)
    cv::Mat applyIdentityGate(const cv::Mat& warp) const;
};

#endif // CMC_MODULE_H
