#include "cmc_module.h"

#include <algorithm>
#include <cmath>
#include <iostream>
// calib3d.hpp provides estimateAffinePartial2D and cv::RANSAC in OpenCV 5
#include <opencv2/calib3d.hpp>

CmcModule::CmcModule(const BotSortConfig& cfg) : cfg_(cfg) {}

void CmcModule::setConfig(const BotSortConfig& cfg) {
    cfg_ = cfg;
}

cv::Mat CmcModule::estimate(const cv::Mat& prevGray, const cv::Mat& currGray,
                             const std::vector<cv::Rect>& exclusionRects) {
    if (prevGray.empty() || currGray.empty()) {
        return cv::Mat();
    }

    cv::Mat prevForCmc = prevGray;
    cv::Mat currForCmc = currGray;
    std::vector<cv::Rect> scaledExclusions;
    const float scale = std::clamp(cfg_.cmcDownscale, 0.1f, 1.0f);
    if (scale < 0.999f) {
        cv::resize(prevGray, prevForCmc, cv::Size(), scale, scale, cv::INTER_AREA);
        cv::resize(currGray, currForCmc, cv::Size(), scale, scale, cv::INTER_AREA);
        scaledExclusions.reserve(exclusionRects.size());
        for (const auto& r : exclusionRects) {
            scaledExclusions.emplace_back(
                cv::Rect(
                    static_cast<int>(std::round(r.x * scale)),
                    static_cast<int>(std::round(r.y * scale)),
                    static_cast<int>(std::round(r.width * scale)),
                    static_cast<int>(std::round(r.height * scale))));
        }
    }
    const auto& maskRects = (scale < 0.999f) ? scaledExclusions : exclusionRects;

    // Build exclusion mask: white everywhere, black over tracked object regions.
    // This prevents pedestrian motion from corrupting the background-based CMC estimate.
    cv::Mat mask;
    if (cfg_.cmcMaskTrackedObjects && !maskRects.empty()) {
        mask = cv::Mat(prevForCmc.size(), CV_8U, cv::Scalar(255));
        for (const auto& r : maskRects) {
            cv::Rect safe = r & cv::Rect(0, 0, prevForCmc.cols, prevForCmc.rows);
            if (safe.area() > 0) {
                mask(safe) = cv::Scalar(0);
            }
        }
    }

    cv::Mat warp;
    switch (cfg_.cmcMethod) {
        case BotSortConfig::CmcMethod::SPARSE_LK:
            warp = estimateSparseLK(prevForCmc, currForCmc, mask);
            break;
        case BotSortConfig::CmcMethod::ECC:
            warp = estimateECC(prevForCmc, currForCmc);
            break;
        case BotSortConfig::CmcMethod::NONE:
        default:
            return cv::Mat();
    }

    if (!warp.empty() && scale < 0.999f) {
        warp.at<double>(0, 2) /= scale;
        warp.at<double>(1, 2) /= scale;
    }

    return applyIdentityGate(warp);
}

cv::Mat CmcModule::estimateSparseLK(const cv::Mat& prev, const cv::Mat& curr,
                                     const cv::Mat& mask) {
    std::vector<cv::Point2f> prevPts;
    cv::goodFeaturesToTrack(
        prev, prevPts,
        cfg_.cmcMaxCorners,
        cfg_.cmcQualityLevel,
        cfg_.cmcMinDistance,
        mask,              // exclusion mask — was always cv::Mat() before
        cfg_.cmcBlockSize);

    if (static_cast<int>(prevPts.size()) < 4) {
        return cv::Mat();
    }

    std::vector<cv::Point2f> currPts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        prev, curr,
        prevPts, currPts,
        status, err,
        cv::Size(21, 21),
        cfg_.cmcLkMaxLevel);

    std::vector<cv::Point2f> validPrev, validCurr;
    validPrev.reserve(prevPts.size());
    validCurr.reserve(currPts.size());
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            validPrev.push_back(prevPts[i]);
            validCurr.push_back(currPts[i]);
        }
    }

    if (static_cast<int>(validPrev.size()) < 4) {
        return cv::Mat();
    }

    cv::Mat inlierMask;
    cv::Mat warp = cv::estimateAffinePartial2D(
        validPrev, validCurr,
        inlierMask,
        cv::RANSAC,
        static_cast<double>(cfg_.cmcRansacThresh),
        static_cast<size_t>(cfg_.cmcRansacMaxIter),
        static_cast<double>(cfg_.cmcConfidence));

    if (warp.empty()) {
        return cv::Mat();
    }

    // Reject warp if RANSAC inlier ratio is too low — unreliable estimate.
    int inlierCount = cv::countNonZero(inlierMask);
    float inlierRatio = static_cast<float>(inlierCount) /
                        static_cast<float>(validPrev.size());
    if (inlierRatio < cfg_.cmcMinInlierRatio) {
        return cv::Mat();
    }

    cv::Mat warp64;
    warp.convertTo(warp64, CV_64F);
    return warp64;
}

cv::Mat CmcModule::estimateECC(const cv::Mat& prev, const cv::Mat& curr) {
    cv::Mat warpMatrix = cv::Mat::eye(2, 3, CV_32F);

    const cv::TermCriteria criteria(
        cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
        cfg_.eccMaxIter,
        cfg_.eccEps);

    try {
        cv::findTransformECC(prev, curr, warpMatrix, cv::MOTION_EUCLIDEAN, criteria);
    } catch (const cv::Exception&) {
        return cv::Mat();
    }

    cv::Mat warp64;
    warpMatrix.convertTo(warp64, CV_64F);
    return warp64;
}

cv::Mat CmcModule::applyIdentityGate(const cv::Mat& warp) const {
    if (warp.empty()) return warp;

    // For estimateAffinePartial2D (4-DOF: scale+rotation+translation):
    //   [a, -b, tx]      a = s*cos(theta), b = s*sin(theta)
    //   [b,  a, ty]
    const double a  = warp.at<double>(0, 0);
    const double b  = warp.at<double>(1, 0);
    const double tx = warp.at<double>(0, 2);
    const double ty = warp.at<double>(1, 2);

    const double scale     = std::sqrt(a * a + b * b);
    const double angleDeg  = std::atan2(b, a) * (180.0 / M_PI);

    const bool transOk  = std::abs(tx) < cfg_.cmcIdentityTransPx &&
                          std::abs(ty) < cfg_.cmcIdentityTransPx;
    const bool scaleOk  = std::abs(scale - 1.0) < cfg_.cmcIdentityScaleEps;
    const bool angleOk  = std::abs(angleDeg) < cfg_.cmcIdentityAngleDeg;

    if (transOk && scaleOk && angleOk) {
        return cv::Mat();  // near-identity — static camera, skip warp
    }

    return warp;
}
