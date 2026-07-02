#include "bot_sort_tracker.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using BotSortTrack = BotSortNS::BotSortTrack;

// ========== Constructor ==========

BotSortTracker::BotSortTracker() : cmc_(BotSortConfig{}) {}

BotSortTracker::BotSortTracker(const BotSortConfig& config)
    : config_(config), cmc_(config) {}

// ========== Frame injection for CMC ==========

void BotSortTracker::setFrame(const cv::Mat& frame) {
    if (frame.empty()) return;
    if (config_.cmcMethod == BotSortConfig::CmcMethod::NONE) {
        currGray_ = cv::Mat();
        return;
    }
    if (frame.channels() == 1) {
        currGray_ = frame.clone();
    } else {
        cv::cvtColor(frame, currGray_, cv::COLOR_BGR2GRAY);
    }
}

// ========== Kalman filter factory ==========

cv::KalmanFilter BotSortTracker::createKalmanFilter() {
    cv::KalmanFilter kf(8, 4, 0);

    const float dt = 1.0f;
    kf.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, dt, 0,  0,  0,
        0, 1, 0, 0, 0, dt,  0,  0,
        0, 0, 1, 0, 0,  0, dt,  0,
        0, 0, 0, 1, 0,  0,  0, dt,
        0, 0, 0, 0, 1,  0,  0,  0,
        0, 0, 0, 0, 0,  1,  0,  0,
        0, 0, 0, 0, 0,  0,  1,  0,
        0, 0, 0, 0, 0,  0,  0,  1
    );

    kf.measurementMatrix = (cv::Mat_<float>(4, 8) <<
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0
    );

    kf.processNoiseCov = cv::Mat::eye(8, 8, CV_32F) * 1e-4f;
    kf.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2f;

    kf.errorCovPost = cv::Mat::eye(8, 8, CV_32F);
    kf.errorCovPost.at<float>(4, 4) = 10.0f;
    kf.errorCovPost.at<float>(5, 5) = 10.0f;
    kf.errorCovPost.at<float>(6, 6) = 10.0f;
    kf.errorCovPost.at<float>(7, 7) = 10.0f;

    return kf;
}

// ========== Predict + CMC warp ==========

std::vector<cv::Rect> BotSortTracker::predictAllAndApplyCMC(const cv::Mat& warp) {
    std::vector<cv::Rect> predictedBoxes;
    predictedBoxes.reserve(tracks_.size());

    // Extract warp components (CV_64F 2x3)
    bool applyWarp = !warp.empty();
    double W00 = 1.0, W01 = 0.0, W02 = 0.0;
    double W10 = 0.0, W11 = 1.0, W12 = 0.0;
    double sx = 1.0, sy = 1.0;
    if (applyWarp) {
        W00 = warp.at<double>(0, 0); W01 = warp.at<double>(0, 1); W02 = warp.at<double>(0, 2);
        W10 = warp.at<double>(1, 0); W11 = warp.at<double>(1, 1); W12 = warp.at<double>(1, 2);
        sx = std::sqrt(W00 * W00 + W10 * W10);
        sy = std::sqrt(W01 * W01 + W11 * W11);
        if (sx < 0.5 || sx > 2.0) applyWarp = false;  // sanity check for degenerate warp
        if (sy < 0.5 || sy > 2.0) applyWarp = false;
    }

    for (auto& t : tracks_) {
        // Adaptive process noise (BoT-SORT/DeepSORT approach)
        float h = static_cast<float>(t.boundingBox.height);
        if (h < 1.0f) h = 1.0f;

        const float sw_pos = 1.0f / 40.0f;
        const float sw_vel = 1.0f / 320.0f;

        t.kalmanFilter.processNoiseCov.at<float>(0, 0) = (sw_pos * h) * (sw_pos * h);
        t.kalmanFilter.processNoiseCov.at<float>(1, 1) = (sw_pos * h) * (sw_pos * h);
        t.kalmanFilter.processNoiseCov.at<float>(2, 2) = 1e-4f;
        t.kalmanFilter.processNoiseCov.at<float>(3, 3) = (sw_pos * h) * (sw_pos * h);
        t.kalmanFilter.processNoiseCov.at<float>(4, 4) = (sw_vel * h) * (sw_vel * h);
        t.kalmanFilter.processNoiseCov.at<float>(5, 5) = (sw_vel * h) * (sw_vel * h);
        t.kalmanFilter.processNoiseCov.at<float>(6, 6) = 1e-10f;
        t.kalmanFilter.processNoiseCov.at<float>(7, 7) = (sw_vel * h) * (sw_vel * h);

        cv::Mat prediction = t.kalmanFilter.predict();

        float px  = prediction.at<float>(0);
        float py  = prediction.at<float>(1);
        float pw  = prediction.at<float>(2);
        float ph  = prediction.at<float>(3);
        float pvx = prediction.at<float>(4);
        float pvy = prediction.at<float>(5);
        float pvw = prediction.at<float>(6);
        float pvh = prediction.at<float>(7);

        // Clamp size velocity
        float max_vw = config_.sizeVelocityClamp * std::max(pw, config_.minBBoxWidth);
        float max_vh = config_.sizeVelocityClamp * std::max(ph, config_.minBBoxHeight);
        pvw = std::max(-max_vw, std::min(pvw, max_vw));
        pvh = std::max(-max_vh, std::min(pvh, max_vh));

        // Clamp bbox size
        pw = std::max(config_.minBBoxWidth,  std::min(pw, config_.maxBBoxWidth));
        ph = std::max(config_.minBBoxHeight, std::min(ph, config_.maxBBoxHeight));
        if (px < 0.0f) px = 0.0f;
        if (py < 0.0f) py = 0.0f;

        // BoT-SORT CMC: warp predicted state to compensate for camera motion
        if (applyWarp) {
            float nx  = static_cast<float>(W00 * px + W01 * py + W02);
            float ny  = static_cast<float>(W10 * px + W11 * py + W12);
            float nw  = static_cast<float>(sx * pw);
            float nh  = static_cast<float>(sy * ph);
            float nvx = static_cast<float>(W00 * pvx + W01 * pvy);
            float nvy = static_cast<float>(W10 * pvx + W11 * pvy);

            // Patch statePre so that correct() sees the CMC-adjusted prediction
            px = nx; py = ny; pw = nw; ph = nh;
            pvx = nvx; pvy = nvy;
        }

        t.kalmanFilter.statePre.at<float>(0) = px;
        t.kalmanFilter.statePre.at<float>(1) = py;
        t.kalmanFilter.statePre.at<float>(2) = pw;
        t.kalmanFilter.statePre.at<float>(3) = ph;
        t.kalmanFilter.statePre.at<float>(4) = pvx;
        t.kalmanFilter.statePre.at<float>(5) = pvy;
        t.kalmanFilter.statePre.at<float>(6) = pvw;
        t.kalmanFilter.statePre.at<float>(7) = pvh;

        t.velocity     = cv::Point2f(pvx, pvy);
        t.sizeVelocity = cv::Point2f(pvw, pvh);
        t.lastPredictedCenter = cv::Point(
            static_cast<int>(px + pw / 2.0f),
            static_cast<int>(py + ph / 2.0f));

        t.cmcPredictedBox = cv::Rect(
            static_cast<int>(px),
            static_cast<int>(py),
            static_cast<int>(pw),
            static_cast<int>(ph));

        predictedBoxes.push_back(t.cmcPredictedBox);
    }

    return predictedBoxes;
}

// ========== Kalman correct with adaptive R ==========

void BotSortTracker::correctTrack(BotSortTrack& t,
                                   const cv::Rect& det,
                                   float conf,
                                   const cv::Mat& feat) {
    const cv::Mat& predictedState = t.kalmanFilter.statePre;
    float pred_h = predictedState.at<float>(3);
    if (pred_h < 1.0f) pred_h = 1.0f;

    const float sw = 1.0f / 20.0f;
    t.kalmanFilter.measurementNoiseCov.at<float>(0, 0) = (sw * pred_h) * (sw * pred_h);
    t.kalmanFilter.measurementNoiseCov.at<float>(1, 1) = (sw * pred_h) * (sw * pred_h);
    t.kalmanFilter.measurementNoiseCov.at<float>(2, 2) = 1.0f;
    t.kalmanFilter.measurementNoiseCov.at<float>(3, 3) = (sw * pred_h) * (sw * pred_h);

    cv::Mat measurement(4, 1, CV_32F);
    measurement.at<float>(0) = static_cast<float>(det.x);
    measurement.at<float>(1) = static_cast<float>(det.y);
    measurement.at<float>(2) = static_cast<float>(det.width);
    measurement.at<float>(3) = static_cast<float>(det.height);
    t.kalmanFilter.correct(measurement);

    const cv::Mat& sp = t.kalmanFilter.statePost;
    t.velocity.x     = sp.at<float>(4);
    t.velocity.y     = sp.at<float>(5);
    t.sizeVelocity.x = sp.at<float>(6);
    t.sizeVelocity.y = sp.at<float>(7);

    t.boundingBox    = det;
    t.missedFrames   = 0;
    t.state          = BotSortTrack::TRACKING;
    t.detectionConfidence = conf;

    // ReID feature EMA update
    if (config_.useReId && !feat.empty()) {
        if (t.featureEmbedding.empty()) {
            t.featureEmbedding = feat.clone();
            t.lastReidSimilarity = 1.0f;
        } else {
            t.lastReidSimilarity = static_cast<float>(
                1.0 - cosineDistance(t.featureEmbedding, feat));
            t.featureEmbedding =
                config_.featureEmaDecay * t.featureEmbedding
                + (1.0 - config_.featureEmaDecay) * feat;
            cv::normalize(t.featureEmbedding, t.featureEmbedding,
                          1.0, 0.0, cv::NORM_L2);
        }
    }

    t.lastActualCenter = cv::Point(
        det.x + det.width / 2,
        det.y + det.height / 2);
    t.trajectory.push_back(t.lastActualCenter);
    if (t.trajectory.size() > 35) {
        t.trajectory.pop_front();
    }
}

// ========== Stage 1: high-confidence association ==========

BotSortTracker::AssociationResult BotSortTracker::associateHighConf(
        const std::vector<cv::Rect>& dets,
        const std::vector<float>& confs,
        const std::vector<cv::Mat>& feats) {
    const int nT = static_cast<int>(tracks_.size());
    const int nD = static_cast<int>(dets.size());

    AssociationResult result;
    result.matchedDetections.assign(nD, false);

    std::vector<int> activeIdx;
    activeIdx.reserve(nT);
    for (int i = 0; i < nT; ++i) {
        if (tracks_[i].state == BotSortTrack::TRACKING) {
            activeIdx.push_back(i);
        }
    }

    if (nD == 0 || activeIdx.empty()) {
        // No association possible; all detections stay unmatched so they can
        // still attempt LOST-track recovery before opening new IDs.
        result.unmatchedTrackIdx = std::move(activeIdx);
        return result;
    }

    const bool reidActive = config_.useReId
        && !feats.empty()
        && static_cast<int>(feats.size()) == nD;

    // Build cost matrix (TRACKING tracks only for stage 1)
    const int nA = static_cast<int>(activeIdx.size());
    std::vector<std::vector<double>> costMatrix(nA, std::vector<double>(nD, 100.0));

    for (int ai = 0; ai < nA; ++ai) {
        const int ti = activeIdx[ai];
        const cv::Rect& predBox = tracks_[ti].cmcPredictedBox;
        for (int j = 0; j < nD; ++j) {
            double iou = computeIoU(config_.iouType, predBox, dets[j]);
            if (iou > config_.iouThreshold) {
                double iouCost = 1.0 - std::max(0.0, std::min(1.0, iou));
                if (reidActive
                    && !tracks_[ti].featureEmbedding.empty()
                    && !feats[j].empty()) {
                    double appCost = cosineDistance(tracks_[ti].featureEmbedding, feats[j]);
                    costMatrix[ai][j] = std::max(0.0,
                        config_.reidAlpha * iouCost
                        + (1.0 - config_.reidAlpha) * appCost);
                } else {
                    costMatrix[ai][j] = std::max(0.0, iouCost);
                }
            }
        }
    }

    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    hungarian.Solve(costMatrix, assignment);

    result.unmatchedTrackIdx.reserve(nA);

    for (int ai = 0; ai < nA; ++ai) {
        const int ti = activeIdx[ai];
        const int detJ = assignment[ai];
        if (detJ >= 0 && detJ < nD && costMatrix[ai][detJ] < 0.9) {
            const cv::Mat& feat = (!feats.empty() && detJ < static_cast<int>(feats.size()))
                                  ? feats[detJ] : cv::Mat();
            correctTrack(tracks_[ti], dets[detJ],
                         (!confs.empty() ? confs[detJ] : 1.0f),
                         feat);
            result.matchedDetections[detJ] = true;
        } else {
            result.unmatchedTrackIdx.push_back(ti);
        }
    }

    return result;
}

// ========== Stage 2: low-confidence association for unmatched TRACKING tracks ==========

BotSortTracker::AssociationResult BotSortTracker::matchTrackingWithLowConf(
        const std::vector<int>& trackIdx,
        const std::vector<cv::Rect>& dets,
        const std::vector<float>& confs,
        const std::vector<cv::Mat>& feats) {
    AssociationResult result;
    result.matchedDetections.assign(dets.size(), false);
    if (trackIdx.empty() || dets.empty()) {
        result.unmatchedTrackIdx = trackIdx;
        return result;
    }

    const int nT = static_cast<int>(trackIdx.size());
    const int nD = static_cast<int>(dets.size());
    const bool reidActiveLow = config_.useReId
        && !feats.empty()
        && static_cast<int>(feats.size()) == nD;

    std::vector<std::vector<double>> costMatrix(nT, std::vector<double>(nD, 100.0));
    for (int i = 0; i < nT; ++i) {
        const int ti = trackIdx[i];
        if (ti < 0 || ti >= static_cast<int>(tracks_.size()) ||
            tracks_[ti].state != BotSortTrack::TRACKING) {
            continue;
        }
        const cv::Rect& box = tracks_[ti].cmcPredictedBox;
        for (int j = 0; j < nD; ++j) {
            double iou = computeIoU(config_.iouType, box, dets[j]);
            if (iou > config_.lowConfIouThreshold) {
                double iouCost = 1.0 - std::max(0.0, std::min(1.0, iou));
                if (reidActiveLow
                    && !tracks_[ti].featureEmbedding.empty()
                    && !feats[j].empty()) {
                    double appCost = cosineDistance(tracks_[ti].featureEmbedding, feats[j]);
                    costMatrix[i][j] = std::max(0.0,
                        config_.reidAlpha * iouCost
                        + (1.0 - config_.reidAlpha) * appCost);
                } else {
                    costMatrix[i][j] = std::max(0.0, iouCost);
                }
            }
        }
    }

    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    hungarian.Solve(costMatrix, assignment);

    for (int i = 0; i < nT; ++i) {
        const int ti = trackIdx[i];
        const int detJ = assignment[i];
        if (ti < 0 || ti >= static_cast<int>(tracks_.size()) ||
            tracks_[ti].state != BotSortTrack::TRACKING) {
            continue;
        }
        if (detJ >= 0 && detJ < nD &&
            costMatrix[i][detJ] < 0.9 &&
            !result.matchedDetections[detJ]) {
            const cv::Mat& feat = (!feats.empty() && detJ < static_cast<int>(feats.size()))
                                  ? feats[detJ] : cv::Mat();
            correctTrack(tracks_[ti], dets[detJ],
                         (!confs.empty() ? confs[detJ] : 1.0f),
                         feat);
            result.matchedDetections[detJ] = true;
        } else {
            result.unmatchedTrackIdx.push_back(ti);
        }
    }

    return result;
}

void BotSortTracker::markTrackingTracksLost(const std::vector<int>& trackIdx) {
    for (int ti : trackIdx) {
        if (ti < 0 || ti >= static_cast<int>(tracks_.size()) ||
            tracks_[ti].state != BotSortTrack::TRACKING) {
            continue;
        }

        tracks_[ti].missedFrames++;
        if (config_.usePredictionInLost) {
            tracks_[ti].boundingBox = tracks_[ti].cmcPredictedBox;
        }
        if (tracks_[ti].missedFrames > config_.lostStateThreshold) {
            tracks_[ti].state = BotSortTrack::LOST;
        }
    }
}

void BotSortTracker::ageExistingLostTracks(const std::vector<int>& lostIdx) {
    for (int ti : lostIdx) {
        if (ti < 0 || ti >= static_cast<int>(tracks_.size()) ||
            tracks_[ti].state != BotSortTrack::LOST) {
            continue;
        }

        tracks_[ti].missedFrames++;
        if (config_.usePredictionInLost) {
            tracks_[ti].boundingBox = tracks_[ti].cmcPredictedBox;
        }
    }
}

// ========== Stage 3/4: revival of LOST tracks ==========

std::vector<bool> BotSortTracker::reviveLostTracks(const std::vector<cv::Rect>& dets,
                                                   const std::vector<float>& confs,
                                                   const std::vector<cv::Mat>& feats) {
    std::vector<bool> matchedDets(dets.size(), false);
    if (dets.empty()) return matchedDets;

    std::vector<int> lostIdx;
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        if (tracks_[i].state == BotSortTrack::LOST) {
            lostIdx.push_back(i);
        }
    }
    if (lostIdx.empty()) return matchedDets;

    const int nL = static_cast<int>(lostIdx.size());
    const int nD = static_cast<int>(dets.size());
    const bool reidActive = config_.useReId
        && !feats.empty()
        && static_cast<int>(feats.size()) == nD;

    std::vector<std::vector<double>> costMatrix(nL, std::vector<double>(nD, 100.0));
    for (int li = 0; li < nL; ++li) {
        const BotSortTrack& track = tracks_[lostIdx[li]];
        const cv::Rect& box = track.cmcPredictedBox;
        for (int j = 0; j < nD; ++j) {
            const double iou = computeIoU(config_.iouType, box, dets[j]);
            const bool iouGate = iou > config_.lowConfIouThreshold;

            double appCost = -1.0;
            if (reidActive && !track.featureEmbedding.empty() && !feats[j].empty()) {
                appCost = cosineDistance(track.featureEmbedding, feats[j]);
            }

            if (iouGate) {
                double iouCost = 1.0 - std::max(0.0, std::min(1.0, iou));
                if (appCost >= 0.0) {
                    costMatrix[li][j] = std::max(0.0,
                        config_.reidAlpha * iouCost
                        + (1.0 - config_.reidAlpha) * appCost);
                } else {
                    costMatrix[li][j] = std::max(0.0, iouCost);
                }
            } else if (appCost >= 0.0 && appCost < config_.reidAppearanceThresh) {
                // ReID-only recovery: object reappeared away from the predicted
                // box but the appearance embedding still matches.
                costMatrix[li][j] = appCost;
            }
        }
    }

    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    hungarian.Solve(costMatrix, assignment);

    for (int li = 0; li < nL; ++li) {
        const int detJ = assignment[li];
        if (detJ < 0 || detJ >= nD) continue;
        if (costMatrix[li][detJ] > 0.9 || matchedDets[detJ]) continue;

        const int ti = lostIdx[li];
        const cv::Mat& feat = (!feats.empty() && detJ < static_cast<int>(feats.size()))
                              ? feats[detJ] : cv::Mat();
        correctTrack(tracks_[ti], dets[detJ],
                     (!confs.empty() ? confs[detJ] : 1.0f),
                     feat);
        matchedDets[detJ] = true;
    }

    return matchedDets;
}

// ========== Initialize new tracks ==========

void BotSortTracker::initNewTracks(const std::vector<cv::Rect>& dets,
                                    const std::vector<bool>& matched,
                                    const std::vector<float>& confs,
                                    const std::vector<cv::Mat>& feats) {
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
        if (matched[i]) continue;

        BotSortTrack nt;
        nt.id          = nextId_++;
        nt.boundingBox = dets[i];
        nt.kalmanFilter = createKalmanFilter();

        cv::Mat initState(8, 1, CV_32F, cv::Scalar(0));
        initState.at<float>(0) = static_cast<float>(dets[i].x);
        initState.at<float>(1) = static_cast<float>(dets[i].y);
        initState.at<float>(2) = static_cast<float>(dets[i].width);
        initState.at<float>(3) = static_cast<float>(dets[i].height);
        nt.kalmanFilter.statePost = initState;

        if (!confs.empty() && i < static_cast<int>(confs.size())) {
            nt.detectionConfidence = confs[i];
        }
        if (!feats.empty() && i < static_cast<int>(feats.size()) && !feats[i].empty()) {
            nt.featureEmbedding = feats[i].clone();
        }

        nt.color = randomColor();
        nt.lastActualCenter = cv::Point(
            dets[i].x + dets[i].width / 2,
            dets[i].y + dets[i].height / 2);
        nt.trajectory.push_back(nt.lastActualCenter);
        nt.cmcPredictedBox = dets[i];

        tracks_.push_back(std::move(nt));
    }
}

// ========== Prune expired tracks ==========

void BotSortTracker::pruneLostTracks() {
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
            [this](const BotSortTrack& t) {
                bool outOfBounds = config_.removeOutOfBounds &&
                    (t.boundingBox.x + t.boundingBox.width  < config_.cameraBounds.x ||
                     t.boundingBox.y + t.boundingBox.height < config_.cameraBounds.y ||
                     t.boundingBox.x > config_.cameraBounds.x + config_.cameraBounds.width  ||
                     t.boundingBox.y > config_.cameraBounds.y + config_.cameraBounds.height);
                return t.missedFrames > config_.maxLostFrames || outOfBounds;
            }),
        tracks_.end());
}

// ========== IObjectTracker::update ==========

std::vector<IObjectTracker::TrackedResult> BotSortTracker::update(
    const std::vector<Detection>& detections) {

    if (telemetry_) {
        ITrackerTelemetry::FrameMetadata metadata;
        metadata.frameIndex = frameIndexCounter_;
        if (frameSizeHint_.has_value()) {
            metadata.sourceSize    = frameSizeHint_->source;
            metadata.processedSize = frameSizeHint_->processed;
        }
        telemetry_->onFrameStart(metadata);
    }

    // Split detections by confidence
    std::vector<cv::Rect> highRects, lowRects;
    std::vector<float>    highConfs, lowConfs;
    std::vector<cv::Mat>  highFeats, lowFeats;

    for (const auto& det : detections) {
        if (det.bbox.width <= 0 || det.bbox.height <= 0 ||
            det.bbox.x < 0 || det.bbox.y < 0) continue;

        if (det.confidence > config_.highConfThreshold) {
            highRects.push_back(det.bbox);
            highConfs.push_back(det.confidence);
            highFeats.push_back(det.featureVector);
        } else if (det.confidence > config_.lowConfFloor) {
            lowRects.push_back(det.bbox);
            lowConfs.push_back(det.confidence);
            lowFeats.push_back(det.featureVector);
        }
    }

    std::vector<int> existingLostIdx;
    existingLostIdx.reserve(tracks_.size());
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        if (tracks_[i].state == BotSortTrack::LOST) {
            existingLostIdx.push_back(i);
        }
    }

    // CMC: estimate warp between previous and current grayscale frame.
    // Pass current track boxes so goodFeaturesToTrack avoids pedestrian regions.
    cv::Mat warp;
    if (!prevGray_.empty() && !currGray_.empty()
        && config_.cmcMethod != BotSortConfig::CmcMethod::NONE) {
        std::vector<cv::Rect> trackBoxes;
        if (config_.cmcMaskTrackedObjects) {
            trackBoxes.reserve(tracks_.size());
            for (const auto& t : tracks_) {
                trackBoxes.push_back(t.boundingBox);
            }
        }
        warp = cmc_.estimate(prevGray_, currGray_, trackBoxes);
    }

    // Predict all tracks and apply CMC warp
    predictAllAndApplyCMC(warp);

    // Stage 1: high-confidence detections → TRACKING tracks
    AssociationResult highMatch = associateHighConf(highRects, highConfs, highFeats);
    std::vector<int> unmatchedTracking = std::move(highMatch.unmatchedTrackIdx);

    // Stage 2: low-confidence detections first recover Stage-1-unmatched
    // TRACKING tracks. This is the ByteTrack step that prevents false misses
    // when detector confidence dips for an otherwise active object.
    AssociationResult lowMatch;
    if (!lowRects.empty()) {
        lowMatch = matchTrackingWithLowConf(
            unmatchedTracking, lowRects, lowConfs, lowFeats);
        unmatchedTracking = std::move(lowMatch.unmatchedTrackIdx);
    }

    markTrackingTracksLost(unmatchedTracking);

    // Stage 3: unmatched high-confidence detections → LOST tracks (IoU + ReID).
    // Recovers objects that reappear with high confidence after a short
    // disappearance, instead of opening a new ID immediately.
    {
        std::vector<cv::Rect> remainingHighRects;
        std::vector<float> remainingHighConfs;
        std::vector<cv::Mat> remainingHighFeats;
        std::vector<size_t> remainingHighOrigIdx;
        remainingHighRects.reserve(highRects.size());
        remainingHighConfs.reserve(highConfs.size());
        remainingHighFeats.reserve(highFeats.size());
        remainingHighOrigIdx.reserve(highRects.size());
        for (size_t i = 0; i < highRects.size(); ++i) {
            if (i < highMatch.matchedDetections.size() && highMatch.matchedDetections[i]) {
                continue;
            }
            remainingHighRects.push_back(highRects[i]);
            if (i < highConfs.size()) remainingHighConfs.push_back(highConfs[i]);
            if (i < highFeats.size()) remainingHighFeats.push_back(highFeats[i]);
            remainingHighOrigIdx.push_back(i);
        }
        if (!remainingHighRects.empty()) {
            std::vector<bool> revived = reviveLostTracks(
                remainingHighRects, remainingHighConfs, remainingHighFeats);
            for (size_t k = 0; k < revived.size(); ++k) {
                if (revived[k]) {
                    highMatch.matchedDetections[remainingHighOrigIdx[k]] = true;
                }
            }
        }
    }

    // Stage 4: BoT-SORT/ReID-style recovery of remaining LOST tracks, using only
    // low-confidence detections not consumed by active tracks above.
    if (!lowRects.empty()) {
        std::vector<cv::Rect> remainingLowRects;
        std::vector<float> remainingLowConfs;
        std::vector<cv::Mat> remainingLowFeats;
        remainingLowRects.reserve(lowRects.size());
        remainingLowConfs.reserve(lowConfs.size());
        remainingLowFeats.reserve(lowFeats.size());
        for (size_t i = 0; i < lowRects.size(); ++i) {
            if (i < lowMatch.matchedDetections.size() && lowMatch.matchedDetections[i]) {
                continue;
            }
            remainingLowRects.push_back(lowRects[i]);
            if (i < lowConfs.size()) remainingLowConfs.push_back(lowConfs[i]);
            if (i < lowFeats.size()) remainingLowFeats.push_back(lowFeats[i]);
        }
        reviveLostTracks(remainingLowRects, remainingLowConfs, remainingLowFeats);
    }

    // Stage 5: open new IDs only for high-confidence detections that matched
    // neither an active nor a LOST track.
    initNewTracks(highRects, highMatch.matchedDetections, highConfs, highFeats);

    ageExistingLostTracks(existingLostIdx);

    // Remove expired tracks
    pruneLostTracks();

    // Build result vector
    std::vector<TrackedResult> results;
    results.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        if (t.state != BotSortTrack::TRACKING) {
            continue;
        }
        TrackedResult r;
        r.trackId      = t.id;
        r.bbox         = t.boundingBox;
        r.velocity     = t.velocity;
        r.confidence   = t.detectionConfidence;
        r.classId      = -1;
        r.isPredicted  = (t.state == BotSortTrack::LOST);
        r.age          = t.age;
        r.missedFrames = t.missedFrames;
        results.push_back(r);
    }

    if (telemetry_) {
        for (const auto& r : results) {
            telemetry_->onTrackResult(frameIndexCounter_, r);
        }
        telemetry_->onFrameEnd(frameIndexCounter_);
    }

    // Increment age
    for (auto& t : tracks_) {
        t.age++;
    }
    frameIndexCounter_++;

    // Rotate CMC frame buffers
    prevGray_ = currGray_;
    currGray_ = cv::Mat();

    return results;
}

// ========== IObjectTracker helpers ==========

void BotSortTracker::reset() {
    tracks_.clear();
    nextId_ = 0;
    frameIndexCounter_ = 0;
    prevGray_ = cv::Mat();
    currGray_ = cv::Mat();
    frameSizeHint_.reset();
    if (telemetry_) {
        telemetry_->flush();
    }
}

int BotSortTracker::getActiveTrackCount() const {
    return static_cast<int>(std::count_if(tracks_.begin(), tracks_.end(),
        [](const BotSortTrack& t) { return t.state == BotSortTrack::TRACKING; }));
}

int BotSortTracker::getTotalTrackCount() const {
    return static_cast<int>(tracks_.size());
}

std::optional<IObjectTracker::TrackState> BotSortTracker::getTrackState(int trackId) const {
    const BotSortTrack* t = findTrack(trackId);
    if (!t) return std::nullopt;
    return mapState(t->state);
}

IObjectTracker::TrackState BotSortTracker::mapState(BotSortTrack::State s) const {
    return (s == BotSortTrack::TRACKING) ? TrackState::TRACKING : TrackState::LOST;
}

const BotSortTrack* BotSortTracker::findTrack(int trackId) const {
    auto it = std::find_if(tracks_.begin(), tracks_.end(),
        [trackId](const BotSortTrack& t) { return t.id == trackId; });
    return (it != tracks_.end()) ? &(*it) : nullptr;
}

// ========== IVisualTracker ==========

void BotSortTracker::draw(cv::Mat& frame, const VisualizationConfig& cfg) {
    for (auto& t : tracks_) {
        if (!cfg.showLostTracks && t.state == BotSortTrack::LOST) continue;

        cv::Scalar color = t.color;
        if (cfg.colorByState) {
            color = (t.state == BotSortTrack::TRACKING) ? cfg.trackingColor : cfg.lostColor;
        }

        if (cfg.showBoundingBox) {
            cv::rectangle(frame, t.boundingBox, color, cfg.boundingBoxThickness);
        }

        if (cfg.showTrackId) {
            std::string idText = "ID:" + std::to_string(t.id);
            cv::putText(frame, idText,
                        cv::Point(t.boundingBox.x, t.boundingBox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        cfg.trackIdFontScale, cfg.trackIdColor, cfg.trackIdThickness);
        }

        if (cfg.showReidScore && t.lastReidSimilarity >= 0.f) {
            int pct = static_cast<int>(t.lastReidSimilarity * 100.f);
            cv::putText(frame, "R:" + std::to_string(pct) + "%",
                        cv::Point(t.boundingBox.x, t.boundingBox.y - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 100, 255), 1);
        }

        if (cfg.showCenter) {
            cv::Point center(t.boundingBox.x + t.boundingBox.width / 2,
                             t.boundingBox.y + t.boundingBox.height / 2);
            t.lastActualCenter = center;
            cv::circle(frame, center, cfg.centerRadius, cfg.centerColor, cv::FILLED);
        }

        if (cfg.showVelocity) {
            cv::Point end = t.lastActualCenter + cv::Point(
                static_cast<int>(t.velocity.x * cfg.velocityScale),
                static_cast<int>(t.velocity.y * cfg.velocityScale));
            cv::arrowedLine(frame, t.lastActualCenter, end,
                            cfg.velocityColor, cfg.velocityThickness);
        }

        if (cfg.showTrajectory) {
            t.trajectory.push_back(t.lastActualCenter);
            if (static_cast<int>(t.trajectory.size()) > cfg.trajectoryLength) {
                t.trajectory.pop_front();
            }
            for (size_t i = 1; i < t.trajectory.size(); ++i) {
                cv::line(frame, t.trajectory[i - 1], t.trajectory[i],
                         cfg.trajectoryColor, cfg.trajectoryThickness);
            }
        }

        if (cfg.showAge) {
            cv::putText(frame, "Age:" + std::to_string(t.age),
                        cv::Point(t.boundingBox.x + t.boundingBox.width - 60,
                                  t.boundingBox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }

        if (cfg.showMissedFrames && t.missedFrames > 0) {
            cv::putText(frame, "Miss:" + std::to_string(t.missedFrames),
                        cv::Point(t.boundingBox.x,
                                  t.boundingBox.y + t.boundingBox.height + 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
        }
    }
}

std::vector<std::vector<cv::Point>> BotSortTracker::getTrajectories() const {
    std::vector<std::vector<cv::Point>> result;
    result.reserve(tracks_.size());
    for (const auto& t : tracks_) {
        result.emplace_back(t.trajectory.begin(), t.trajectory.end());
    }
    return result;
}

std::vector<cv::Point> BotSortTracker::getTrajectory(int trackId) const {
    const BotSortTrack* t = findTrack(trackId);
    if (!t) return {};
    return std::vector<cv::Point>(t->trajectory.begin(), t->trajectory.end());
}

void BotSortTracker::clearTrajectories() {
    for (auto& t : tracks_) t.trajectory.clear();
}

void BotSortTracker::setTrajectoryLength(int length) {
    config_.maxTrajectoryLength = length;
}

// ========== IConfigurableTracker ==========

BotSortConfig BotSortTracker::getConfig() const { return config_; }

void BotSortTracker::setConfig(const BotSortConfig& cfg) {
    if (validateConfig(cfg)) {
        config_ = cfg;
        cmc_.setConfig(cfg);
    } else {
        throw std::invalid_argument("Invalid BotSortConfig");
    }
}

void BotSortTracker::setParameter(const std::string& key, const std::any& value) {
    if (key == "iouThreshold")        config_.iouThreshold        = std::any_cast<float>(value);
    else if (key == "maxLostFrames")  config_.maxLostFrames        = std::any_cast<int>(value);
    else if (key == "highConfThreshold") config_.highConfThreshold = std::any_cast<float>(value);
    else if (key == "cmcMethod")      config_.cmcMethod            = std::any_cast<BotSortConfig::CmcMethod>(value);
    else throw std::invalid_argument("Unknown BotSortConfig key: " + key);
}

std::any BotSortTracker::getParameter(const std::string& key) const {
    if (key == "iouThreshold")        return config_.iouThreshold;
    if (key == "maxLostFrames")       return config_.maxLostFrames;
    if (key == "highConfThreshold")   return config_.highConfThreshold;
    if (key == "cmcMethod")           return config_.cmcMethod;
    throw std::invalid_argument("Unknown BotSortConfig key: " + key);
}

bool BotSortTracker::validateConfig(const BotSortConfig& cfg) const {
    return cfg.iouThreshold >= 0.0f && cfg.iouThreshold <= 1.0f
        && cfg.maxLostFrames >= 0
        && cfg.processNoise >= 0.0f
        && cfg.measurementNoise >= 0.0f
        && cfg.maxTrajectoryLength >= 1
        && cfg.softIoUSlack > 0.0f
        && cfg.alphaIoUExponent > 0.0f;
}

// ========== Telemetry ==========

void BotSortTracker::setTelemetry(const std::shared_ptr<ITrackerTelemetry>& tl) {
    telemetry_ = tl;
}

void BotSortTracker::clearTelemetry() { telemetry_.reset(); }

void BotSortTracker::setFrameSizeHint(const cv::Size& src, const cv::Size& proc) {
    frameSizeHint_ = FrameSizeInfo{src, proc};
}

void BotSortTracker::clearFrameSizeHint() { frameSizeHint_.reset(); }

// ========== IoU calculations (ported from KalmanIoUByteTrack) ==========

double BotSortTracker::calcIoU(const cv::Rect& a, const cv::Rect& b) {
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width,  b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    const int iw = std::max(0, x2 - x1);
    const int ih = std::max(0, y2 - y1);
    const int inter = iw * ih;
    const int uni   = a.area() + b.area() - inter;
    return (uni <= 0) ? 0.0 : static_cast<double>(inter) / uni;
}

double BotSortTracker::calcGIoU(const cv::Rect& a, const cv::Rect& b) {
    double iou = calcIoU(a, b);
    const int ex1 = std::min(a.x, b.x);
    const int ey1 = std::min(a.y, b.y);
    const int ex2 = std::max(a.x + a.width,  b.x + b.width);
    const int ey2 = std::max(a.y + a.height, b.y + b.height);
    const double encArea = static_cast<double>((ex2 - ex1) * (ey2 - ey1));
    if (encArea <= 0.0) return iou;
    const double uniArea = a.area() + b.area() - (a & b).area();
    return std::max(-1.0, std::min(iou - (encArea - uniArea) / encArea, 1.0));
}

double BotSortTracker::calcDIoU(const cv::Rect& a, const cv::Rect& b) {
    double iou = calcIoU(a, b);
    const cv::Point2f ca(a.x + a.width  * 0.5f, a.y + a.height * 0.5f);
    const cv::Point2f cb(b.x + b.width  * 0.5f, b.y + b.height * 0.5f);
    const double cd2 = cv::norm(ca - cb) * cv::norm(ca - cb);
    const int ex1 = std::min(a.x, b.x);
    const int ey1 = std::min(a.y, b.y);
    const int ex2 = std::max(a.x + a.width,  b.x + b.width);
    const int ey2 = std::max(a.y + a.height, b.y + b.height);
    const double ew = ex2 - ex1, eh = ey2 - ey1;
    const double diag2 = ew * ew + eh * eh;
    if (diag2 < 1e-6) return iou;
    return std::max(-1.0, std::min(iou - cd2 / diag2, 1.0));
}

double BotSortTracker::calcCIoU(const cv::Rect& a, const cv::Rect& b) {
    double iou   = calcIoU(a, b);
    double diou  = calcDIoU(a, b);
    if (a.height <= 0 || b.height <= 0) return diou;
    const double v = (4.0 / (M_PI * M_PI))
                   * std::pow(std::atan(static_cast<double>(a.width)  / a.height)
                            - std::atan(static_cast<double>(b.width)  / b.height), 2);
    const double alpha = v / ((1.0 - iou) + v + 1e-8);
    return std::max(-1.0, std::min(diou - alpha * v, 1.0));
}

double BotSortTracker::calcSoftIoU(const cv::Rect& a, const cv::Rect& b) {
    const double iou = std::clamp(calcIoU(a, b), 0.0, 1.0);
    const double slack = std::max(0.0, static_cast<double>(config_.softIoUSlack));
    const double denom = iou + slack * (1.0 - iou) + 1e-12;
    return std::clamp(iou / denom, 0.0, 1.0);
}

double BotSortTracker::calcAlphaIoU(const cv::Rect& a, const cv::Rect& b) {
    const double iou   = std::clamp(calcIoU(a, b), 0.0, 1.0);
    const double alpha = std::max(1e-3, static_cast<double>(config_.alphaIoUExponent));
    return std::pow(iou, alpha);
}

double BotSortTracker::computeIoU(BotSortConfig::IoUType type,
                                   const cv::Rect& a, const cv::Rect& b) {
    if (a.area() <= 0 || b.area() <= 0 ||
        a.width <= 0 || a.height <= 0 ||
        b.width <= 0 || b.height <= 0) return 0.0;

    switch (type) {
        case BotSortConfig::IoUType::IOU:  return calcIoU(a, b);
        case BotSortConfig::IoUType::GIOU: return calcGIoU(a, b);
        case BotSortConfig::IoUType::DIOU: return calcDIoU(a, b);
        case BotSortConfig::IoUType::CIOU: return calcCIoU(a, b);
        case BotSortConfig::IoUType::SIOU: return calcSoftIoU(a, b);
        case BotSortConfig::IoUType::AIOU: return calcAlphaIoU(a, b);
        default: return 0.0;
    }
}

double BotSortTracker::cosineDistance(const cv::Mat& a, const cv::Mat& b) {
    return 1.0 - std::max(0.0, std::min(1.0, a.dot(b)));
}

// ========== Utility ==========

cv::Scalar BotSortTracker::randomColor() {
    cv::RNG rng(static_cast<uint64_t>(timeSinceEpochMs()));
    return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}

uint64_t BotSortTracker::timeSinceEpochMs() const {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}
