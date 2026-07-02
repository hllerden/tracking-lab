#include "kalman_iou_byte_track.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Use namespaced TrackedObject within this file
using TrackedObject = KalmanIoUByteTrackNS::TrackedObject;

KalmanIoUByteTrack::KalmanIoUByteTrack() : nextId(0) {}

KalmanIoUByteTrack::KalmanIoUByteTrack(const KalmanIoUConfig& cfg)
    : nextId(0), config(cfg) {}

uint64_t KalmanIoUByteTrack::timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

float KalmanIoUByteTrack::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersectionWidth = std::max(0, x2 - x1);
    int intersectionHeight = std::max(0, y2 - y1);
    int intersectionArea = intersectionWidth * intersectionHeight;

    int unionArea = box1.area() + box2.area() - intersectionArea;
    if (unionArea <= 0) {
        throw std::runtime_error("Invalid union area: cannot divide by zero or negative.");
    }
    return static_cast<float>(intersectionArea) / unionArea;
}



cv::KalmanFilter KalmanIoUByteTrack::createKalmanFilter() {
    cv::KalmanFilter kf(8, 4, 0); // 8 state, 4 measurement

    // Transition matrix (8x8) with dt = 1
    float dt = 1.0f;
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

    // Measurement matrix (4x8)
    kf.measurementMatrix = (cv::Mat_<float>(4, 8) << 
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0
    );

    // Process noise covariance (8x8) - INITIAL VALUES ONLY
    // These will be dynamically updated in updateTrackersWithHungarian()
    // Set to reasonable defaults (will be overridden per tracker based on bbox height)
    kf.processNoiseCov = cv::Mat::eye(8, 8, CV_32F) * 1e-4f;

    // Measurement noise covariance (4x4) - INITIAL VALUES ONLY
    // These will be dynamically updated before each correct() call
    kf.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 1e-2f;

    // Initial error covariance (8x8) - Higher uncertainty for velocities
    kf.errorCovPost = cv::Mat::eye(8, 8, CV_32F);
    kf.errorCovPost.at<float>(0,0) = 1.0f;  // x
    kf.errorCovPost.at<float>(1,1) = 1.0f;  // y
    kf.errorCovPost.at<float>(2,2) = 1.0f;  // w
    kf.errorCovPost.at<float>(3,3) = 1.0f;  // h
    kf.errorCovPost.at<float>(4,4) = 10.0f; // vx (high uncertainty - DeepSORT: 10x)
    kf.errorCovPost.at<float>(5,5) = 10.0f; // vy
    kf.errorCovPost.at<float>(6,6) = 10.0f; // vw (high uncertainty)
    kf.errorCovPost.at<float>(7,7) = 10.0f; // vh

    return kf;
}

double KalmanIoUByteTrack::computeIoU(KalmanIoUConfig::IoUType type, const cv::Rect& boxA, const cv::Rect& boxB) {
    // Defensive programming: return 0 for invalid boxes (will not match)
    if (boxA.area() <= 0 || boxB.area() <= 0) {
        return 0.0;
    }

    // Additional safety check for negative dimensions
    if (boxA.width <= 0 || boxA.height <= 0 || boxB.width <= 0 || boxB.height <= 0) {
        return 0.0;
    }

    switch (type) {
    case KalmanIoUConfig::IoUType::IOU:
        return calculateIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::GIOU:
        return calculateGIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::DIOU:
        return calculateDIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::CIOU:
        return calculateCIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::SIOU:
        return calculateSoftIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::AIOU:
        return calculateAlphaIoU(boxA, boxB);
    default:
        return 0.0; // Unknown type -> no match
    }
}


double KalmanIoUByteTrack::calculateGIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    double iou = calculateIoU(boxA, boxB);
    
    // Enclosing box'u doğru şekilde hesapla
    int enclosing_x1 = std::min(boxA.x, boxB.x);
    int enclosing_y1 = std::min(boxA.y, boxB.y);
    int enclosing_x2 = std::max(boxA.x + boxA.width, boxB.x + boxB.width);
    int enclosing_y2 = std::max(boxA.y + boxA.height, boxB.y + boxB.height);
    
    cv::Rect enclosingBox(enclosing_x1, enclosing_y1, 
                         enclosing_x2 - enclosing_x1, 
                         enclosing_y2 - enclosing_y1);
    
    double enclosingArea = enclosingBox.area();
    if (enclosingArea <= 0.0) return iou; // Exception yerine fallback
    
    double unionArea = boxA.area() + boxB.area() - (boxA & boxB).area();
    
    // GIoU formülü: IoU - (enclosingArea - unionArea) / enclosingArea
    double giou = iou - (enclosingArea - unionArea) / enclosingArea;
    
    return std::max(-1.0, std::min(giou, 1.0)); // [-1, 1] aralığı
}

double KalmanIoUByteTrack::calculateDIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
     double iou = calculateIoU(boxA, boxB);
    
    // Merkez noktaları
    cv::Point2f centerA(boxA.x + boxA.width * 0.5f, boxA.y + boxA.height * 0.5f);
    cv::Point2f centerB(boxB.x + boxB.width * 0.5f, boxB.y + boxB.height * 0.5f);
    
    // Merkezler arası mesafe
    double centerDistance = cv::norm(centerA - centerB);
    
    // Enclosing box ve köşegen uzunluğu
    int enclosing_x1 = std::min(boxA.x, boxB.x);
    int enclosing_y1 = std::min(boxA.y, boxB.y);
    int enclosing_x2 = std::max(boxA.x + boxA.width, boxB.x + boxB.width);
    int enclosing_y2 = std::max(boxA.y + boxA.height, boxB.y + boxB.height);
    
    double enclosingWidth = enclosing_x2 - enclosing_x1;
    double enclosingHeight = enclosing_y2 - enclosing_y1;
    double enclosingDiagonal = std::sqrt(enclosingWidth * enclosingWidth + 
                                        enclosingHeight * enclosingHeight);
    
    if (enclosingDiagonal < 1e-6) return iou;
    
    double diou = iou - (centerDistance * centerDistance) / (enclosingDiagonal * enclosingDiagonal);
    return std::max(-1.0, std::min(diou, 1.0));
}


double KalmanIoUByteTrack::calculateCIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    double iou = calculateIoU(boxA, boxB);
    double diou = calculateDIoU(boxA, boxB); // DIoU'yu yeniden kullan
    
    // Aspect ratio consistency - zero division koruması
    if (boxA.height <= 0 || boxB.height <= 0) return diou;
    
    double wA = static_cast<double>(boxA.width);
    double hA = static_cast<double>(boxA.height);
    double wB = static_cast<double>(boxB.width);
    double hB = static_cast<double>(boxB.height);
    
    double arctanA = std::atan(wA / hA); // radyan
    double arctanB = std::atan(wB / hB); // radyan
    
    double v = (4.0 / (M_PI * M_PI)) * std::pow(arctanA - arctanB, 2);
    
    // α hesaplaması düzeltildi
    double alpha = v / ((1 - iou) + v + 1e-8);
    
    double ciou = diou - alpha * v;
    return std::max(-1.0, std::min(ciou, 1.0));
}

double KalmanIoUByteTrack::calculateSoftIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    const double iou = std::clamp(static_cast<double>(calculateIoU(boxA, boxB)), 0.0, 1.0);
    const double slack = std::max(0.0, static_cast<double>(config.softIoUSlack));
    const double denominator = iou + slack * (1.0 - iou) + 1e-12;
    if (denominator <= 0.0) {
        return 0.0;
    }
    return std::clamp(iou / denominator, 0.0, 1.0);
}

double KalmanIoUByteTrack::calculateAlphaIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    const double iou = std::clamp(static_cast<double>(calculateIoU(boxA, boxB)), 0.0, 1.0);
    const double alpha = std::max(1e-3, static_cast<double>(config.alphaIoUExponent));
    return std::pow(iou, alpha);
}

// Legacy getter/setter methods removed - use getConfig()/setConfig() instead

double KalmanIoUByteTrack::cosineDistance(const cv::Mat& a, const cv::Mat& b) {
    double sim = a.dot(b);
    return 1.0 - std::max(0.0, std::min(1.0, sim));
}

void KalmanIoUByteTrack::updateTrackersWithHungarian(const std::vector<cv::Rect>& detections,
                                                   const std::vector<float>& confidences,
                                                   float iouOverride,
                                                   bool allowNewTrackers,
                                                   bool removeLost,
                                                   const std::vector<cv::Mat>& features,
                                                   std::vector<bool>* matchedDetectionsOut) {
    std::vector<cv::Rect> predictedBoxes;
    for (auto& tracker : trackers) {
        // Adaptive process noise covariance (DeepSORT/BoT-SORT approach)
        // Q matrix scales with bounding box height
        float h = static_cast<float>(tracker.boundingBox.height);
        if (h < 1.0f) h = 1.0f; // Prevent division by zero

        // DeepSORT standard weights
        // const float std_weight_position = 1.0f / 20.0f;   // 0.05
        // const float std_weight_velocity = 1.0f / 160.0f;  // 0.00625
        const float std_weight_position = 1.0f / 40.0f;  // 0.025
        const float std_weight_velocity = 1.0f / 320.0f; // 0.003125
        // Position noise (adaptive)
        float std_pos_x = std_weight_position * h;
        float std_pos_y = std_weight_position * h;
        float std_pos_w = 1e-2f;  // Fixed (size changes slowly)
        float std_pos_h = std_weight_position * h;

        // Velocity noise (adaptive)
        float std_vel_x = std_weight_velocity * h;
        float std_vel_y = std_weight_velocity * h;
        float std_vel_w = 1e-5f;  // Fixed (size velocity almost zero)
        float std_vel_h = std_weight_velocity * h;

        // Update Q matrix (process noise covariance)
        tracker.kalmanFilter.processNoiseCov.at<float>(0, 0) = std_pos_x * std_pos_x;
        tracker.kalmanFilter.processNoiseCov.at<float>(1, 1) = std_pos_y * std_pos_y;
        tracker.kalmanFilter.processNoiseCov.at<float>(2, 2) = std_pos_w * std_pos_w;
        tracker.kalmanFilter.processNoiseCov.at<float>(3, 3) = std_pos_h * std_pos_h;
        tracker.kalmanFilter.processNoiseCov.at<float>(4, 4) = std_vel_x * std_vel_x;
        tracker.kalmanFilter.processNoiseCov.at<float>(5, 5) = std_vel_y * std_vel_y;
        tracker.kalmanFilter.processNoiseCov.at<float>(6, 6) = std_vel_w * std_vel_w;
        tracker.kalmanFilter.processNoiseCov.at<float>(7, 7) = std_vel_h * std_vel_h;

        cv::Mat prediction = tracker.kalmanFilter.predict();

        float pred_x  = prediction.at<float>(0);
        float pred_y  = prediction.at<float>(1);
        float pred_w  = prediction.at<float>(2);
        float pred_h  = prediction.at<float>(3);
        float pred_vx = prediction.at<float>(4);
        float pred_vy = prediction.at<float>(5);
        float pred_vw = prediction.at<float>(6);
        float pred_vh = prediction.at<float>(7);

        // Position constraints
        if (pred_x < 0.0f) pred_x = 0.0f;
        if (pred_y < 0.0f) pred_y = 0.0f;

        float max_vw = config.sizeVelocityClamp * std::max(pred_w, config.minBBoxWidth);
        float max_vh = config.sizeVelocityClamp * std::max(pred_h, config.minBBoxHeight);
        pred_vw = std::max(-max_vw, std::min(pred_vw, max_vw));
        pred_vh = std::max(-max_vh, std::min(pred_vh, max_vh));

        pred_w = std::max(config.minBBoxWidth, std::min(pred_w, config.maxBBoxWidth));
        pred_h = std::max(config.minBBoxHeight, std::min(pred_h, config.maxBBoxHeight));

        predictedBoxes.emplace_back(
            static_cast<int>(pred_x),
            static_cast<int>(pred_y),
            static_cast<int>(pred_w),
            static_cast<int>(pred_h));

        tracker.velocity     = cv::Point2f(pred_vx, pred_vy);
        tracker.sizeVelocity = cv::Point2f(pred_vw, pred_vh);

        tracker.lastPredictedCenter = cv::Point(
            static_cast<int>(pred_x + pred_w / 2.0f),
            static_cast<int>(pred_y + pred_h / 2.0f));
    }

    int numTrackers = static_cast<int>(trackers.size());
    int numDetections = static_cast<int>(detections.size());

    const float effectiveIouThreshold = (iouOverride >= 0.0f) ? iouOverride : config.iouThreshold;

    if (numTrackers == 0 || numDetections == 0) {
        std::vector<bool> matchedDetections(numDetections, false);
        if (allowNewTrackers) {
            addNewTrackers(detections, matchedDetections, confidences, features);
        }
        if (removeLost) {
            removeLostTrackers();
        }
        if (matchedDetectionsOut) {
            *matchedDetectionsOut = std::move(matchedDetections);
        }
        return;
    }
    // cost matrix: pure IoU or hybrid (IoU + cosine appearance) when ReID is enabled
    const bool reidActive = config.useReId && !features.empty()
                            && static_cast<int>(features.size()) == numDetections;
    std::vector<std::vector<double>> costMatrix(numTrackers, std::vector<double>(numDetections, 100.0));
    for (int i = 0; i < numTrackers; ++i) {
        for (int j = 0; j < numDetections; ++j) {
            double iou = computeIoU(config.iouType, predictedBoxes[i], detections[j]);
            if (iou > effectiveIouThreshold) {
                // Clamp to [0.0, 1.0] to handle GIOU/DIOU/SIOU negative values
                double iouCost = 1.0 - std::max(0.0, std::min(1.0, iou));
                if (reidActive
                    && !trackers[i].featureEmbedding.empty()
                    && !features[j].empty()) {
                    double appCost = cosineDistance(trackers[i].featureEmbedding, features[j]);
                    costMatrix[i][j] = std::max(0.0,
                        config.reidAlpha * iouCost + (1.0 - config.reidAlpha) * appCost);
                } else {
                    costMatrix[i][j] = std::max(0.0, iouCost);
                }
            } else {
                costMatrix[i][j] = 100.0;
            }
        }
    }
    // old
    // std::vector<std::vector<double>> costMatrix(numTrackers, std::vector<double>(numDetections, 100.0));
    // for (int i = 0; i < numTrackers; ++i) {
    //     for (int j = 0; j < numDetections; ++j) {
    //         double iou = computeIoU(config.iouType, predictedBoxes[i], detections[j]);
    //         costMatrix[i][j] = (iou > config.iouThreshold) ? 1.0 - iou : 100.0;
    //     }
    // }
    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    hungarian.Solve(costMatrix, assignment);

    std::vector<bool> matchedDetections(numDetections, false);
    for (int i = 0; i < numTrackers; ++i) {
        if (assignment[i] >= 0 && assignment[i] < numDetections) {
            if (costMatrix[i][assignment[i]] > 0.9) {
                assignment[i] = -1;
                continue;
            }

            // Adaptive measurement noise covariance (DeepSORT approach)
            // R matrix scales with predicted bounding box height
            const cv::Mat& predictedState = trackers[i].kalmanFilter.statePre;
            float pred_h = predictedState.at<float>(3);
            if (pred_h < 1.0f) pred_h = 1.0f;

            const float std_weight_position = 1.0f / 20.0f;  // 0.05
            float std_meas_x = std_weight_position * pred_h;
            float std_meas_y = std_weight_position * pred_h;
            float std_meas_w = 1e-0f;  // Fixed (DeepSORT standard)
            float std_meas_h = std_weight_position * pred_h;

            // Update R matrix (measurement noise covariance)
            trackers[i].kalmanFilter.measurementNoiseCov.at<float>(0, 0) = std_meas_x * std_meas_x;
            trackers[i].kalmanFilter.measurementNoiseCov.at<float>(1, 1) = std_meas_y * std_meas_y;
            trackers[i].kalmanFilter.measurementNoiseCov.at<float>(2, 2) = std_meas_w * std_meas_w;
            trackers[i].kalmanFilter.measurementNoiseCov.at<float>(3, 3) = std_meas_h * std_meas_h;

            cv::Mat measurement(4, 1, CV_32F);
            measurement.at<float>(0) = static_cast<float>(detections[assignment[i]].x);
            measurement.at<float>(1) = static_cast<float>(detections[assignment[i]].y);
            measurement.at<float>(2) = static_cast<float>(detections[assignment[i]].width);
            measurement.at<float>(3) = static_cast<float>(detections[assignment[i]].height);
            trackers[i].kalmanFilter.correct(measurement);

            const cv::Mat& statePost = trackers[i].kalmanFilter.statePost;
            trackers[i].velocity.x     = statePost.at<float>(4);
            trackers[i].velocity.y     = statePost.at<float>(5);
            trackers[i].sizeVelocity.x = statePost.at<float>(6);
            trackers[i].sizeVelocity.y = statePost.at<float>(7);

            // std::cout << "[KF] Tracker " << trackers[i].id
            //           << " vw=" << trackers[i].sizeVelocity.x
            //           << " vh=" << trackers[i].sizeVelocity.y << std::endl;
            trackers[i].boundingBox = detections[assignment[i]];
            trackers[i].missedFrames = 0;
            trackers[i].state = TrackedObject::TRACKING;

            // Store detection confidence
            if (!confidences.empty() && assignment[i] < static_cast<int>(confidences.size())) {
                trackers[i].detectionConfidence = confidences[assignment[i]];
            }

            // ReID feature EMA update + similarity recording
            if (reidActive && assignment[i] < static_cast<int>(features.size())
                && !features[assignment[i]].empty()) {
                const cv::Mat& newFeat = features[assignment[i]];
                if (trackers[i].featureEmbedding.empty()) {
                    trackers[i].featureEmbedding = newFeat.clone();
                    trackers[i].lastReidSimilarity = 1.0f;
                } else {
                    trackers[i].lastReidSimilarity = static_cast<float>(
                        1.0 - cosineDistance(trackers[i].featureEmbedding, newFeat));
                    trackers[i].featureEmbedding =
                        config.featureEmaDecay * trackers[i].featureEmbedding
                        + (1.0 - config.featureEmaDecay) * newFeat;
                    cv::normalize(trackers[i].featureEmbedding,
                                  trackers[i].featureEmbedding,
                                  1.0, 0.0, cv::NORM_L2);
                }
            }

            // Update center and trajectory (matches original stalker.cpp)
            trackers[i].lastActualCenter = cv::Point(
                trackers[i].boundingBox.x + trackers[i].boundingBox.width / 2,
                trackers[i].boundingBox.y + trackers[i].boundingBox.height / 2);

            trackers[i].trajectory.push_back(trackers[i].lastActualCenter);
            if (trackers[i].trajectory.size() > 35) {
                trackers[i].trajectory.pop_front();
            }

            matchedDetections[assignment[i]] = true;
        } else {
            trackers[i].missedFrames++;
            if (config.usePredictionInLost && trackers[i].missedFrames <= config.maxLostFrames) {
                // Use prediction for lost tracks (matches original stalker.cpp behavior)
                cv::Mat state = trackers[i].kalmanFilter.statePost;
                trackers[i].boundingBox = cv::Rect(
                    static_cast<int>(state.at<float>(0)),
                    static_cast<int>(state.at<float>(1)),
                    static_cast<int>(state.at<float>(2)),
                    static_cast<int>(state.at<float>(3)));
            }

            if (trackers[i].missedFrames > config.lostStateThreshold) {
                trackers[i].state = TrackedObject::LOST;
            }
        }
    }

    if (allowNewTrackers) {
        addNewTrackers(detections, matchedDetections, confidences, features);
    }
    if (removeLost) {
        removeLostTrackers();
    }
    if (matchedDetectionsOut) {
        *matchedDetectionsOut = std::move(matchedDetections);
    }
}

std::vector<bool> KalmanIoUByteTrack::matchLostTrackers(const std::vector<cv::Rect>& detections,
                                                        const std::vector<float>& confidences,
                                                        const std::vector<cv::Mat>& features) {
    const float kLowConfidenceIouThreshold = config.lowConfIouThreshold;

    std::vector<bool> matchedDetections(detections.size(), false);
    if (detections.empty()) {
        return matchedDetections;
    }

    std::vector<int> lostIndices;
    lostIndices.reserve(trackers.size());
    for (size_t i = 0; i < trackers.size(); ++i) {
        if (trackers[i].state == TrackedObject::LOST) {
            lostIndices.push_back(static_cast<int>(i));
        }
    }

    if (lostIndices.empty()) {
        return matchedDetections;
    }

    const bool reidActive = config.useReId && !features.empty()
                            && features.size() == detections.size();
    std::vector<std::vector<double>> costMatrix(lostIndices.size(), std::vector<double>(detections.size(), 100.0));
    for (size_t i = 0; i < lostIndices.size(); ++i) {
        const TrackedObject& trk = trackers[lostIndices[i]];
        for (size_t j = 0; j < detections.size(); ++j) {
            const double iou = computeIoU(config.iouType, trk.boundingBox, detections[j]);
            const bool iouGate = iou > kLowConfidenceIouThreshold;

            double appCost = -1.0;
            if (reidActive && !trk.featureEmbedding.empty() && !features[j].empty()) {
                appCost = cosineDistance(trk.featureEmbedding, features[j]);
            }

            if (iouGate) {
                double iouCost = 1.0 - std::max(0.0, std::min(1.0, iou));
                if (appCost >= 0.0) {
                    costMatrix[i][j] = std::max(0.0,
                        config.reidAlpha * iouCost + (1.0 - config.reidAlpha) * appCost);
                } else {
                    costMatrix[i][j] = std::max(0.0, iouCost);
                }
            } else if (appCost >= 0.0 && appCost < config.reidAppearanceThresh) {
                // ReID-only recovery: object reappeared away from the predicted
                // box but the appearance embedding still matches.
                costMatrix[i][j] = appCost;
            }
        }
    }

    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    hungarian.Solve(costMatrix, assignment);
    for (size_t i = 0; i < lostIndices.size(); ++i) {
        int detIdx = assignment[i];
        if (detIdx < 0 || detIdx >= static_cast<int>(detections.size())) {
            continue;
        }
        if (costMatrix[i][detIdx] > 0.9 || matchedDetections[detIdx]) {
            continue;
        }

        int trackerIndex = lostIndices[i];

        const cv::Mat& predictedState = trackers[trackerIndex].kalmanFilter.statePre;
        float pred_h = predictedState.at<float>(3);
        if (pred_h < 1.0f) pred_h = 1.0f;

        const float std_weight_position = 1.0f / 20.0f;  // 0.05
        float std_meas_x = std_weight_position * pred_h;
        float std_meas_y = std_weight_position * pred_h;
        float std_meas_w = 1e-0f;
        float std_meas_h = std_weight_position * pred_h;

        trackers[trackerIndex].kalmanFilter.measurementNoiseCov.at<float>(0, 0) = std_meas_x * std_meas_x;
        trackers[trackerIndex].kalmanFilter.measurementNoiseCov.at<float>(1, 1) = std_meas_y * std_meas_y;
        trackers[trackerIndex].kalmanFilter.measurementNoiseCov.at<float>(2, 2) = std_meas_w * std_meas_w;
        trackers[trackerIndex].kalmanFilter.measurementNoiseCov.at<float>(3, 3) = std_meas_h * std_meas_h;

        cv::Mat measurement(4, 1, CV_32F);
        measurement.at<float>(0) = static_cast<float>(detections[detIdx].x);
        measurement.at<float>(1) = static_cast<float>(detections[detIdx].y);
        measurement.at<float>(2) = static_cast<float>(detections[detIdx].width);
        measurement.at<float>(3) = static_cast<float>(detections[detIdx].height);
        trackers[trackerIndex].kalmanFilter.correct(measurement);

        const cv::Mat& statePost = trackers[trackerIndex].kalmanFilter.statePost;
        trackers[trackerIndex].velocity.x     = statePost.at<float>(4);
        trackers[trackerIndex].velocity.y     = statePost.at<float>(5);
        trackers[trackerIndex].sizeVelocity.x = statePost.at<float>(6);
        trackers[trackerIndex].sizeVelocity.y = statePost.at<float>(7);

        trackers[trackerIndex].boundingBox = detections[detIdx];
        trackers[trackerIndex].missedFrames = 0;
        trackers[trackerIndex].state = TrackedObject::TRACKING;

        if (!confidences.empty() && detIdx < static_cast<int>(confidences.size())) {
            trackers[trackerIndex].detectionConfidence = confidences[detIdx];
        }

        // ReID feature EMA update on revival
        if (reidActive && detIdx < static_cast<int>(features.size())
            && !features[detIdx].empty()) {
            const cv::Mat& newFeat = features[detIdx];
            if (trackers[trackerIndex].featureEmbedding.empty()) {
                trackers[trackerIndex].featureEmbedding = newFeat.clone();
            } else {
                trackers[trackerIndex].featureEmbedding =
                    config.featureEmaDecay * trackers[trackerIndex].featureEmbedding
                    + (1.0 - config.featureEmaDecay) * newFeat;
                cv::normalize(trackers[trackerIndex].featureEmbedding,
                              trackers[trackerIndex].featureEmbedding,
                              1.0, 0.0, cv::NORM_L2);
            }
        }

        trackers[trackerIndex].lastActualCenter = cv::Point(
            trackers[trackerIndex].boundingBox.x + trackers[trackerIndex].boundingBox.width / 2,
            trackers[trackerIndex].boundingBox.y + trackers[trackerIndex].boundingBox.height / 2);

        trackers[trackerIndex].trajectory.push_back(trackers[trackerIndex].lastActualCenter);
        if (trackers[trackerIndex].trajectory.size() > 35) {
            trackers[trackerIndex].trajectory.pop_front();
        }

        matchedDetections[detIdx] = true;
    }

    return matchedDetections;
}

void KalmanIoUByteTrack::updateTrackers(const std::vector<cv::Rect>& detections, std::vector<bool>& matchedDetections) {
    for (auto& tracker : trackers) {
        cv::Mat prediction = tracker.kalmanFilter.predict();

        float pred_x  = prediction.at<float>(0);
        float pred_y  = prediction.at<float>(1);
        float pred_w  = prediction.at<float>(2);
        float pred_h  = prediction.at<float>(3);
        float pred_vx = prediction.at<float>(4);
        float pred_vy = prediction.at<float>(5);
        float pred_vw = prediction.at<float>(6);
        float pred_vh = prediction.at<float>(7);

        // Position constraints
        if (pred_x < 0.0f) pred_x = 0.0f;
        if (pred_y < 0.0f) pred_y = 0.0f;

        float max_vw = config.sizeVelocityClamp * std::max(pred_w, config.minBBoxWidth);
        float max_vh = config.sizeVelocityClamp * std::max(pred_h, config.minBBoxHeight);
        pred_vw = std::max(-max_vw, std::min(pred_vw, max_vw));
        pred_vh = std::max(-max_vh, std::min(pred_vh, max_vh));

        pred_w = std::max(config.minBBoxWidth, std::min(pred_w, config.maxBBoxWidth));
        pred_h = std::max(config.minBBoxHeight, std::min(pred_h, config.maxBBoxHeight));

        tracker.velocity     = cv::Point2f(pred_vx, pred_vy);
        tracker.sizeVelocity = cv::Point2f(pred_vw, pred_vh);

        tracker.boundingBox = cv::Rect(static_cast<int>(pred_x),
                                       static_cast<int>(pred_y),
                                       static_cast<int>(pred_w),
                                       static_cast<int>(pred_h));
    }

    for (size_t i = 0; i < trackers.size(); ++i) {
        double bestIoU = 0.0;
        int bestIndex = -1;
        for (size_t j = 0; j < detections.size(); ++j) {
            double iou = computeIoU(config.iouType, trackers[i].boundingBox, detections[j]);
            if (iou > bestIoU && iou > config.iouThreshold) {
                bestIoU = iou;
                bestIndex = static_cast<int>(j);
            }
        }

        if (bestIndex != -1) {
            cv::Mat measurement(4, 1, CV_32F);
            measurement.at<float>(0) = static_cast<float>(detections[bestIndex].x);
            measurement.at<float>(1) = static_cast<float>(detections[bestIndex].y);
            measurement.at<float>(2) = static_cast<float>(detections[bestIndex].width);
            measurement.at<float>(3) = static_cast<float>(detections[bestIndex].height);
            trackers[i].kalmanFilter.correct(measurement);
            trackers[i].boundingBox = detections[bestIndex];
            trackers[i].missedFrames = 0;
            trackers[i].state = TrackedObject::TRACKING;

            // Update trajectory (matches original stalker.cpp)
            trackers[i].lastActualCenter = cv::Point(
                trackers[i].boundingBox.x + trackers[i].boundingBox.width / 2,
                trackers[i].boundingBox.y + trackers[i].boundingBox.height / 2
            );
            trackers[i].trajectory.push_back(trackers[i].lastActualCenter);
            if (trackers[i].trajectory.size() > 35) {
                trackers[i].trajectory.pop_front();
            }

            matchedDetections[bestIndex] = true;
        } else {
            trackers[i].missedFrames++;
            if (config.usePredictionInLost) {
                cv::Mat state = trackers[i].kalmanFilter.statePost;
                trackers[i].boundingBox = cv::Rect(
                    static_cast<int>(state.at<float>(0)),
                    static_cast<int>(state.at<float>(1)),
                    static_cast<int>(state.at<float>(2)),
                    static_cast<int>(state.at<float>(3)));
            }
            trackers[i].state = TrackedObject::LOST;
        }
    }
}

void KalmanIoUByteTrack::addNewTrackers(const std::vector<cv::Rect>& detections, const std::vector<bool>& matchedDetections,
                                        const std::vector<float>& confidences,
                                        const std::vector<cv::Mat>& features) {
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!matchedDetections[i]) {
            TrackedObject newTracker;
            newTracker.id = nextId++;
            newTracker.boundingBox = detections[i];
            newTracker.kalmanFilter = createKalmanFilter();


            // Initialize state (8x1)
            cv::Mat initialState(8, 1, CV_32F);
            initialState.at<float>(0) = static_cast<float>(detections[i].x);
            initialState.at<float>(1) = static_cast<float>(detections[i].y);
            initialState.at<float>(2) = static_cast<float>(detections[i].width);
            initialState.at<float>(3) = static_cast<float>(detections[i].height);
            initialState.at<float>(4) = 0.0f; // vx
            initialState.at<float>(5) = 0.0f; // vy
            initialState.at<float>(6) = 0.0f; // vw
            initialState.at<float>(7) = 0.0f; // vh

            newTracker.kalmanFilter.statePost = initialState;
            newTracker.velocity = cv::Point2f(0.f, 0.f);
            newTracker.sizeVelocity = cv::Point2f(0.f, 0.f);

            // Store detection confidence
            if (!confidences.empty() && i < confidences.size()) {
                newTracker.detectionConfidence = confidences[i];
            }

            // Seed ReID embedding so appearance matching works from the next frame
            if (config.useReId && i < features.size() && !features[i].empty()) {
                newTracker.featureEmbedding = features[i].clone();
            }

            newTracker.color = generateRandomColor();
            newTracker.lastActualCenter = cv::Point(
                detections[i].x + detections[i].width / 2,
                detections[i].y + detections[i].height / 2);
            newTracker.trajectory.push_back(newTracker.lastActualCenter);

            trackers.push_back(newTracker);
        }
    }
}

void KalmanIoUByteTrack::removeLostTrackers() {
    trackers.erase(
        std::remove_if(trackers.begin(), trackers.end(),
                       [this](const TrackedObject& tracker) {
                           bool outOfBounds = config.removeOutOfBounds &&
                               (tracker.boundingBox.x + tracker.boundingBox.width < config.cameraBounds.x ||
                                tracker.boundingBox.y + tracker.boundingBox.height < config.cameraBounds.y ||
                                tracker.boundingBox.x > config.cameraBounds.x + config.cameraBounds.width ||
                                tracker.boundingBox.y > config.cameraBounds.y + config.cameraBounds.height);

                           bool exceededMissedFrames = tracker.missedFrames > config.maxLostFrames;
                           return exceededMissedFrames || outOfBounds;
                       }),
        trackers.end());
}

// ========== Legacy Methods (Deprecated) ==========

std::vector<std::pair<int, cv::Rect>> KalmanIoUByteTrack::processDetections(const std::vector<cv::Rect>& detections) {
    std::vector<bool> matchedDetections(detections.size(), false);
    updateTrackers(detections, matchedDetections);
    addNewTrackers(detections, matchedDetections);
    removeLostTrackers();

    std::vector<std::pair<int, cv::Rect>> results;
    for (const auto& tracker : trackers) {
        results.emplace_back(tracker.id, tracker.boundingBox);
    }
    return results;
}

std::vector<std::pair<int, cv::Rect>> KalmanIoUByteTrack::processDetectionsWithHungarian(const std::vector<cv::Rect>& detections) {
    updateTrackersWithHungarian(detections);

    std::vector<std::pair<int, cv::Rect>> results;
    for (const auto& tracker : trackers) {
        results.emplace_back(tracker.id, tracker.boundingBox);
    }
    return results;
}

std::vector<TrackedObject> KalmanIoUByteTrack::processDetectionsWithHungarianTrackers(const std::vector<cv::Rect>& detections) {
    updateTrackersWithHungarian(detections);
    return trackers;
}

cv::Scalar KalmanIoUByteTrack::generateRandomColor() {
    cv::RNG rng(static_cast<uint64_t>(timeSinceEpochMillisec()));
    return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}

// ========== IObjectTracker Implementation ==========

std::vector<IObjectTracker::TrackedResult> KalmanIoUByteTrack::update(const std::vector<Detection>& detections) {
    if (telemetry) {
        ITrackerTelemetry::FrameMetadata metadata;
        metadata.frameIndex = frameIndexCounter;
        if (frameSizeHint.has_value()) {
            metadata.sourceSize = frameSizeHint->source;
            metadata.processedSize = frameSizeHint->processed;
        }
        telemetry->onFrameStart(metadata);
    }

    // Convert IObjectTracker::Detection to cv::Rect for internal processing
    std::vector<cv::Rect> highConfRects;
    std::vector<float> highConfidences;
    std::vector<cv::Mat> highConfFeatures;
    std::vector<cv::Rect> lowConfRects;
    std::vector<float> lowConfidences;
    std::vector<cv::Mat> lowConfFeatures;
    highConfRects.reserve(detections.size());
    highConfidences.reserve(detections.size());
    highConfFeatures.reserve(detections.size());
    lowConfRects.reserve(detections.size());
    lowConfidences.reserve(detections.size());
    lowConfFeatures.reserve(detections.size());

    const float kHighConfidenceThreshold = config.highConfThreshold;
    const float kLowConfidenceFloor = config.lowConfFloor;

    for (const auto& det : detections) {
        // Validate bounding box
        if (det.bbox.width <= 0 || det.bbox.height <= 0 ||
            det.bbox.x < 0 || det.bbox.y < 0 ||
            det.bbox.area() <= 0) {
            continue;
        }

        if (det.confidence > kHighConfidenceThreshold) {
            highConfRects.push_back(det.bbox);
            highConfidences.push_back(det.confidence);
            highConfFeatures.push_back(det.featureVector);
        } else if (det.confidence > kLowConfidenceFloor && det.confidence <= kHighConfidenceThreshold) {
            lowConfRects.push_back(det.bbox);
            lowConfidences.push_back(det.confidence);
            lowConfFeatures.push_back(det.featureVector);
        }
        // detections with confidence <= kLowConfidenceFloor are ignored
    }

    // Stage 1: High-confidence association (new trackers deferred until after
    // LOST recovery so a reappearing object keeps its old ID)
    std::vector<bool> matchedHigh;
    updateTrackersWithHungarian(highConfRects, highConfidences, -1.0f, false, false,
                                highConfFeatures, &matchedHigh);

    // Stage 2: Unmatched high-confidence detections revive LOST trackers
    // (IoU + ReID, ReID-only fallback)
    {
        std::vector<cv::Rect> remainingHighRects;
        std::vector<float> remainingHighConfs;
        std::vector<cv::Mat> remainingHighFeats;
        std::vector<size_t> remainingHighOrigIdx;
        remainingHighRects.reserve(highConfRects.size());
        remainingHighConfs.reserve(highConfidences.size());
        remainingHighFeats.reserve(highConfFeatures.size());
        remainingHighOrigIdx.reserve(highConfRects.size());
        for (size_t i = 0; i < highConfRects.size(); ++i) {
            if (i < matchedHigh.size() && matchedHigh[i]) {
                continue;
            }
            remainingHighRects.push_back(highConfRects[i]);
            if (i < highConfidences.size()) remainingHighConfs.push_back(highConfidences[i]);
            if (i < highConfFeatures.size()) remainingHighFeats.push_back(highConfFeatures[i]);
            remainingHighOrigIdx.push_back(i);
        }
        if (!remainingHighRects.empty()) {
            std::vector<bool> revived = matchLostTrackers(
                remainingHighRects, remainingHighConfs, remainingHighFeats);
            for (size_t k = 0; k < revived.size(); ++k) {
                if (revived[k]) {
                    matchedHigh[remainingHighOrigIdx[k]] = true;
                }
            }
        }
    }

    // Stage 3: Low-confidence association (revive remaining lost trackers)
    if (!lowConfRects.empty()) {
        matchLostTrackers(lowConfRects, lowConfidences, lowConfFeatures);
    }

    // Stage 4: Open new IDs only for high-confidence detections that matched
    // neither an active nor a LOST tracker
    addNewTrackers(highConfRects, matchedHigh, highConfidences, highConfFeatures);

    // Final cleanup after all stages
    removeLostTrackers();

    // Convert internal trackers to TrackedResult format
    std::vector<TrackedResult> results;
    results.reserve(trackers.size());

    for (const auto& tracker : trackers) {
        TrackedResult result;
        result.trackId = tracker.id;
        result.bbox = tracker.boundingBox;
        result.velocity = tracker.velocity;
        // Use stored detection confidence, fallback to state-based if not available
        result.confidence = tracker.detectionConfidence;
        result.classId = -1; // We don't track class ID in this implementation
        result.isPredicted = (tracker.state == TrackedObject::LOST);
        result.age = tracker.age;
        result.missedFrames = tracker.missedFrames;

        results.push_back(result);
    }

    if (telemetry) {
        for (const auto& result : results) {
            telemetry->onTrackResult(frameIndexCounter, result);
        }
        telemetry->onFrameEnd(frameIndexCounter);
        telemetry->flush();
    }

    // Increment age for all trackers
    for (auto& tracker : trackers) {
        tracker.age++;
    }

    frameIndexCounter++;

    return results;
}

void KalmanIoUByteTrack::reset() {
    trackers.clear();
    nextId = 0;
    frameIndexCounter = 0;
    frameSizeHint.reset();
    if (telemetry) {
        telemetry->flush();
    }
}

int KalmanIoUByteTrack::getActiveTrackCount() const {
    return std::count_if(trackers.begin(), trackers.end(),
        [](const TrackedObject& t) { return t.state == TrackedObject::TRACKING; });
}

int KalmanIoUByteTrack::getTotalTrackCount() const {
    return static_cast<int>(trackers.size());
}

std::optional<IObjectTracker::TrackState> KalmanIoUByteTrack::getTrackState(int trackId) const {
    const TrackedObject* tracker = findTracker(trackId);
    if (!tracker) {
        return std::nullopt;
    }
    return toTrackState(tracker->state);
}

IObjectTracker::TrackState KalmanIoUByteTrack::toTrackState(TrackedObject::State state) const {
    switch (state) {
        case TrackedObject::TRACKING:
            return TrackState::TRACKING;
        case TrackedObject::LOST:
            return TrackState::LOST;
        default:
            return TrackState::DELETED;
    }
}

const TrackedObject* KalmanIoUByteTrack::findTracker(int trackId) const {
    auto it = std::find_if(trackers.begin(), trackers.end(),
        [trackId](const TrackedObject& t) { return t.id == trackId; });
    return (it != trackers.end()) ? &(*it) : nullptr;
}

// ========== IVisualTracker Implementation ==========

void KalmanIoUByteTrack::draw(cv::Mat& frame, const VisualizationConfig& config) {
    for (auto& tracker : trackers) {
        if (!config.showLostTracks && tracker.state == TrackedObject::LOST)
            continue;

        cv::Scalar color = tracker.color;

        // State-based coloring
        if (config.colorByState) {
            color = (tracker.state == TrackedObject::TRACKING) ? config.trackingColor : config.lostColor;
        }

        // Bounding box
        if (config.showBoundingBox) {
            cv::rectangle(frame, tracker.boundingBox, color, config.boundingBoxThickness);
        }

        // Track ID
        if (config.showTrackId) {
            std::string idText = "ID:" + std::to_string(tracker.id);
            cv::Point textPos(tracker.boundingBox.x, tracker.boundingBox.y - 5);
            cv::putText(frame, idText, textPos, cv::FONT_HERSHEY_SIMPLEX,
                       config.trackIdFontScale, config.trackIdColor, config.trackIdThickness);
        }

        // ReID similarity score
        if (config.showReidScore && tracker.lastReidSimilarity >= 0.f) {
            int pct = static_cast<int>(tracker.lastReidSimilarity * 100.f);
            std::string reidText = "R:" + std::to_string(pct) + "%";
            cv::Point reidPos(tracker.boundingBox.x, tracker.boundingBox.y - 20);
            cv::putText(frame, reidText, reidPos, cv::FONT_HERSHEY_SIMPLEX,
                       0.45, cv::Scalar(255, 100, 255), 1);
        }

        // Center point
        if (config.showCenter) {
            cv::Point center(tracker.boundingBox.x + tracker.boundingBox.width / 2,
                           tracker.boundingBox.y + tracker.boundingBox.height / 2);
            tracker.lastActualCenter = center;
            cv::circle(frame, center, config.centerRadius, config.centerColor, cv::FILLED);
        }

        // Velocity/Direction
        if (config.showVelocity) {
            cv::Point startPoint = tracker.lastActualCenter;
            cv::Point endPoint = startPoint + cv::Point(
                static_cast<int>(tracker.velocity.x * config.velocityScale),
                static_cast<int>(tracker.velocity.y * config.velocityScale));
            cv::arrowedLine(frame, startPoint, endPoint, config.velocityColor, config.velocityThickness);
        }

        // Trajectory
        if (config.showTrajectory) {
            // Update trajectory
            tracker.trajectory.push_back(tracker.lastActualCenter);
            if (static_cast<int>(tracker.trajectory.size()) > config.trajectoryLength) {
                tracker.trajectory.pop_front();
            }

            // Draw trajectory
            for (size_t i = 1; i < tracker.trajectory.size(); ++i) {
                cv::line(frame, tracker.trajectory[i - 1], tracker.trajectory[i],
                        config.trajectoryColor, config.trajectoryThickness);
            }
        }

        // Confidence
        if (config.showConfidence) {
            float conf = (tracker.state == TrackedObject::TRACKING) ? 0.9f : 0.5f;
            std::string confText = "Conf:" + std::to_string(conf).substr(0, 4);
            cv::Point textPos(tracker.boundingBox.x, tracker.boundingBox.y + tracker.boundingBox.height + 15);
            cv::putText(frame, confText, textPos, cv::FONT_HERSHEY_SIMPLEX,
                       0.4, config.confidenceColor, 1);
        }

        // Age
        if (config.showAge) {
            std::string ageText = "Age:" + std::to_string(tracker.age);
            cv::Point textPos(tracker.boundingBox.x + tracker.boundingBox.width - 60,
                            tracker.boundingBox.y - 5);
            cv::putText(frame, ageText, textPos, cv::FONT_HERSHEY_SIMPLEX,
                       0.4, cv::Scalar(255, 255, 255), 1);
        }

        // Missed frames
        if (config.showMissedFrames && tracker.missedFrames > 0) {
            std::string missedText = "Miss:" + std::to_string(tracker.missedFrames);
            cv::Point textPos(tracker.boundingBox.x, tracker.boundingBox.y + tracker.boundingBox.height + 30);
            cv::putText(frame, missedText, textPos, cv::FONT_HERSHEY_SIMPLEX,
                       0.4, cv::Scalar(0, 0, 255), 1);
        }
    }
}

std::vector<std::vector<cv::Point>> KalmanIoUByteTrack::getTrajectories() const {
    std::vector<std::vector<cv::Point>> trajectories;
    trajectories.reserve(trackers.size());

    for (const auto& tracker : trackers) {
        std::vector<cv::Point> traj(tracker.trajectory.begin(), tracker.trajectory.end());
        trajectories.push_back(traj);
    }

    return trajectories;
}

std::vector<cv::Point> KalmanIoUByteTrack::getTrajectory(int trackId) const {
    const TrackedObject* tracker = findTracker(trackId);
    if (!tracker) {
        return {};
    }
    return std::vector<cv::Point>(tracker->trajectory.begin(), tracker->trajectory.end());
}

void KalmanIoUByteTrack::clearTrajectories() {
    for (auto& tracker : trackers) {
        tracker.trajectory.clear();
    }
}

void KalmanIoUByteTrack::setTrajectoryLength(int length) {
    config.maxTrajectoryLength = length;
}

// ========== IConfigurableTracker Implementation ==========

KalmanIoUConfig KalmanIoUByteTrack::getConfig() const {
    return config;
}

void KalmanIoUByteTrack::setConfig(const KalmanIoUConfig& newConfig) {
    if (validateConfig(newConfig)) {
        config = newConfig;
    } else {
        throw std::invalid_argument("Invalid configuration");
    }
}

void KalmanIoUByteTrack::setParameter(const std::string& key, const std::any& value) {
    if (key == "iouType") {
        config.iouType = std::any_cast<KalmanIoUConfig::IoUType>(value);
    } else if (key == "iouThreshold") {
        config.iouThreshold = std::any_cast<float>(value);
    } else if (key == "usePredictionInLost") {
        config.usePredictionInLost = std::any_cast<bool>(value);
    } else if (key == "maxLostFrames") {
        config.maxLostFrames = std::any_cast<int>(value);
    } else if (key == "removeOutOfBounds") {
        config.removeOutOfBounds = std::any_cast<bool>(value);
    } else if (key == "cameraBounds") {
        config.cameraBounds = std::any_cast<cv::Rect>(value);
    } else if (key == "processNoise") {
        config.processNoise = std::any_cast<float>(value);
    } else if (key == "measurementNoise") {
        config.measurementNoise = std::any_cast<float>(value);
    } else if (key == "errorCovPost") {
        config.errorCovPost = std::any_cast<float>(value);
    } else if (key == "maxTrajectoryLength") {
        config.maxTrajectoryLength = std::any_cast<int>(value);
    } else if (key == "softIoUSlack") {
        config.softIoUSlack = std::any_cast<float>(value);
    } else if (key == "alphaIoUExponent") {
        config.alphaIoUExponent = std::any_cast<float>(value);
    } else {
        throw std::invalid_argument("Unknown parameter key: " + key);
    }
}

std::any KalmanIoUByteTrack::getParameter(const std::string& key) const {
    if (key == "iouType") {
        return config.iouType;
    } else if (key == "iouThreshold") {
        return config.iouThreshold;
    } else if (key == "usePredictionInLost") {
        return config.usePredictionInLost;
    } else if (key == "maxLostFrames") {
        return config.maxLostFrames;
    } else if (key == "removeOutOfBounds") {
        return config.removeOutOfBounds;
    } else if (key == "cameraBounds") {
        return config.cameraBounds;
    } else if (key == "processNoise") {
        return config.processNoise;
    } else if (key == "measurementNoise") {
        return config.measurementNoise;
    } else if (key == "errorCovPost") {
        return config.errorCovPost;
    } else if (key == "maxTrajectoryLength") {
        return config.maxTrajectoryLength;
    } else if (key == "softIoUSlack") {
        return config.softIoUSlack;
    } else if (key == "alphaIoUExponent") {
        return config.alphaIoUExponent;
    } else {
        throw std::invalid_argument("Unknown parameter key: " + key);
    }
}

bool KalmanIoUByteTrack::validateConfig(const KalmanIoUConfig& cfg) const {
    if (cfg.iouThreshold < 0.0f || cfg.iouThreshold > 1.0f) {
        return false;
    }
    if (cfg.maxLostFrames < 0) {
        return false;
    }
    if (cfg.processNoise < 0.0f || cfg.measurementNoise < 0.0f || cfg.errorCovPost < 0.0f) {
        return false;
    }
    if (cfg.maxTrajectoryLength < 1) {
        return false;
    }
    if (cfg.softIoUSlack <= 0.0f) {
        return false;
    }
    if (cfg.alphaIoUExponent <= 0.0f) {
        return false;
    }
    return true;
}

void KalmanIoUByteTrack::setTelemetry(const std::shared_ptr<ITrackerTelemetry>& telemetryLogger) {
    telemetry = telemetryLogger;
}

void KalmanIoUByteTrack::clearTelemetry() {
    telemetry.reset();
}

void KalmanIoUByteTrack::setFrameSizeHint(const cv::Size& sourceSize, const cv::Size& processedSize) {
    frameSizeHint = FrameSizeInfo{sourceSize, processedSize};
}

void KalmanIoUByteTrack::clearFrameSizeHint() {
    frameSizeHint.reset();
}
